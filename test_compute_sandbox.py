"""Exercise the kernel boundary, including a hostile worker and its children."""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest
import psutil

from src.network.compute_process import ComputeProcessError, ComputeSupervisor
from src.network.compute_sandbox import CPUComputeSandbox, ComputeSandboxError
from experiments.qualify_compute_sandbox import verify_fixture_result


@pytest.fixture
def kernel_sandbox(tmp_path):
    if sys.platform != 'linux' or shutil.which('bwrap') is None:
        pytest.skip('Linux bubblewrap boundary requires a supported local runtime')
    runtime = tmp_path/'runtime'; runtime.mkdir()
    output = tmp_path/'job'; output.mkdir()
    sandbox = CPUComputeSandbox((runtime,), (), output)
    command, env = sandbox.launch(['/usr/bin/true'], {'device':'cpu'})
    probe = subprocess.run(command,env=env,capture_output=True,timeout=5)
    if probe.returncode:
        pytest.skip('Kernel namespace policy does not support the required sandbox')
    return runtime, output, sandbox


def run(sandbox, command, *, allowed=lambda:True):
    execution=SimpleNamespace(raise_if_cancelled=lambda:None, deadline=time.time()+10)
    return ComputeSupervisor().run(command,{'device':'cpu'},execution,allowed=allowed,ram_gb=.4,sandbox=sandbox)


def test_kernel_blocks_private_files_network_pid_access_and_input_writes(kernel_sandbox, tmp_path, monkeypatch):
    runtime, output, _ = kernel_sandbox
    private=tmp_path/'operator-private'; private.write_text('local fixture secret')
    data=tmp_path/'public-input'; data.write_text('verified public data')
    monkeypatch.setenv('LIXY_SANDBOX_CANARY','local fixture environment secret')
    listener=socket.socket(); listener.bind(('127.0.0.1',0)); listener.listen()
    script=runtime/'worker.py'
    script.write_text('''import json,os,socket,sys
from pathlib import Path
public,private,output,port,parent=sys.argv[1:]
def denied(operation):
    try: operation(); return False
    except OSError: return True
with socket.socket() as client:
    client.settimeout(.2)
    result={'input':Path(public).read_text(),
        'input_readonly':denied(lambda:Path(public).write_text('changed')),
        'private_hidden':denied(lambda:Path(private).read_text()),
        'parent_hidden':not Path('/proc/'+parent+'/root').exists(),
        'network_blocked':denied(lambda:client.connect(('127.0.0.1',int(port)))),
        'environment_clean':'LIXY_SANDBOX_CANARY' not in os.environ,
        'gpu_hidden':not any(Path('/dev').glob('nvidia*'))}
Path(output,'result.json').write_text(json.dumps(result))
print(json.dumps(result))
''')
    sandbox=CPUComputeSandbox((runtime,), (data,), output)
    try:
        value=run(sandbox,[sys.executable,str(script),str(data),str(private),str(output),
            str(listener.getsockname()[1]),str(os.getpid())])
    finally:
        listener.close()
    assert value == {'input':'verified public data', 'input_readonly':True,'private_hidden':True,
        'parent_hidden':True,'network_blocked':True,'environment_clean':True,'gpu_hidden':True}
    assert json.loads((output/'result.json').read_text())==value
    assert private.read_text()=='local fixture secret' and data.read_text()=='verified public data'


def test_kernel_confines_children_and_symlinks(kernel_sandbox, tmp_path):
    runtime, output, sandbox=kernel_sandbox
    secret=tmp_path/'unmounted'; secret.write_text('private fixture')
    (runtime/'escape').symlink_to(secret)
    script=runtime/'worker.py'
    script.write_text('''import json,subprocess,sys
from pathlib import Path
child=subprocess.run([sys.executable,'-c',"import sys; from pathlib import Path; print(Path(sys.argv[1]).exists())",sys.argv[1]],capture_output=True,text=True,check=True)
print(json.dumps({'child_cannot_follow_host_symlink':child.stdout.strip()=='False'}))
''')
    assert run(sandbox,[sys.executable,str(script),str(runtime/'escape')]) == {'child_cannot_follow_host_symlink':True}


def test_required_isolation_never_falls_back_to_running_host_command(kernel_sandbox, tmp_path, monkeypatch):
    runtime, output, sandbox=kernel_sandbox
    script=runtime/'worker.py'; script.write_text("from pathlib import Path; Path('executed').touch()")
    monkeypatch.setattr('src.network.compute_sandbox.shutil.which',lambda *a,**kw:None)
    with pytest.raises(ComputeProcessError,match='isolation'):
        run(sandbox,[sys.executable,str(script)])
    assert not (output/'executed').exists()


def test_supervisor_rejects_peer_supplied_sandbox_settings(tmp_path):
    with pytest.raises(ComputeProcessError,match='locally'):
        run({'runtime_roots':['/']},[sys.executable,'-c','print("{}")'])


def test_mount_policy_rejects_broad_alias_overlapping_or_cuda_access(kernel_sandbox, tmp_path):
    runtime, output, sandbox=kernel_sandbox
    for bad in [CPUComputeSandbox((Path.home(),),(),output),
                CPUComputeSandbox((tmp_path,),(),output),
                CPUComputeSandbox((runtime,), (output/'missing',), output)]:
        with pytest.raises((ComputeSandboxError,OSError)):
            bad.launch([sys.executable],{'device':'cpu'})
    alias=tmp_path/'alias'; alias.symlink_to(runtime,target_is_directory=True)
    with pytest.raises(ComputeSandboxError,match='canonical'):
        CPUComputeSandbox((alias,),(),output).launch([sys.executable],{'device':'cpu'})
    with pytest.raises(ComputeSandboxError,match='CPU'):
        sandbox.launch([sys.executable],{'device':'cuda'})


def test_bubblewrap_setup_failure_cannot_execute_command_unsandboxed(kernel_sandbox):
    runtime, output, sandbox=kernel_sandbox
    script=runtime/'worker.py'; script.write_text("from pathlib import Path; Path('executed').touch()")
    # A missing argument deliberately makes namespace setup fail before exec.
    command,env=sandbox.launch([sys.executable,str(script)],{'device':'cpu'})
    command[command.index('--unshare-user')]='--nonexistent-required-namespace'
    failed=subprocess.run(command,env=env,capture_output=True,timeout=5)
    assert failed.returncode != 0 and not (output/'executed').exists()


def test_revocation_terminates_the_sandbox_and_its_child(kernel_sandbox):
    runtime, output, sandbox=kernel_sandbox
    script=runtime/'worker.py'
    script.write_text('''import subprocess,sys,time
from pathlib import Path
subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)',sys.argv[1]])
Path('started').touch()
time.sleep(60)
''')
    permission=threading.Event(); permission.set(); errors=[]
    def work():
        try:run(sandbox,[sys.executable,str(script),str(output)],allowed=permission.is_set)
        except ComputeProcessError as error:errors.append(error)
    thread=threading.Thread(target=work);thread.start()
    try:
        until=time.monotonic()+5
        while not (output/'started').exists() and time.monotonic()<until:time.sleep(.01)
        assert (output/'started').exists()
        owned=[]
        for process in psutil.Process().children(recursive=True):
            try:
                if str(output) in process.cmdline():owned.append(process)
            except psutil.Error:pass
        assert len(owned)>=2
        permission.clear();thread.join(4)
        assert not thread.is_alive() and errors
        assert all(not p.is_running() or p.status()==psutil.STATUS_ZOMBIE for p in owned)
    finally:
        permission.clear();thread.join(4)


def test_cpu_fixture_formats_do_not_invent_candidate_evaluation_for_old_engine():
    original={'success':True,'frozen':True,'torch_version':'2.12.0+cpu','cuda_runtime':False,
        'inference_logits':256,'gradient_tokens':64,'personal_state_loaded':False,'network_identity_loaded':False}
    assert verify_fixture_result(original)['candidate_evaluation_exercised'] is False
    expanded={k:v for k,v in original.items() if k!='network_identity_loaded'}
    expanded.update(candidate_evaluated=True,candidate_signed_or_promoted=False,candidate_loss_before=.9,
        candidate_loss_after=.8,test_fixture_signers=3,job_process_signing_keys_loaded=False)
    assert verify_fixture_result(expanded)['candidate_evaluation_exercised'] is True
    for value in [{**original,'network_identity_loaded':True},{**original,'gradient_tokens':True},
                  {**original,'unexpected_peer_claim':True},{**expanded,'candidate_signed_or_promoted':True},
                  {**expanded,'job_process_signing_keys_loaded':True}]:
        with pytest.raises(ValueError):verify_fixture_result(value)
