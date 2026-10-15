import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from src.network.compute_process import ComputeProcessError, ComputeSupervisor
from src.network.reference_compute import EngineSettings


def worker(tmp_path,source):
    path=tmp_path/"engine.py"; path.write_text(source)
    return [sys.executable,str(path)]


def execution():
    return SimpleNamespace(raise_if_cancelled=lambda:None,deadline=time.time()+10)


def test_process_returns_bounded_json_without_loading_engine_in_parent(tmp_path):
    command=worker(tmp_path,"import json,sys; request=json.load(sys.stdin); print(json.dumps({'device':request['device']}))")
    result=ComputeSupervisor().run(command,{"device":"cpu"},execution(),allowed=lambda:True,ram_gb=0.2)
    assert result=={"device":"cpu"}


def test_permission_withdrawal_kills_only_the_owned_engine(tmp_path):
    pid_file=tmp_path/"owned-pid"
    command=worker(tmp_path,f"import os,time; from pathlib import Path; Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(60)")
    allowed=threading.Event(); allowed.set(); done=threading.Event()
    failures=[]
    def run():
        try:
            ComputeSupervisor().run(command,{},execution(),allowed=allowed.is_set,ram_gb=0.2)
        except ComputeProcessError as error:
            failures.append(str(error))
        finally:
            done.set()
    thread=threading.Thread(target=run); thread.start()
    until=time.monotonic()+3
    while not pid_file.exists() and time.monotonic()<until: time.sleep(0.01)
    assert pid_file.exists()
    pid=int(pid_file.read_text()); assert psutil.pid_exists(pid)
    allowed.clear(); assert done.wait(2); thread.join()
    assert not psutil.pid_exists(pid) and failures


def test_memory_and_output_limits_do_not_accept_a_result(tmp_path):
    command=worker(tmp_path,"import time; value=bytearray(150*1024**2); time.sleep(60)")
    with pytest.raises(ComputeProcessError,match="RAM lease"):
        ComputeSupervisor().run(command,{},execution(),allowed=lambda:True,ram_gb=0.04)
    command=worker(tmp_path,"print('x'*300000)")
    with pytest.raises(ComputeProcessError):
        ComputeSupervisor().run(command,{},execution(),allowed=lambda:True,ram_gb=0.2)


def test_revoked_permission_does_not_launch_even_a_fast_engine(tmp_path):
    marker=tmp_path/'started'
    command=worker(tmp_path,f"from pathlib import Path; Path({str(marker)!r}).touch(); print('{{}}')")
    with pytest.raises(ComputeProcessError,match='permission'):
        ComputeSupervisor().run(command,{},execution(),allowed=lambda:False,ram_gb=.2)
    assert not marker.exists()


def test_engine_command_is_machine_local_and_unknown_fields_are_rejected(tmp_path):
    with pytest.raises(ValueError,match="locally installed"):
        EngineSettings(enabled=True,command=["downloaded-peer-command"]).save(tmp_path)
    EngineSettings(enabled=True,command=[sys.executable,str(tmp_path/"engine.py")]).save(tmp_path)
    assert EngineSettings.load(tmp_path).enabled
    path=tmp_path/"compute-engine.json"; value=json.loads(path.read_text()); value["peer_program"]="arbitrary"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match="schema"):
        EngineSettings.load(tmp_path)


def test_old_engine_settings_keep_devices_and_cpu_packages_cannot_claim_cuda(tmp_path):
    EngineSettings(enabled=True,command=[sys.executable],devices=["cpu"]).save(tmp_path)
    assert EngineSettings.load(tmp_path).devices==["cpu"]
    path=tmp_path/"compute-engine.json"; value=json.loads(path.read_text()); value.pop("devices")
    path.write_text(json.dumps(value))
    assert EngineSettings.load(tmp_path).devices==["cpu","cuda"]
    with pytest.raises(ValueError,match="devices"):
        EngineSettings(devices=["cpu","cpu"]).validate()
