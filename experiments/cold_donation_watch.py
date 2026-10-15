"""Request one native public gradient when the physical cold trial is ready.

SSH carries administrative state and compact signed evidence only. The native
requester and worker exchange all dataset/gradient bytes over public LSP.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shlex
import signal
import subprocess
import time
import uuid

from desktop.backend import NodeController
from src.learning.language_policy import dataset_id,policy_id
from src.network.telemetry import atomic_json
from .cold_native_trial import LAPTOP,PROVIDER,ORIGINAL,TRIAL,verify_donation
from .prepare_public_math import fingerprints,outside_checkout


def run(args):
    if args.directory.exists() or not outside_checkout(args.directory):raise ValueError('Use a fresh external evidence directory')
    controller=NodeController(args.node_directory);snapshot=controller.snapshot();policy=asdict(controller.settings().policy)
    if not snapshot['fresh'] or snapshot['diagnostics'].get('node_id')!=PROVIDER:raise ValueError('Only the physical desktop can request this pilot work')
    before=fingerprints(args.node_directory);pid=snapshot['diagnostics']['pid'];cancelled=False
    def cancel(*_):
        nonlocal cancelled
        cancelled=True
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,cancel)
    args.directory.mkdir(mode=0o700,parents=True)
    report={'success':False,'scope':'Native cold-prepared physical laptop donation and restoration',
        'extra_physical_participant':False,'model_promoted':False,'started_at':time.time()}
    ssh=['ssh','-i',str(args.ssh_key),'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
        '-o','ConnectTimeout=8','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2','toxxy@100.84.3.4']
    def admin(command,stdin=None):
        completed=subprocess.run([*ssh,command],input=stdin,capture_output=True,text=True,timeout=30)
        if completed.returncode:raise RuntimeError('Administrative laptop access temporarily unavailable')
        if len(completed.stdout)>256*1024:raise ValueError('Administrative evidence exceeds its bound')
        return json.loads(completed.stdout) if completed.stdout.strip() else None
    def peek():
        command=shlex.join([str(args.remote_python),'-m','experiments.cold_native_trial','peek','--session',str(args.session)])
        return admin('cd '+shlex.quote(str(args.remote_code))+' && '+command)
    def allowed():
        now=controller.settings();live=controller.snapshot()
        if (cancelled or not now.connected or now.paused or asdict(now.policy)!=policy or fingerprints(args.node_directory)!=before
                or not live['fresh'] or live['diagnostics'].get('pid')!=pid):raise RuntimeError('Desktop participation permissions or native process changed')
    end=time.monotonic()+6*3600;ready=None;identifier=None
    try:
        while time.monotonic()<end:
            allowed()
            try:state=peek()
            except (OSError,RuntimeError,subprocess.TimeoutExpired):
                time.sleep(20);continue
            atomic_json(args.directory/'status.json',state)
            if state['ready'] is not None and state['units'][TRIAL]['MainPID']!='0':ready=state['ready'];break
            if state['units'][TRIAL]['ActiveState'] in {'inactive','failed'}:raise RuntimeError('Cold trial ended before becoming ready')
            time.sleep(20)
        if ready is None or ready['physical_id']!=LAPTOP:raise TimeoutError('Physical laptop did not finish native preparation')
        payload={'base_name':'qwen3-1.7b','release_id':ready['release_id'],'adapter_id':ready['model_id'],
            'dataset_id':dataset_id(4),'training_policy_id':policy_id(4),'start':2048,'count':2}
        identifier=uuid.uuid4().hex
        controller._chat_call('/language-work/start',{'request_id':identifier,'worker_id':LAPTOP,
            'operation':'language.gradient.v5','payload':payload},timeout=10)
        request_end=time.monotonic()+1200
        while time.monotonic()<request_end:
            allowed();donation=controller._chat_call('/language-work/'+identifier)
            if donation['state']=='complete':break
            if donation['state']!='running':raise RuntimeError('New native donation failed: '+donation.get('error_type',donation['state']))
            time.sleep(1)
        else:raise TimeoutError('Native donation did not complete within its bound')
        donation['payload']=payload
        verify_donation({'model_id':ready['model_id'],'planned_inventory':{'heads':{'qwen3-1.7b':ready['release_id']}}},donation,ready['prepared_at'])
        atomic_json(args.directory/'donation.json',donation)
        # This record contains a signed result and hashes, not the gradient body.
        code=('import json,os,sys,tempfile;from pathlib import Path;p=Path(sys.argv[1]);'
            'raw=sys.stdin.buffer.read(262145);assert len(raw)<=262144;json.loads(raw);'
            'fd,name=tempfile.mkstemp(prefix=".donation-",dir=p.parent);'
            'f=os.fdopen(fd,"wb");f.write(raw);f.flush();os.fsync(f.fileno());f.close();os.replace(name,p)')
        admin(shlex.join(['python3','-c',code,str(args.session.parent/'donation.json')]),json.dumps(donation))
        report.update(donation_verified=True,request_id=identifier,physical_worker=LAPTOP,physical_requester=PROVIDER,
            release_id=ready['release_id'],model_id=ready['model_id'],contribution=donation)
        atomic_json(args.directory/'report.json',report)
        # Recovery can need hours if the original's second model family is cold.
        end=time.monotonic()+24*3600
        while time.monotonic()<end and not cancelled:
            try:state=peek()
            except (OSError,RuntimeError,subprocess.TimeoutExpired):time.sleep(20);continue
            atomic_json(args.directory/'status.json',state)
            recovered=state['recovered'];old=state['original']
            if (recovered and old['fresh'] and str(old['pid'])==state['units'][ORIGINAL]['MainPID']
                    and old['node_id']==LAPTOP and old['language']['ready'] and old['language_training']['ready']):
                report.update(success=True,original_restored=True,recovery=recovered,finished_at=time.time());break
            time.sleep(20)
        else:raise TimeoutError('Independent recovery has not yet restored the original native participant')
    except Exception as error:
        report.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        if identifier is not None:
            try:controller._chat_call('/language-work/cancel',{'request_id':identifier})
            except Exception:pass
        report['desktop_controls_unchanged']=fingerprints(args.node_directory)==before
        atomic_json(args.directory/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['directory','node-directory','ssh-key','remote-python','remote-code','session']:
        parser.add_argument('--'+name,type=Path,required=True)
    print(json.dumps(run(parser.parse_args()),allow_nan=False),flush=True)


if __name__=='__main__':main()
