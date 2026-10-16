"""One evaluated own-model round using both existing native physical nodes."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import threading
import time
import uuid

from desktop.backend import NodeController
from src.learning.swarm_distributed import (WORKERS,GRADIENT_OPERATION,CANDIDATE_OPERATION,
    policy_id,verify_contributions,definition,verdict)
from src.network.client_state import NodeStateLock,ClientSettings
from src.network.public_knowledge import managed_public_bytes
from src.network.telemetry import atomic_json
from .prepare_public_math import fingerprints,outside_checkout
from .research_artifact import retained_bytes


def run(args,cancel):
    node=args.node_directory.resolve();directory=args.directory.resolve()
    if (directory.exists() or not outside_checkout(directory) or not outside_checkout(node)
            or not directory.is_relative_to(Path.home()/'.local/state/lixyswarm-own-foundation-20261015')
            or type(args.start) is not int or type(args.count) is not int
            or not 1<=args.count<=256 or not 0<=args.start<=4096-2*args.count):
        raise ValueError('Own round needs fresh accounted state and two bounded disjoint document ranges')
    controller=NodeController(node);initial=controller.snapshot();diagnostic=initial['diagnostics']
    own=diagnostic.get('own_swarm',{})
    if (not initial['fresh'] or not own.get('ready') or diagnostic['node_id'] not in WORKERS
            or own.get('training_policy_id')!=policy_id()):
        raise ValueError('The existing physical node must have its own native learner ready')
    owner=diagnostic['node_id'];pid=diagnostic['pid'];before=fingerprints(node)
    selected=(node/'own-swarm.json').read_bytes();deadline=time.monotonic()+600
    report={'success':False,'training_policy_id':policy_id(),'native_pid':pid,'requester_id':owner,
        'state_id':own['state_id'],'model_id':own['model_id'],'contributions':[],
        'shards':[{'worker_id':worker,'start':args.start+index*args.count,'count':args.count}
            for index,worker in enumerate(WORKERS)],'started_at':time.time(),
        'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'new_shadow_requester':False,'new_release_published':False,'native_autonomous_cycles_added':0}
    directory.mkdir(parents=True,mode=0o700)
    requests=[]
    def checkpoint():atomic_json(directory/'report.json',report)
    def allowed():
        now=controller.snapshot();policy=ClientSettings.load(node).policy
        return (not cancel.is_set() and time.monotonic()<deadline and now['fresh']
            and now['diagnostics']['pid']==pid and fingerprints(node)==before
            and (node/'own-swarm.json').read_bytes()==selected
            and now['diagnostics'].get('own_swarm',{}).get('state_id')==own['state_id']
            and managed_public_bytes(node)+retained_bytes()+.08*1024**3<=policy.max_disk_gb*1024**3)
    def start(worker,operation,payload):
        if not allowed():raise RuntimeError('Native own-round state, storage or permissions changed')
        request=uuid.uuid4().hex
        controller._chat_call('/own-work/start',{'request_id':request,'worker_id':worker,
            'operation':operation,'payload':payload},timeout=4)
        requests.append(request);return request
    def wait(request):
        while allowed():
            row=controller._chat_call('/own-work/'+request,timeout=4)
            if row['state']!='running':
                if row['state']!='complete':raise RuntimeError('Native own work failed: '+row.get('error_type',row['state']))
                return row
            cancel.wait(1)
        raise RuntimeError('Native own work expired or its ownership changed')
    common={k:report[k] for k in ['state_id','model_id','training_policy_id']}
    from src.learning.public_math import ARTIFACTS
    common['dataset_id']=ARTIFACTS['train'][0]
    checkpoint()
    try:
        with NodeStateLock(node/'research/own-round'):
            pending=[(row,start(row['worker_id'],GRADIENT_OPERATION,
                {**common,'start':row['start'],'count':row['count']})) for row in report['shards']]
            # Poll both owner leases, including while either physical worker is
            # slower, so the other lease never expires because of this script.
            completed={}
            while len(completed)<2:
                if not allowed():raise RuntimeError('Own round ownership or storage changed')
                for row,request in pending:
                    status=controller._chat_call('/own-work/'+request,timeout=4)
                    if status['state'] not in {'running','complete'}:
                        raise RuntimeError('Native own gradient failed: '+status.get('error_type',status['state']))
                    if status['state']=='complete' and request not in completed:
                        completed[request]=status
                        report['contributions'].append({'worker_id':row['worker_id'],'result':status['result']})
                        report.setdefault('work',[]).append(status);checkpoint()
                cancel.wait(.5)
            verify_contributions(report['contributions'],model_id=own['model_id'],
                training_state_id=own['state_id'],requester_id=owner)
            remote=next(row for row in report['work'] if row['route']['transport']!='local')
            if (not remote['route']['public_internet'] or not remote['route']['encrypted']
                    or remote['artifact_initially_local'] or remote['encoded_data_bytes']<=0):
                raise ValueError('The remote own gradient did not arrive freshly over public encrypted LSP')
            candidate=wait(start(owner,CANDIDATE_OPERATION,{**common,'contributions':report['contributions']}))
            value=candidate['result']['output']
            if value['accepted']!=verdict(value['before'],value['after']):
                raise ValueError('The native own candidate does not meet its declared evaluation gate')
            report.update(success=True,candidate=candidate,accepted=value['accepted'],
                optimizer_step_before=value['previous_optimizer_step'],optimizer_step_after=value['candidate_optimizer_step'],
                native_controls_unchanged=fingerprints(node)==before,finished_at=time.time(),
                scope=definition()['scope']);checkpoint();return report
    except BaseException as error:
        report.update(error_type=type(error).__name__,error=str(error)[:1024],finished_at=time.time());checkpoint();raise
    finally:
        for request in requests:
            try:controller._chat_call('/own-work/cancel',{'request_id':request},timeout=2)
            except (OSError,ValueError,RuntimeError):pass


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--node-directory',type=Path,required=True)
    parser.add_argument('--start',type=int,default=0);parser.add_argument('--count',type=int,default=128)
    args=parser.parse_args();cancel=threading.Event()
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:cancel.set())
    result=run(args,cancel)
    print(json.dumps({k:v for k,v in result.items() if k not in {'work','contributions','candidate'}},allow_nan=False),flush=True)


if __name__=='__main__':main()
