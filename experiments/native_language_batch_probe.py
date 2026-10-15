"""Fixed public batches using the physical node's actual local/remote execution path."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import ipaddress
import json
from pathlib import Path
import signal
import threading
import time
import urllib.request
import uuid

from src.learning.language_policy import BATCH_POLICY_ID
from src.learning.public_conversation import public_corpus,corpus_digest
from src.network.telemetry import atomic_json
from src.network.work_protocol import WorkResult,ResultReceipt
from src.release.language_versions import LanguageVersionRegistry,WORKERS,verify_contributions
from .language_gradient_probe import shard_plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--node-directory',type=Path,required=True)
    parser.add_argument('--worker',action='append',required=True)
    parser.add_argument('--worker-slots',action='append',default=[])
    parser.add_argument('--count',type=int,default=64)
    parser.add_argument('--shards',type=int,default=3)
    parser.add_argument('--parallel-jobs',type=int,default=3)
    parser.add_argument('--candidate',action='store_true')
    parser.add_argument('--learning-rate',type=float,default=.0001)
    args=parser.parse_args();repo=Path(__file__).resolve().parents[1]
    if (args.directory.exists() or args.directory.resolve().is_relative_to(repo)
            or not set(args.worker).issubset(WORKERS) or not 1<=args.parallel_jobs<=4
            or not 1e-6<=args.learning_rate<=1e-3):
        parser.error('Use fresh private state, known workers and bounded local requests')
    plan=shard_plan(args.worker,args.count,args.shards,policy_version=2)
    slots={worker:1 for worker in args.worker}
    for item in args.worker_slots:
        worker,_,value=item.partition('=')
        if worker not in slots or value not in {'1','2','3','4'}:parser.error('Use known NODE_ID=1..4 job limits')
        slots[worker]=int(value)
    diagnostic=json.loads((args.node_directory/'diagnostics.json').read_text());own=diagnostic['node_id']
    if own not in args.worker:parser.error('The actual physical requester must be one of the compared workers')
    head=LanguageVersionRegistry(args.node_directory/'public/language','qwen3-1.7b').head()
    history=LanguageVersionRegistry(args.node_directory/'public/language','qwen3-1.7b').history(head['release']['release_id'])
    api=json.loads((args.node_directory/'local-api.json').read_text())
    cancel=threading.Event()
    for signum in [signal.SIGTERM,signal.SIGINT]:signal.signal(signum,lambda *_:cancel.set())
    def request(path,value=None):
        data=None if value is None else json.dumps(value,allow_nan=False).encode()
        req=urllib.request.Request(f'http://127.0.0.1:{api["port"]}'+path,data=data,
            headers={'Authorization':'Bearer '+api['token'],'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=5) as response:
            raw=response.read(512*1024+1)
        if len(raw)>512*1024:raise ValueError('Native work response exceeds its bound')
        return json.loads(raw)
    def execute(worker,operation,payload):
        identifier=uuid.uuid4().hex
        request('/language-work/start',{'request_id':identifier,'worker_id':worker,'operation':operation,'payload':payload})
        try:
            while not cancel.is_set():
                state=request('/language-work/'+identifier)
                if state['state']=='complete':return state
                if state['state']!='running':raise RuntimeError('Native work failed: '+state.get('error_type',state['state']))
                cancel.wait(.2)
            raise RuntimeError('Native batch trial cancelled')
        finally:
            try:request('/language-work/cancel',{'request_id':identifier})
            except OSError:pass
    args.directory.mkdir(mode=0o700,parents=True)
    report={'success':False,'operation':'language.gradient.v3','training_policy_version':2,'base_name':'qwen3-1.7b',
        'head':head,'history':history,'physical_workers':args.worker,'requester_id':own,'native_physical_requester':True,
        'requester_is_extra_physical_worker':False,'contributions':[],'signed_results':[],'started_at':time.time(),
        'receive_bandwidth_mbps':min(5,diagnostic['settings']['policy']['max_bandwidth_mbps']),
        'worker_slot_limits':slots,'candidate_promoted':False,'scope':'Native local artifacts reused; only remote contributions traverse public LSP; no publisher invoked'}
    report['learning_rate']=args.learning_rate
    report['shard_plan']=[{'worker_id':worker,'start':index*args.count,'count':args.count} for index,worker in plan]
    path=args.directory/'report.json';atomic_json(path,report)
    semaphores={worker:threading.BoundedSemaphore(count) for worker,count in slots.items()}
    def contribute(index,worker):
        payload={'base_name':'qwen3-1.7b','release_id':head['release']['release_id'],'adapter_id':head['release']['model_artifact_id'],
            'dataset_id':corpus_digest(public_corpus()),'start':index*args.count,'count':args.count,'training_policy_id':BATCH_POLICY_ID}
        with semaphores[worker]:
            if cancel.is_set():raise RuntimeError('Native work cancelled before dispatch')
            state=execute(worker,'language.gradient.v3',payload)
        result=WorkResult.from_dict(state['result'])
        if not ResultReceipt.from_dict(result.receipt).verify(result,expected_worker_id=worker,expected_requester_id=own):
            raise ValueError('Native result has another physical owner')
        verify_contributions([result.to_dict()],base_name='qwen3-1.7b',adapter_id=payload['adapter_id'],
            dataset_id=payload['dataset_id'],require_distributed=False,policy_version=2)
        route=state['route']
        if worker!=own and (not route['encrypted'] or not ipaddress.ip_address(route.get('relay_host') or route['host']).is_global):
            raise ValueError('Remote native work must traverse encrypted public Internet')
        row={**result.output,'receipt':result.receipt,'worker_id':worker,
            **{key:state[key] for key in ['seconds','compute_seconds','transfer_seconds','encoded_data_bytes','artifact_initially_local']},
            **{key:route[key] for key in ['transport','encrypted','relay_host'] if key in route}}
        return row,result.to_dict()
    started=time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=min(args.parallel_jobs,len(plan))) as pool:
            futures=[pool.submit(contribute,index,worker) for index,worker in plan]
            try:
                for future in as_completed(futures):
                    row,result=future.result();report['contributions'].append(row);report['signed_results'].append(result);atomic_json(path,report)
            except BaseException:cancel.set();raise
        report['contribution_wall_seconds']=time.monotonic()-started
        report['contributions'].sort(key=lambda row:row['start']);report['signed_results'].sort(key=lambda row:row['output']['start'])
        if args.candidate:
            payload={'base_name':'qwen3-1.7b','release_id':head['release']['release_id'],'adapter_id':head['release']['model_artifact_id'],
                'dataset_id':corpus_digest(public_corpus()),'training_policy_id':BATCH_POLICY_ID,
                'contributions':report['signed_results'],'learning_rate':args.learning_rate}
            state=execute(own,'language.candidate.v2',payload);report['candidate_result']=state['result']
        report.update(success=True,total_tokens=sum(row['tokens'] for row in report['contributions']),
            total_elapsed_seconds=time.monotonic()-started,finished_at=time.time())
        if args.candidate:report['seconds_until_evaluated_candidate']=report['total_elapsed_seconds']
    finally:
        cancel.set();atomic_json(path,report)


if __name__=='__main__':main()
