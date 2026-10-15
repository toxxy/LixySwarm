"""Obtain typed public LoRA contributions from explicit physical workers."""
import argparse
from contextlib import contextmanager
import ipaddress
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from types import SimpleNamespace

from src.contribution import ContributionPolicy, ResourceGovernor, ResourceRequirements
from src.learning.public_bases import base_hash
from src.learning.public_conversation import public_corpus, corpus_digest
from src.network.artifact_store import ArtifactService, ArtifactStore
from src.network.bootstrap import get_seed_endpoints
from src.network.client_state import NodeStateLock,state_directory,ClientSettings
from src.network.public_knowledge import BandwidthBudget
from src.network.language_training import OPERATION,VERSIONED_OPERATION
from src.network.language_version_service import fetch_history
from src.network.lsp import LSPIdentity
from src.network.lsp_v3 import LSPNodeV3
from src.network.telemetry import atomic_json
from src.network.work_protocol import ResultReceipt, WorkCoordinator, WorkResult
from src.release.language_family import ADAPTERS
from src.release.language_versions import verify_contributions
from src.learning.language_capacity import allocate_shards,crossed_calibration


def shard_plan(workers,count,shards=None):
    total=len(workers) if shards is None else shards
    if (not 1<=len(workers)<=8 or len(set(workers))!=len(workers)
            or type(count) is not int or not 1<=count<=8
            or type(total) is not int or not len(workers)<=total<=16
            or any(not isinstance(worker,str) or len(worker)!=64 or any(c not in '0123456789abcdef' for c in worker) for worker in workers)):
        raise ValueError('Use distinct physical workers and at most sixteen disjoint shards')
    return [(index,workers[index%len(workers)]) for index in range(total)]


@contextmanager
def requester_lock(node_id,*,timeout_s=120,directory=None):
    # The waiting two-PC pilot and a baseline must never open the same network
    # identity at once, even when they write reports to different directories.
    lock=NodeStateLock((state_directory()/'requesters' if directory is None else directory)/node_id)
    deadline=time.monotonic()+timeout_s
    while True:
        try:lock.__enter__();break
        except RuntimeError:
            if time.monotonic()>=deadline:raise RuntimeError('Public requester identity is already in use')
            time.sleep(min(.1,max(0,deadline-time.monotonic())))
    try:yield
    finally:lock.__exit__(None,None,None)


def load_checkpoint(path,*,worker,requester,base_name,release_id,payload):
    """Recover only signed work bound to this exact worker, head and shard."""
    if path.stat().st_size>256*1024:raise ValueError('Contribution checkpoint exceeds its bound')
    saved=json.loads(path.read_text())
    if (saved['worker_id']!=worker or saved['base_name']!=base_name
            or saved['release_id']!=release_id):
        raise ValueError('Contribution checkpoint belongs to another worker or version')
    result=WorkResult.from_dict(saved['result'])
    if not ResultReceipt.from_dict(result.receipt).verify(result,
            expected_worker_id=worker,expected_requester_id=requester):
        raise ValueError('Contribution checkpoint receipt is invalid')
    verify_contributions([result.to_dict()],base_name=base_name,
        adapter_id=payload['adapter_id'],dataset_id=payload['dataset_id'],require_distributed=False)
    if any(result.output[key]!=payload[key] for key in ['start','count']):
        raise ValueError('Contribution checkpoint belongs to another shard')
    for key in ['compute_seconds','request_started_at']:
        if type(saved[key]) not in (int,float) or not math.isfinite(saved[key]) or saved[key]<=0:
            raise ValueError('Contribution checkpoint timing is invalid')
    return saved


def capacity_plan(args):
    slots={worker:1 for worker in args.worker}
    supplied=set()
    for item in getattr(args,'worker_slots',None) or []:
        worker,separator,value=item.partition('=')
        if not separator or worker not in slots or worker in supplied or value not in {'1','2','3','4'}:
            raise ValueError('Use one explicit slot limit per known worker: NODE_ID=1..4')
        supplied.add(worker);slots[worker]=int(value)
    count=len(args.worker) if args.shards is None else args.shards
    capacity=None;path=getattr(args,'capacity_report',None)
    if path is not None:
        if args.legacy or path.stat().st_size>1024**2:raise ValueError('Capacity evidence requires bounded versioned trials')
        data=json.loads(path.read_text());capacity=crossed_calibration(data['reports'])
        if set(capacity['workers'])!=set(args.worker) or capacity['base_name']!=args.base_name:
            raise ValueError('Capacity calibration belongs to another family or worker set')
    rows=allocate_shards(args.worker,args.count,count,capacity=capacity,slots=slots)
    return [(index,row['worker_id']) for index,row in enumerate(rows)],slots,capacity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--identity-key', type=Path, required=True)
    parser.add_argument('--worker', action='append', required=True)
    parser.add_argument('--count', type=int, default=4)
    parser.add_argument('--shards',type=int,help='Total distinct shards; defaults to one per worker')
    parser.add_argument('--parallel-jobs',type=int,help='Concurrent requests, from one to four')
    parser.add_argument('--base-name',choices=['qwen3-0.6b','qwen3-1.7b'],default='qwen3-0.6b')
    parser.add_argument('--legacy',action='store_true')
    parser.add_argument('--candidate-worker')
    parser.add_argument('--learning-rate',type=float,default=.001)
    parser.add_argument('--resume',action='store_true',help='Recover signed contributions and partial downloads from this directory')
    parser.add_argument('--capacity-report',type=Path,help='Local signed crossed-calibration reports used for observed allocation')
    parser.add_argument('--worker-slots',action='append',help='Locally selected concurrent request cap: NODE_ID=1..4; worker consent remains authoritative')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.directory.resolve().is_relative_to(repo):
        parser.error('Experiment state must be outside the checkout')
    try:shard_plan(args.worker,args.count,args.shards)
    except ValueError as error:parser.error(str(error))
    if args.parallel_jobs is not None and not 1<=args.parallel_jobs<=4:parser.error('Use one to four concurrent jobs')
    if args.legacy and (args.base_name!='qwen3-0.6b' or args.candidate_worker):
        parser.error('Legacy gradients support only the original small family and no candidate')
    if args.resume and args.legacy:parser.error('Recovery requires versioned contributions')
    if not 1e-6<=args.learning_rate<=1e-3:parser.error('Candidate step is outside its bound')
    identity = LSPIdentity.load(str(args.identity_key))
    if identity is None:
        parser.error('A locally authorized requester identity is required')
    with requester_lock(identity.node_id_hex):run(args,identity)


def run(args,identity):
    shard_plan(args.worker,args.count,args.shards)
    plan,slots,capacity=capacity_plan(args)
    parallel_jobs=min(len(plan),args.parallel_jobs if args.parallel_jobs is not None else min(4,len(args.worker)))
    args.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    resuming=getattr(args,'resume',False)
    prior_path=args.directory/'report.json'
    if prior_path.exists() and not resuming:
        raise ValueError('Existing experiment requires --resume or a new directory')
    if resuming and not prior_path.exists():raise ValueError('No experiment exists to resume')
    prior=json.loads(prior_path.read_text()) if resuming else None
    expected_plan=[{'worker_id':worker,'start':index*args.count,'count':args.count} for index,worker in plan]
    if prior is not None and any(prior.get(key)!=value for key,value in {
            'base_name':args.base_name,'requester_id':identity.node_id_hex,
            'operation':VERSIONED_OPERATION,'physical_workers':args.worker,'shard_plan':expected_plan}.items()):
        raise ValueError('Recovery must preserve the original requester, workers, model and shard plan')
    node = LSPNodeV3(identity, host='0.0.0.0', port=0, seeds=get_seed_endpoints(),
        address_book_path=args.directory/'peers.json', allow_private=False)
    work = WorkCoordinator(node, ResourceGovernor(ContributionPolicy.for_mode('relay'),
        hardware=SimpleNamespace(cpu_cores=1, ram_gb=1, disk_gb=1, has_gpu=False, gpu_vram_gb=0),
        storage_path=args.directory))
    store = ArtifactStore(args.directory/'gradients', max_artifact_bytes=32*1024**2, max_total_bytes=256*1024**2)
    def receive_rate():
        controls=ClientSettings.load(state_directory())
        return min(2.5,controls.policy.max_bandwidth_mbps/2) if controls.connected and not controls.paused else 0
    artifacts = ArtifactService(work, store, namespace='language.artifact', register_handlers=False,
        receive_budget=BandwidthBudget(receive_rate))
    report = {'schema_version': 1, 'scope': 'Explicit public-data gradient pilot; no model promoted',
        'physical_workers': args.worker, 'requester_id': identity.node_id_hex,
        'requester_is_extra_physical_worker': False, 'contributions': [], 'success': False,
        'candidate_promoted': False, 'started_at': time.time()}
    report.update(base_name=args.base_name,operation=OPERATION if args.legacy else VERSIONED_OPERATION,
        signed_results=[],shard_plan=[{'worker_id':worker,'start':index*args.count,'count':args.count} for index,worker in plan],
        parallel_jobs=parallel_jobs,receive_bandwidth_mbps=receive_rate())
    report.update(resumed=resuming,reused_contributions=0)
    report['worker_slot_limits']=slots
    if capacity is not None:report['capacity_profile']={key:value for key,value in capacity.items() if key!='observations'}
    if resuming:
        report['eligible_for_uncontended_scaling_comparison']=False
        report['original_started_at']=prior.get('original_started_at',prior['started_at'])
        report['recovery_attempt']=prior.get('recovery_attempt',0)+1
        if 'head' in prior:report.update(head=prior['head'],history=prior['history'])
        atomic_json(args.directory/f"report-before-recovery-{report['recovery_attempt']}.json",prior)
    path = args.directory/'report.json'
    atomic_json(path, report)
    try:
        node.start()
        def connect(worker):
            until = time.monotonic()+90
            while worker not in node.connected_node_ids:
                if time.monotonic() >= until:
                    raise TimeoutError('Physical worker discovery expired')
                node.connect_node(worker, timeout=12)
                time.sleep(.1)
            route = next(p for p in node.peers() if p['node_id'] == worker)
            if not route['encrypted'] or not ipaddress.ip_address(route.get('relay_host') or route['host']).is_global:
                raise ValueError('Gradient pilot requires encrypted public Internet transport')
            return route
        routes={}
        for worker in args.worker:
            route=connect(worker)
            routes[worker]=route
            if not args.legacy:
                history=fetch_history(work,args.base_name,worker);head=history[-1]
                if prior is not None and prior.get('head')!=head:
                    raise ValueError('Language head changed; recovered gradients cannot update the new head')
                if 'head' in report and head['release']['release_id']!=report['head']['release']['release_id']:
                    raise ValueError('Physical workers have different language versions')
                report['head']=head
                report['history']=history
        if prior is not None and prior.get('head')!=report.get('head'):
            raise ValueError('Language head changed; recovered gradients cannot update the new head')
        if capacity is not None and (capacity['release_id']!=report['head']['release']['release_id']
                or capacity['adapter_id']!=report['head']['release']['model_artifact_id']):
            raise ValueError('Capacity observations belong to another accepted version')
        def payload_for(index):
            payload={'adapter_id':ADAPTERS[args.base_name] if args.legacy else report['head']['release']['model_artifact_id'],
                'dataset_id':corpus_digest(public_corpus()),'start':index*args.count,'count':args.count}
            if not args.legacy:payload.update(base_name=args.base_name,release_id=report['head']['release']['release_id'])
            return payload
        # Validate every existing receipt before dispatching any new work. A
        # corrupt checkpoint is evidence to inspect, never a reason to retrain.
        checkpoints={}
        if resuming:
            for index,worker in plan:
                checkpoint=args.directory/f'signed-contribution-{index}.json'
                if checkpoint.exists():
                    checkpoints[index]=load_checkpoint(checkpoint,worker=worker,requester=identity.node_id_hex,
                        base_name=args.base_name,release_id=report['head']['release']['release_id'],payload=payload_for(index))
            report['reused_contributions']=len(checkpoints)
        atomic_json(path,report)
        cancelled=threading.Event()
        calculations={worker:threading.BoundedSemaphore(count) for worker,count in slots.items()}
        def contribute(index,worker):
            route=routes[worker]
            payload=payload_for(index)
            started = time.monotonic()
            began_at=time.time()
            if index in checkpoints:
                saved=checkpoints[index];result=WorkResult.from_dict(saved['result'])
                compute_seconds=saved['compute_seconds'];began_at=saved['request_started_at']
            else:
                with calculations[worker]:
                    if cancelled.is_set():raise RuntimeError('Capacity trial cancelled before dispatch')
                    compute_started=time.monotonic();began_at=time.time()
                    result = work.submit(OPERATION if args.legacy else VERSIONED_OPERATION, payload, ResourceRequirements(kind='training', gpu_required=True,
                        ram_gb=6 if args.base_name=='qwen3-1.7b' else 4, disk_gb=.04), peer_id=worker, timeout_s=240,cancel_event=cancelled)
                    compute_seconds=time.monotonic()-compute_started
            if result.status != 'ok':
                raise RuntimeError(result.error)
            if not ResultReceipt.from_dict(result.receipt).verify(result, expected_worker_id=worker,
                    expected_requester_id=identity.node_id_hex):
                raise ValueError('Language contribution receipt is invalid')
            output = result.output
            if output['base_hash'] != base_hash(args.base_name) or any(output[k] != payload[k] for k in ['adapter_id','dataset_id','start','count']):
                raise ValueError('Returned language contribution is from another version or shard')
            # Preserve the signed calculation before acquiring its bytes. A
            # transport failure must not erase the proof of completed work.
            atomic_json(args.directory/f'signed-contribution-{index}.json',{
                'result':result.to_dict(),'worker_id':worker,'base_name':args.base_name,
                'release_id':None if args.legacy else report['head']['release']['release_id'],
                'compute_seconds':compute_seconds,'request_started_at':began_at})
            for attempt in range(3):
                try:
                    gradient = artifacts.fetch(output['gradient_id'], peer_id=worker, timeout_s=120,
                        expected_size=output['gradient_bytes'],cancel_event=cancelled)
                    break
                except (OSError,ValueError,RuntimeError,TimeoutError):
                    if attempt==2 or cancelled.is_set():raise
                    route=connect(worker)
            row={**output, 'path': str(gradient), 'worker_id': worker,
                'seconds': round(time.monotonic()-started, 3), 'receipt': result.receipt,
                'compute_seconds':compute_seconds,'request_started_at':began_at,'received_at':time.time(),
                'reused_signed_contribution':index in checkpoints,
                'transport': route['transport'], 'encrypted': route['encrypted']}
            return row,result.to_dict()
        began=time.monotonic()
        with ThreadPoolExecutor(max_workers=parallel_jobs) as pool:
            futures=[pool.submit(contribute,index,worker) for index,worker in plan]
            try:
                for future in as_completed(futures):
                    row,result=future.result();report['contributions'].append(row);report['signed_results'].append(result)
                    atomic_json(path,report)
            except BaseException:
                cancelled.set()
                raise
        report['contributions'].sort(key=lambda row:row['start'])
        report['signed_results'].sort(key=lambda row:row['output']['start'])
        report.update(parallel_workers=min(parallel_jobs,len(args.worker)),contribution_wall_seconds=time.monotonic()-began)
        if args.candidate_worker:
            route=connect(args.candidate_worker)
            payload={'base_name':args.base_name,'release_id':report['head']['release']['release_id'],
                'adapter_id':report['head']['release']['model_artifact_id'],'dataset_id':corpus_digest(public_corpus()),
                'contributions':report['signed_results'],'learning_rate':args.learning_rate}
            report.update(candidate_state='evaluating',candidate_worker=args.candidate_worker)
            atomic_json(path,report)
            result=work.submit('language.candidate.v1',payload,ResourceRequirements(kind='training',gpu_required=True,
                ram_gb=6 if args.base_name=='qwen3-1.7b' else 4,disk_gb=.1),peer_id=args.candidate_worker,timeout_s=900)
            if result.status!='ok':raise RuntimeError(result.error)
            if not ResultReceipt.from_dict(result.receipt).verify(result,expected_worker_id=args.candidate_worker,
                    expected_requester_id=identity.node_id_hex):raise ValueError('Candidate evaluation receipt is invalid')
            report.update(candidate_state='completed',candidate_result=result.to_dict(),candidate_transport=route['transport'])
        report.update(success=True, finished_at=time.time(), total_tokens=sum(c['tokens'] for c in report['contributions']))
        atomic_json(path, report)
        print(f"Verified {len(report['contributions'])} contributions; {report['total_tokens']} tokens; no promotion", flush=True)
    except Exception as error:
        report.update(error_type=type(error).__name__, finished_at=time.time())
        atomic_json(path, report)
        raise
    finally:
        work.close()
        node.stop()


if __name__ == '__main__':
    main()
