"""Obtain typed public LoRA contributions from explicit physical workers."""
import argparse
import ipaddress
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
from src.network.language_training import OPERATION,VERSIONED_OPERATION
from src.network.language_version_service import fetch_history
from src.network.lsp import LSPIdentity
from src.network.lsp_v3 import LSPNodeV3
from src.network.telemetry import atomic_json
from src.network.work_protocol import ResultReceipt, WorkCoordinator
from src.release.language_family import ADAPTERS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--identity-key', type=Path, required=True)
    parser.add_argument('--worker', action='append', required=True)
    parser.add_argument('--count', type=int, default=4)
    parser.add_argument('--base-name',choices=['qwen3-0.6b','qwen3-1.7b'],default='qwen3-0.6b')
    parser.add_argument('--legacy',action='store_true')
    parser.add_argument('--candidate-worker')
    parser.add_argument('--learning-rate',type=float,default=.001)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.directory.resolve().is_relative_to(repo):
        parser.error('Experiment state must be outside the checkout')
    if len(set(args.worker)) != len(args.worker) or not 1 <= len(args.worker) <= 8 or not 1 <= args.count <= 8:
        parser.error('Use distinct physical workers and bounded shard sizes')
    if args.legacy and (args.base_name!='qwen3-0.6b' or args.candidate_worker):
        parser.error('Legacy gradients support only the original small family and no candidate')
    if not 1e-6<=args.learning_rate<=1e-3:parser.error('Candidate step is outside its bound')
    identity = LSPIdentity.load(str(args.identity_key))
    if identity is None:
        parser.error('A locally authorized requester identity is required')
    args.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    node = LSPNodeV3(identity, host='0.0.0.0', port=0, seeds=get_seed_endpoints(),
        address_book_path=args.directory/'peers.json', allow_private=False)
    work = WorkCoordinator(node, ResourceGovernor(ContributionPolicy.for_mode('relay'),
        hardware=SimpleNamespace(cpu_cores=1, ram_gb=1, disk_gb=1, has_gpu=False, gpu_vram_gb=0),
        storage_path=args.directory))
    store = ArtifactStore(args.directory/'gradients', max_artifact_bytes=32*1024**2, max_total_bytes=256*1024**2)
    artifacts = ArtifactService(work, store, namespace='language.artifact', register_handlers=False)
    report = {'schema_version': 1, 'scope': 'Explicit public-data gradient pilot; no model promoted',
        'physical_workers': args.worker, 'requester_id': identity.node_id_hex,
        'requester_is_extra_physical_worker': False, 'contributions': [], 'success': False,
        'candidate_promoted': False, 'started_at': time.time()}
    report.update(base_name=args.base_name,operation=OPERATION if args.legacy else VERSIONED_OPERATION,
        signed_results=[])
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
                head=fetch_history(work,args.base_name,worker)[-1]
                if 'head' in report and head['release']['release_id']!=report['head']['release']['release_id']:
                    raise ValueError('Physical workers have different language versions')
                report['head']=head
        atomic_json(path,report)
        cancelled=threading.Event()
        def contribute(index,worker):
            route=routes[worker]
            payload = {'adapter_id': ADAPTERS[args.base_name] if args.legacy else report['head']['release']['model_artifact_id'], 'dataset_id': corpus_digest(public_corpus()),
                'start': index*args.count, 'count': args.count}
            if not args.legacy:payload.update(base_name=args.base_name,release_id=report['head']['release']['release_id'])
            started = time.monotonic()
            began_at=time.time()
            result = work.submit(OPERATION if args.legacy else VERSIONED_OPERATION, payload, ResourceRequirements(kind='training', gpu_required=True,
                ram_gb=6 if args.base_name=='qwen3-1.7b' else 4, disk_gb=.04), peer_id=worker, timeout_s=240,cancel_event=cancelled)
            compute_seconds=time.monotonic()-started
            if result.status != 'ok':
                raise RuntimeError(result.error)
            if not ResultReceipt.from_dict(result.receipt).verify(result, expected_worker_id=worker,
                    expected_requester_id=identity.node_id_hex):
                raise ValueError('Language contribution receipt is invalid')
            output = result.output
            if output['base_hash'] != base_hash(args.base_name) or any(output[k] != payload[k] for k in ['adapter_id','dataset_id','start','count']):
                raise ValueError('Returned language contribution is from another version or shard')
            gradient = artifacts.fetch(output['gradient_id'], peer_id=worker, timeout_s=60,
                expected_size=output['gradient_bytes'],cancel_event=cancelled)
            row={**output, 'path': str(gradient), 'worker_id': worker,
                'seconds': round(time.monotonic()-started, 3), 'receipt': result.receipt,
                'compute_seconds':compute_seconds,'request_started_at':began_at,'received_at':time.time(),
                'transport': route['transport'], 'encrypted': route['encrypted']}
            return row,result.to_dict()
        began=time.monotonic()
        with ThreadPoolExecutor(max_workers=min(4,len(args.worker))) as pool:
            futures=[pool.submit(contribute,index,worker) for index,worker in enumerate(args.worker)]
            try:
                for future in as_completed(futures):
                    row,result=future.result();report['contributions'].append(row);report['signed_results'].append(result)
                    atomic_json(path,report)
            except BaseException:
                cancelled.set()
                raise
        report['contributions'].sort(key=lambda row:row['start'])
        report['signed_results'].sort(key=lambda row:row['output']['start'])
        report.update(parallel_workers=min(4,len(args.worker)),contribution_wall_seconds=time.monotonic()-began)
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
