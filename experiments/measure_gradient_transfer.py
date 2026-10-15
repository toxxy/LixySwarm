"""Crossed public-LSP artifact timing; does not execute or accept a gradient."""
import argparse
import ipaddress
import json
from pathlib import Path
import re
import signal
import statistics
import sys
import threading
import time
from types import SimpleNamespace

from src.contribution import ContributionPolicy, ResourceGovernor
from src.network.artifact_store import ArtifactStore, ArtifactService, digest_file
from src.network.bootstrap import get_seed_endpoints
from src.network.client_state import ClientSettings
from src.network.lsp import LSPIdentity
from src.network.lsp_v3 import LSPNodeV3
from src.network.public_knowledge import BandwidthBudget, managed_public_bytes
from src.network.telemetry import atomic_json
from src.network.work_protocol import WorkCoordinator, ResultReceipt


def run(args):
    cancel=threading.Event()
    for signum in [signal.SIGTERM,signal.SIGINT]:signal.signal(signum,lambda *_:cancel.set())
    identity=LSPIdentity.generate();controls=ClientSettings.load(args.node_directory)
    physical=json.loads((args.node_directory/'diagnostics.json').read_text())['node_id']
    if physical==args.provider:raise ValueError('Use a different physical host for the receiver')
    charged=managed_public_bytes(args.node_directory)+2*(args.original_bytes+args.packed_bytes)+1024**2
    def rate():
        value=ClientSettings.load(args.node_directory)
        return (min(5,value.policy.max_bandwidth_mbps) if value.connected and not value.paused
            and value.policy.max_disk_gb*1024**3>=charged else 0)
    if rate()<=0:raise ValueError('Transfer requires current connectivity and storage consent')
    node=LSPNodeV3(identity,host='0.0.0.0',port=0,seeds=get_seed_endpoints(),address_book_path=args.directory/'peers.json',allow_private=False)
    work=WorkCoordinator(node,ResourceGovernor(ContributionPolicy.for_mode('relay'),
        hardware=SimpleNamespace(cpu_cores=1,ram_gb=1,has_gpu=False,gpu_vram_gb=0,disk_gb=1),storage_path=args.directory))
    submit=work.submit;jobs=set();manifests=[]
    def verified_submit(*a,**kw):
        result=submit(*a,**kw)
        if result.status!='ok' or not ResultReceipt.from_dict(result.receipt).verify(result,
                expected_worker_id=args.provider,expected_requester_id=identity.node_id_hex):
            raise ValueError('Unverified public artifact response')
        if result.job_id in jobs:raise ValueError('Timing reused a previous signed work result')
        jobs.add(result.job_id)
        if a[0].endswith('.describe.v1'):manifests.append(result.to_dict())
        return result
    work.submit=verified_submit
    rows=[];began=time.monotonic()
    try:
        node.start()
        while args.provider not in node.connected_node_ids:
            if time.monotonic()-began>60 or cancel.is_set():raise TimeoutError('Public provider discovery expired')
            node.connect_node(args.provider,timeout=10)
        # ABBA order gives each representation an early and late observation.
        for index,mode in enumerate(['fp32','half','half','fp32']):
            if rate()<=0:raise ValueError('Transfer consent withdrawn')
            identifier,size=(args.original_id,args.original_bytes) if mode=='fp32' else (args.packed_id,args.packed_bytes)
            route=next(row for row in node.peers() if row['node_id']==args.provider)
            if not route['encrypted'] or not ipaddress.ip_address(route.get('relay_host') or route['host']).is_global:
                raise ValueError('Only public encrypted Internet transport counts')
            store=ArtifactStore(args.directory/f'trial-{index}',max_artifact_bytes=32*1024**2,max_total_bytes=64*1024**2)
            if store.has(identifier):raise ValueError('Each measurement requires empty artifact storage')
            transfer=ArtifactService(work,store,namespace='language.artifact',register_handlers=False,receive_budget=BandwidthBudget(rate))
            stats=[];start=time.monotonic()
            path=transfer.fetch(identifier,peer_id=args.provider,expected_size=size,timeout_s=120,cancel_event=cancel,observer=stats.append)
            elapsed=time.monotonic()-start
            if digest_file(path)!=(identifier,size):raise ValueError('Received bytes differ from the expected artifact')
            rows.append({'index':index,'mode':mode,'artifact_id':identifier,'bytes':size,'seconds':elapsed,
                'encoded_data_bytes':sum(row['encoded_data_bytes'] for row in stats),'chunks':len(stats),
                'hash_verified':True,'initially_cached':False,'route':{k:route.get(k) for k in ['host','relay_host','transport','encrypted']}})
            atomic_json(args.directory/'progress.json',{'rows':rows,'time':time.time()})
        times={mode:statistics.median(r['seconds'] for r in rows if r['mode']==mode) for mode in ['fp32','half']}
        report={'success':True,'scope':'Artifact transfer only, two known physical PCs; not one-versus-two training speed or model promotion',
            'physical_receiver':physical,'physical_provider':args.provider,'requester_id':identity.node_id_hex,
            'requester_is_additional_physical_worker':False,'sequence':'ABBA','receive_bandwidth_mbps':min(5,controls.policy.max_bandwidth_mbps),
            'rows':rows,'median_seconds':times,'observed_transfer_speedup':times['fp32']/times['half'],
            'verified_work_results':len(jobs),'manifest_results':manifests,'tensor_engine_imported':False,
            'new_model_trained':False,'new_model_accepted':False,'private_inputs':False}
        assert 'torch' not in sys.modules
        atomic_json(args.directory/'report.json',report);print(json.dumps({'success':True,'median_seconds':times,'speedup':report['observed_transfer_speedup']}),flush=True)
    finally:
        work.close();node.stop()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--node-directory',type=Path,required=True)
    for name in ['provider','original-id','packed-id']:parser.add_argument('--'+name,required=True)
    for name in ['original-bytes','packed-bytes']:parser.add_argument('--'+name,type=int,required=True)
    args=parser.parse_args()
    if (any(not re.fullmatch('[0-9a-f]{64}',v) for v in [args.provider,args.original_id,args.packed_id])
            or any(not 1<=v<=32*1024**2 for v in [args.original_bytes,args.packed_bytes])
            or any(p.resolve().is_relative_to(Path.home()/'Dropbox') for p in [args.directory,args.node_directory])):
        parser.error('Use exact public identities, bounded artifacts and storage outside Dropbox')
    args.directory.mkdir(parents=True,exist_ok=False,mode=0o700);run(args)


if __name__=='__main__':main()
