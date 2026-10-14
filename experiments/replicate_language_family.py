"""Acquire complete public language dependencies by LSP, never VPN/file copy."""
import argparse
import ipaddress
import json
import signal
import shutil
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from src.contribution import ContributionPolicy,ResourceGovernor
from src.learning.public_bases import BASES
from src.network.artifact_store import ArtifactStore
from src.network.bootstrap import get_seed_endpoints
from src.network.language_families import fetch_family
from src.network.lsp import LSPIdentity
from src.network.lsp_v3 import LSPNodeV3
from src.network.public_knowledge import BandwidthBudget,managed_public_bytes
from src.network.telemetry import atomic_json
from src.network.work_protocol import WorkCoordinator
from src.release.language_family import dependencies


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--node-directory',type=Path,required=True)
    parser.add_argument('--requester-directory',type=Path,required=True)
    parser.add_argument('--base-name',choices=BASES,required=True)
    parser.add_argument('--provider',required=True)
    args=parser.parse_args()
    from src.network.client_state import ClientSettings
    directory=args.node_directory;root=directory/'public/language';requester=args.requester_directory
    requester.mkdir(parents=True,exist_ok=True,mode=0o700)
    identity=LSPIdentity.load(str(requester/'identity.key'))
    if identity is None:identity=LSPIdentity.generate();identity.save(str(requester/'identity.key'))
    node=LSPNodeV3(identity,host='0.0.0.0',port=0,seeds=get_seed_endpoints(),address_book_path=requester/'peers.json',allow_private=False)
    policy=ContributionPolicy.for_mode('relay');hardware=SimpleNamespace(cpu_cores=1,ram_gb=.5,has_gpu=False,gpu_vram_gb=0,disk_gb=10)
    work=WorkCoordinator(node,ResourceGovernor(policy,hardware=hardware,storage_path=requester))
    store=ArtifactStore(root/'artifacts',max_artifact_bytes=4*1024**3,max_total_bytes=8*1024**3)
    stop=threading.Event();report=requester/'progress.json';last_error=None
    trace=requester/f'transfer-{time.time_ns()}.jsonl';last_trace=0
    def timing(identifier,stats):
        nonlocal last_trace
        if time.monotonic()-last_trace>=10:
            with trace.open('a') as output:output.write(json.dumps({'timestamp':time.time(),'artifact_id':identifier,**stats})+'\n')
            last_trace=time.monotonic()
    def allowed():
        current=ClientSettings.load(directory)
        return current.connected and not current.paused and current.policy.mode!='disabled' and current.policy.max_bandwidth_mbps>0 and managed_public_bytes(directory)<=current.policy.max_disk_gb*1024**3
    def rate():return min(5,ClientSettings.load(directory).policy.max_bandwidth_mbps/2) if allowed() else 0
    budget=BandwidthBudget(rate)
    def preflight(bundle):
        required=0
        for identifier,size in dependencies(bundle):
            if store.has(identifier):continue
            partial=store.incoming/f'{identifier}.part'
            required+=(size if size is not None else 256*1024)-(partial.stat().st_size if partial.exists() else 0)
        limit=ClientSettings.load(directory).policy.max_disk_gb*1024**3
        if managed_public_bytes(directory)+required>limit or shutil.disk_usage(directory).free<required:
            raise ValueError('The complete family does not fit the authorized/free disk space')
    def status(state):
        partial=sum(p.stat().st_size for p in store.incoming.glob('*.part') if p.is_file())
        atomic_json(report,{'state':state,'base_name':args.base_name,'received_bytes':store.total_bytes()+partial,
            'updated_at':time.time(),'error_type':last_error,'worker_id':args.provider,'requester_id':identity.node_id_hex,
            'scope':'Visitor requester on the laptop; not an additional physical worker','transport':'public encrypted LSP'})
    def observe():
        next_report=0
        while not stop.wait(.5):
            if not allowed():stop.set()
            if time.monotonic()>=next_report or stop.is_set():
                status('downloading' if not stop.is_set() else 'paused');next_report=time.monotonic()+10
    for signum in (signal.SIGTERM,signal.SIGINT):signal.signal(signum,lambda *_:stop.set())
    watcher=threading.Thread(target=observe,daemon=True)
    try:
        if not allowed():raise ValueError('Public storage/bandwidth permission is unavailable')
        node.start();watcher.start();status('connecting')
        # The per-object fetch deadline is one hour; partial files persist.
        # Bound retries to one day, then preserve evidence for a later resume.
        until=time.monotonic()+86400
        while not stop.is_set() and time.monotonic()<until:
            try:
                if not node.connect_node(args.provider,timeout=12):raise ConnectionError('Provider not discovered')
                assert all(ipaddress.ip_address(p.get('relay_host') or p['host']).is_global for p in node.peers())
                if not allowed():raise ValueError('Public contribution permission withdrawn')
                bundle=fetch_family(work,store,root,args.base_name,args.provider,cancel_event=stop,
                    receive_budget=budget,progress=lambda *_:status('downloading'),preflight=preflight,observer=timing)
                stop.set();watcher.join(timeout=1)
                status('available');value=json.loads(report.read_text());value.update(release_id=bundle['release']['release_id'],
                    adapter_id=bundle['descriptor']['adapter_id'],success=True)
                atomic_json(report,value);return 0
            except Exception as exc:
                last_error=type(exc).__name__;status('retrying' if allowed() and not stop.is_set() else 'paused')
                if not allowed():break
                stop.wait(20)
        status('paused' if stop.is_set() else 'incomplete');return 1
    finally:
        stop.set();watcher.join(timeout=1) if watcher.is_alive() else None
        work.close();node.stop()


if __name__=='__main__':raise SystemExit(main())
