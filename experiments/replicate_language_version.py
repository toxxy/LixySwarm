"""Acquire/repair signed language history over Internet, preserving local consent."""
import argparse
import ipaddress
import json
from pathlib import Path
import signal
import threading
import time
from types import SimpleNamespace

from src.contribution import ContributionPolicy,ResourceGovernor
from src.learning.public_bases import BASES
from src.network.artifact_store import ArtifactStore
from src.network.bootstrap import get_seed_endpoints
from src.network.client_state import ClientSettings,NodeStateLock
from src.network.language_replication import replicate_version
from src.network.lsp import LSPIdentity
from src.network.lsp_v3 import LSPNodeV3
from src.network.public_knowledge import KnowledgeSettings,BandwidthBudget,managed_public_bytes
from src.network.telemetry import atomic_json
from src.network.work_protocol import WorkCoordinator
from src.release.language_versions import dependencies


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--visitor-directory',type=Path,required=True)
    parser.add_argument('--base-name',choices=BASES,required=True)
    parser.add_argument('--provider',action='append',required=True)
    args=parser.parse_args();repo=Path(__file__).resolve().parents[1]
    if (not 1<=len(args.provider)<=8 or len(set(args.provider))!=len(args.provider)
            or any(len(value)!=64 or any(c not in '0123456789abcdef' for c in value) for value in args.provider)):
        parser.error('Use one to eight distinct provider identities')
    if any(path.resolve().is_relative_to(repo) for path in [args.directory,args.visitor_directory]):
        parser.error('Knowledge and visitor state must stay outside the checkout')
    if (args.directory.resolve().is_relative_to(args.visitor_directory.resolve())
            or args.visitor_directory.resolve().is_relative_to(args.directory.resolve())):
        parser.error('Visitor and physical node state must use disjoint directories')
    cancel=threading.Event()
    for signum in [signal.SIGTERM,signal.SIGINT]:signal.signal(signum,lambda *_:cancel.set())
    def allowed():
        try:
            settings=ClientSettings.load(args.directory);sharing=KnowledgeSettings.load(args.directory)
            return (settings.connected and not settings.paused and sharing.enabled and settings.policy.mode!='disabled'
                and settings.policy.max_bandwidth_mbps>0 and settings.policy.max_disk_gb>0)
        except (OSError,ValueError,TypeError):return False
    if not allowed():raise ValueError('Local public-knowledge storage and connectivity must already be authorized')
    with NodeStateLock(args.visitor_directory):
        key=args.visitor_directory/'visitor.key';identity=LSPIdentity.load(str(key))
        if identity is None:identity=LSPIdentity.generate();identity.save(str(key))
        node=LSPNodeV3(identity,host='0.0.0.0',port=0,seeds=get_seed_endpoints(),
            address_book_path=args.visitor_directory/'peers.json',allow_private=False)
        work=WorkCoordinator(node,ResourceGovernor(ContributionPolicy.for_mode('relay'),hardware=SimpleNamespace(
            cpu_cores=1,ram_gb=1,disk_gb=1,has_gpu=False,gpu_vram_gb=0),storage_path=args.visitor_directory))
        root=args.directory/'public/language';store=ArtifactStore(root/'artifacts',max_artifact_bytes=4*1024**3,max_total_bytes=8*1024**3)
        def preflight(chain):
            objects={identifier:size for bundle in chain for identifier,size in dependencies(bundle)}
            missing=sum(size for identifier,size in objects.items() if not store.has(identifier))
            if managed_public_bytes(args.directory)+2*missing>ClientSettings.load(args.directory).policy.max_disk_gb*1024**3:
                raise ValueError('Language replication exceeds local storage consent')
        last_update=0
        def progress(identifier,done,provider):
            nonlocal last_update
            now=time.monotonic()
            if now-last_update>=1:
                if not allowed():cancel.set();raise ValueError('Language replication permission withdrawn')
                if managed_public_bytes(args.directory)>ClientSettings.load(args.directory).policy.max_disk_gb*1024**3:
                    cancel.set();raise ValueError('Language replication storage consent withdrawn')
                atomic_json(args.visitor_directory/'progress.json',{'artifact_id':identifier,'bytes':done,'provider':provider,'timestamp':time.time()})
                last_update=now
        def guard():
            while not cancel.wait(.5):
                if not allowed():cancel.set();return
        monitor=threading.Thread(target=guard,daemon=True);monitor.start()
        report={'success':False,'visitor_is_extra_physical_worker':False,'started_at':time.time(),
            'base_name':args.base_name,'active_expert_changed':False,'resource_permissions_changed':False}
        atomic_json(args.visitor_directory/'report.json',report)
        try:
            node.start();providers=[]
            for provider in args.provider:
                until=time.monotonic()+45
                while provider not in node.connected_node_ids and not cancel.is_set() and time.monotonic()<until:
                    node.connect_node(provider,timeout=10)
                if provider not in node.connected_node_ids:continue
                route=next(peer for peer in node.peers() if peer['node_id']==provider)
                if not route['encrypted'] or not ipaddress.ip_address(route.get('relay_host') or route['host']).is_global:
                    raise ValueError('Language replication requires encrypted public Internet transport')
                providers.append(provider)
            budget=BandwidthBudget(lambda:ClientSettings.load(args.directory).policy.max_bandwidth_mbps/2 if allowed() else 0)
            result=replicate_version(work,store,root,args.base_name,providers,cancel_event=cancel,receive_budget=budget,
                allowed=allowed,preflight=preflight,progress=progress)
            result.update(report,success=True,finished_at=time.time())
            atomic_json(args.visitor_directory/'report.json',result)
            print(json.dumps(result),flush=True)
        except Exception as error:
            report.update(error_type=type(error).__name__,finished_at=time.time())
            atomic_json(args.visitor_directory/'report.json',report)
            raise
        finally:cancel.set();monitor.join(timeout=2);work.close();node.stop()


if __name__=='__main__':main()
