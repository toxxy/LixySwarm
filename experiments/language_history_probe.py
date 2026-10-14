"""Discover an anchored language history over public encrypted LSP."""
import argparse
import ipaddress
import time
from pathlib import Path
from types import SimpleNamespace

from src.contribution import ContributionPolicy,ResourceGovernor
from src.network.bootstrap import get_seed_endpoints
from src.network.language_version_service import fetch_history
from src.network.lsp import LSPIdentity
from src.network.lsp_v3 import LSPNodeV3
from src.network.telemetry import atomic_json
from src.network.work_protocol import WorkCoordinator
from src.release.language_family import BASES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--provider',required=True)
    parser.add_argument('--base-name',choices=BASES,required=True)
    args=parser.parse_args()
    if args.directory.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        parser.error('Probe state must be outside the checkout')
    args.directory.mkdir(parents=True,exist_ok=True)
    node=LSPNodeV3(LSPIdentity.generate(),host='0.0.0.0',port=0,seeds=get_seed_endpoints(),
        address_book_path=args.directory/'peers.json',allow_private=False)
    work=WorkCoordinator(node,ResourceGovernor(ContributionPolicy.for_mode('relay'),hardware=SimpleNamespace(
        cpu_cores=1,ram_gb=1,disk_gb=1,has_gpu=False,gpu_vram_gb=0),storage_path=args.directory))
    try:
        node.start();deadline=time.monotonic()+60
        while args.provider not in node.connected_node_ids:
            if time.monotonic()>deadline:raise TimeoutError('Public provider discovery expired')
            node.connect_node(args.provider,timeout=12)
        route=next(p for p in node.peers() if p['node_id']==args.provider)
        assert route['encrypted'] and ipaddress.ip_address(route.get('relay_host') or route['host']).is_global
        start=time.monotonic();chain=fetch_history(work,args.base_name,args.provider)
        atomic_json(args.directory/'report.json',{'scope':'Version metadata on two existing PCs; visitor adds no physical worker',
            'seconds':time.monotonic()-start,'provider_id':args.provider,'transport':route['transport'],'encrypted':True,
            'base_name':args.base_name,'versions':len(chain),'head_id':chain[-1]['release']['release_id'],
            'sequence':chain[-1]['release']['sequence'],'new_language_update_promoted':False,'success':True})
        print('Public signed language history verified',flush=True)
    finally:work.close();node.stop()


if __name__=='__main__':main()
