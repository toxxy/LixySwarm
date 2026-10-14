"""Request a public language expert across Internet, without loading tensors."""
from __future__ import annotations
import argparse
import ipaddress
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--state",type=Path,required=True)
    parser.add_argument("--worker",required=True);parser.add_argument("--model-id",required=True)
    parser.add_argument("--report",type=Path,required=True);args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    if args.state.resolve().is_relative_to(repo):parser.error("Probe state must be outside the checkout")
    from src.network.lsp import LSPIdentity
    from src.network.lsp_v3 import LSPNodeV3
    from src.network.bootstrap import get_seed_endpoints
    from src.network.work_protocol import WorkCoordinator,ResultReceipt
    from src.network.language_compute import LANGUAGE_OPERATION
    from src.contribution import ContributionPolicy,ResourceGovernor,ResourceRequirements
    args.state.mkdir(parents=True,exist_ok=True)
    identity=LSPIdentity.generate()
    node=LSPNodeV3(identity,host="0.0.0.0",port=0,seeds=get_seed_endpoints(),address_book_path=args.state/"peers.json")
    work=WorkCoordinator(node,ResourceGovernor(ContributionPolicy.for_mode("relay"),
        hardware=SimpleNamespace(cpu_cores=1,ram_gb=1,has_gpu=False,gpu_vram_gb=0,disk_gb=1),storage_path=args.state))
    report={"schema_version":1,"scope":"Public test questions from a requester; no extra physical worker is counted","worker_id":args.worker,
            "model_id":args.model_id,"results":[],"requester_loaded_tensors":False}
    try:
        node.start();until=time.monotonic()+90
        while args.worker not in node.connected_node_ids:
            if time.monotonic()>until:raise RuntimeError("Public expert connection deadline exceeded")
            node.connect_node(args.worker,timeout=12);time.sleep(0.1)
        route=next(item for item in node.peers() if item['node_id']==args.worker)
        if not route['encrypted'] or not ipaddress.ip_address(route.get('relay_host') or route['host']).is_global:
            raise ValueError('Language probe requires encrypted public Internet transport')
        for messages in [[{"role":"user","content":"¿Qué es una red descentralizada? Contesta en español y brevemente."}],
                         [{"role":"user","content":"Three computers hold the same learned model. If one disconnects, is the model necessarily lost? Explain briefly."}],
                         [{"role":"user","content":"In this fictional example, the expert is named Amber."},{"role":"assistant","content":"The expert is Amber."},
                          {"role":"user","content":"What is the expert called? Reply only with its name."}]]:
            started=time.monotonic()
            result=work.submit(LANGUAGE_OPERATION,{"model_artifact_id":args.model_id,"messages":messages},
                ResourceRequirements(kind="inference",gpu_required=True,ram_gb=5),peer_id=args.worker,timeout_s=60)
            if result.status!="ok":raise RuntimeError(result.error)
            assert ResultReceipt.from_dict(result.receipt).verify(result,expected_worker_id=args.worker,expected_requester_id=identity.node_id_hex)
            assert result.output["model_id"]==args.model_id and "torch" not in sys.modules and "transformers" not in sys.modules
            report["results"].append({"messages":messages,"answer":result.output["answer"],"output_tokens":result.output["output_tokens"],
                                     "seconds":time.monotonic()-started,"receipt":result.receipt})
            print(json.dumps({"answer":result.output["answer"],"seconds":report["results"][-1]["seconds"]},ensure_ascii=False),flush=True)
        route=next(item for item in node.peers() if item["node_id"]==args.worker)
        report.update(success=True,transport=route["transport"],encrypted=route["encrypted"],public_internet=True)
        args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    finally:work.close();node.stop()


if __name__=="__main__":main()
