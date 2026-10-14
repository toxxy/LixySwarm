"""Public Internet work against the two persistent physical-node identities."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def identity_at(state):
    from src.network.lsp import LSPIdentity
    state.mkdir(parents=True,exist_ok=True,mode=0o700)
    identity=LSPIdentity.load(str(state/"identity.key"))
    if identity is None:
        identity=LSPIdentity.generate(); identity.save(str(state/"identity.key"))
    return identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state",type=Path,required=True)
    parser.add_argument("--workers",default="")
    parser.add_argument("--print-id",action="store_true")
    parser.add_argument("--pause-local",action="store_true")
    args=parser.parse_args(); identity=identity_at(args.state)
    if args.print_id: print(identity.node_id_hex); return
    from src.contribution import ContributionPolicy,ResourceGovernor,ResourceRequirements
    from src.network.artifact_store import ArtifactStore,ArtifactService
    from src.network.bootstrap import get_seed_endpoints
    from src.network.lsp_v3 import LSPNodeV3
    from src.network.work_protocol import WorkCoordinator,ResultReceipt
    from src.network.reference_compute import INFERENCE_OPERATION,TRAINING_OPERATION,PUBLIC_DATASET_ID
    from src.network.reference_model import reference_corpus,load_reference_checkpoint
    from src.network.training_worker import validate_gradient_artifact
    from src.release import ReleaseRegistry
    from src.release.reference_catalog import REFERENCE_CATALOG
    from experiments.internet_probe import route
    import numpy as np
    workers=args.workers.split(",")
    if len(workers)!=2 or len(set(workers))!=2: parser.error("Two distinct physical workers required")
    state=args.state.resolve()
    repo=Path(__file__).resolve().parents[1]
    if state==repo or repo in state.parents: parser.error("Probe state must be outside the checkout")
    governor=ResourceGovernor(ContributionPolicy.for_mode("balanced"),storage_path=state,
        hardware=SimpleNamespace(cpu_cores=4,ram_gb=8,gpu_vram_gb=0,has_gpu=False,disk_gb=1))
    node=LSPNodeV3(identity,seeds=get_seed_endpoints(),port=0,address_book_path=state/"peers.json",target_outbound=4,
        capabilities={"peer_exchange":True,"pheromone":False,"global_memory":False},
        resource_profile=governor.advertised_profile(available_work={"artifact"}))
    work=WorkCoordinator(node,governor,max_offers_per_minute=120)
    store=ArtifactStore(state/"artifacts",max_artifact_bytes=32*1024**2,max_total_bytes=64*1024**2)
    artifacts=ArtifactService(work,store)
    tokens,_=reference_corpus(); dataset=state/"public-tokens.npy"; np.save(dataset,tokens,allow_pickle=False)
    data=store.import_file(dataset,kind="dataset",media_type="application/x-npy")
    assert data.artifact_id==PUBLIC_DATASET_ID
    parent=Path.home()/".local/state/lixyswarm/public"
    release=ReleaseRegistry(parent/"releases").verify_active(REFERENCE_CATALOG.policy,ArtifactStore(parent/"artifacts"))
    model=load_reference_checkpoint(ArtifactStore(parent/"artifacts").object_path(release.model_artifact_id),"cpu")
    inference,training=[],[]
    try:
        node.start()
        print(json.dumps({"event":"started","requester":identity.node_id_hex}),flush=True)
        for index,peer in enumerate(workers):
            deadline=time.monotonic()+90
            while peer not in node.connected_node_ids:
                if time.monotonic()>deadline: raise RuntimeError("Worker connection deadline exceeded")
                node.connect_node(peer,timeout=12); time.sleep(0.1)
            for operation,payload,kind in (
                (INFERENCE_OPERATION,{"model_artifact_id":release.model_artifact_id,"prompt":"La memoria personal "},"inference"),
                (TRAINING_OPERATION,{"model_artifact_id":release.model_artifact_id,"dataset_artifact_id":PUBLIC_DATASET_ID,"start_token":index*7200,"token_count":65},"training")):
                began=time.monotonic()
                result=work.submit(operation,payload,ResourceRequirements(kind=kind,gpu_required=True,ram_gb=2,disk_gb=0.01),peer_id=peer,timeout_s=60)
                if result.status!="ok": raise RuntimeError(f"{kind}: {result.error}")
                assert ResultReceipt.from_dict(result.receipt).verify(result,expected_worker_id=peer,expected_requester_id=identity.node_id_hex)
                value={"worker_id":peer,"seconds":round(time.monotonic()-began,3),"receipt":result.receipt}
                if kind=="inference":
                    value["logits"]=result.output["next_byte_logits"]; inference.append(value)
                else:
                    path=artifacts.fetch(result.output["gradient"]["artifact_id"],peer_id=peer,timeout_s=60)
                    validate_gradient_artifact(path,model,expected_model_id=release.model_artifact_id,
                        expected_dataset_id=PUBLIC_DATASET_ID,expected_start_token=payload["start_token"],expected_token_count=65)
                    value.update(gradient_id=result.output["gradient"]["artifact_id"],loss=result.output["loss"],start_token=payload["start_token"]); training.append(value)
                print(json.dumps({"event":"validated","kind":kind,"worker":peer,"seconds":value["seconds"]}),flush=True)
        difference=max(abs(a-b) for a,b in zip(inference[0]["logits"],inference[1]["logits"]))
        pause_report=None
        if args.pause_local:
            from concurrent.futures import ThreadPoolExecutor
            from src.network.client_state import ClientSettings
            import psutil
            local_state=Path.home()/".local/state/lixyswarm"
            diagnostics=json.loads((local_state/"diagnostics.json").read_text())
            local_id=diagnostics["node_id"]; original_pid=diagnostics["pid"]
            assert local_id in workers
            policy=ClientSettings.load(local_state)
            assert not policy.paused
            with ThreadPoolExecutor(max_workers=1) as pool:
                future=pool.submit(work.submit,INFERENCE_OPERATION,
                    {"model_artifact_id":release.model_artifact_id,"prompt":"Una hormiga "},
                    ResourceRequirements(kind="inference",gpu_required=True,ram_gb=2,disk_gb=0.01),peer_id=local_id,timeout_s=60)
                child_pid=None; deadline=time.monotonic()+10
                while time.monotonic()<deadline and not future.done():
                    for child in psutil.Process(original_pid).children():
                        try:
                            if any(Path(x).name=="reference_job.py" for x in child.cmdline()): child_pid=child.pid
                        except psutil.Error: pass
                    if child_pid is not None: break
                    time.sleep(0.01)
                assert child_pid is not None,"The cancellation gate requires a live owned engine"
                try:
                    policy.paused=True; policy.save(local_state)
                    cancelled=future.result(timeout=15)
                    assert cancelled.status!="ok" and not psutil.pid_exists(child_pid)
                    current=json.loads((local_state/"diagnostics.json").read_text())
                    assert current["pid"]==original_pid and current["peers"]
                    assert ArtifactStore(parent/"artifacts").has(release.model_artifact_id)
                    pause_report={"cancelled_live_engine":True,"same_node_process":True,
                                  "connections_preserved":True,"model_preserved":True}
                finally:
                    policy.paused=False; policy.save(local_state)
        for value in inference: value.pop("logits")
        routes=[{"node_id":p["node_id"],"transport":p["transport"],"encrypted":p["encrypted"],"route":route(p.get("relay_host") or p["host"])} for p in node.peers()]
        success=difference<0.001 and all(p["route"].get("public") and not p["route"].get("vpn") for p in routes)
        report={"success":success,"model_id":release.model_artifact_id,"requester":identity.node_id_hex,
            "inference":inference,"training":training,"inference_max_difference":difference,"routes":routes}
        report["pause"]=pause_report
        (state/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps({"event":"summary","success":success,"inference_max_difference":difference}),flush=True)
        if not success: raise RuntimeError("Persistent compute gate failed")
    finally:
        work.close(); node.stop()


if __name__=="__main__": main()
