"""CPU-only engine entry point; no network listener or signing identity."""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import replace
import hashlib
import io
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def self_test():
    import numpy as np
    import torch
    from src.network.reference_model import REFERENCE_CONFIG, make_reference_model, reference_corpus
    from src.network.artifact_store import ArtifactStore
    from src.network.lsp import LSPIdentity
    from src.network.work_protocol import WorkResult, ResultReceipt
    from src.network.training_worker import validate_gradient_artifact
    from src.release import ReleaseManifest, ReleaseRegistry, TrustPolicy

    if torch.version.cuda is not None:
        raise ValueError("The CPU package must not include a CUDA runtime")
    torch.set_num_threads(2)
    torch.manual_seed(20261014)
    with contextlib.redirect_stdout(sys.stderr):
        model = make_reference_model(REFERENCE_CONFIG)
    with tempfile.TemporaryDirectory(prefix="lixy-engine-check-") as directory:
        root = Path(directory)
        checkpoint = root / "model.pt"
        torch.save({"format": 1, "config": REFERENCE_CONFIG, "model": model.state_dict()}, checkpoint)
        model_id = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        tokens = root / "tokens.npy"
        public_tokens, _ = reference_corpus()
        np.save(tokens, public_tokens, allow_pickle=False)
        dataset_id = hashlib.sha256(tokens.read_bytes()).hexdigest()
        command = [sys.executable] if getattr(sys, "frozen", False) else [sys.executable, str(Path(__file__).resolve())]
        request = {"device": "cpu", "cpu_threads": 2, "model_path": str(checkpoint),
                   "output_directory": str(root / "job"), "deadline": time.time() + 45,
                   "dataset_path": str(tokens)}
        request["action"] = "inference"
        request["payload"] = {"model_artifact_id": model_id, "prompt": "red"}
        inference = subprocess.run(command, input=json.dumps(request), text=True, capture_output=True, timeout=45, check=True)
        result = json.loads(inference.stdout)
        if result["model_id"] != model_id or len(result["next_byte_logits"]) != 256 or not all(math.isfinite(x) for x in result["next_byte_logits"]):
            raise ValueError("Frozen inference did not produce valid byte logits")
        request["action"] = "training"
        request["payload"] = {"model_artifact_id": model_id, "dataset_artifact_id": dataset_id,
                              "start_token": 0, "token_count": 65}
        gradient = subprocess.run(command, input=json.dumps(request), text=True, capture_output=True, timeout=45, check=True)
        result = json.loads(gradient.stdout)
        path = Path(result["gradient_path"])
        validate_gradient_artifact(path, model, expected_model_id=model_id, expected_dataset_id=dataset_id,
                                   expected_start_token=0, expected_token_count=65)
        if hashlib.sha256(path.read_bytes()).hexdigest() != result["gradient"]["artifact_id"]:
            raise ValueError("Frozen gradient hash mismatch")
        # These are test-only signing fixtures, not connected physical nodes.
        # Only public keys reach the compute child; no signing keys or personal
        # state are loaded by ordinary inference/training/candidate jobs.
        publisher = LSPIdentity.generate()
        workers = [LSPIdentity.generate(), LSPIdentity.generate()]
        store = ArtifactStore(root / "public" / "artifacts")
        base = store.import_file(checkpoint, kind="model")
        genesis = ReleaseManifest.create(model_artifact_id=base.artifact_id,
                                        model_format="pytorch-weights-only-v1", sequence=0).sign(publisher)
        policy = TrustPolicy(trusted_signers=(publisher.node_id_hex,), pinned_genesis_release_id=genesis.release_id)
        registry = ReleaseRegistry(root / "public" / "releases")
        registry.accept(genesis, policy, store)
        registry.activate(genesis.release_id, policy, store)
        contributions = []
        for index, worker in enumerate(workers):
            start = index * 7200
            if index:
                request["output_directory"] = str(root / "second-job")
                request["payload"]["start_token"] = start
                gradient = subprocess.run(command, input=json.dumps(request), text=True, capture_output=True, timeout=45, check=True)
                result = json.loads(gradient.stdout)
            imported = store.import_file(Path(result.pop("gradient_path")), kind="gradient", media_type="application/x-npz")
            result["gradient"] = imported.to_dict()
            work = WorkResult(job_id=hashlib.sha256(f"package-check-{index}".encode()).hexdigest(), status="ok", output=result)
            work = replace(work, receipt=ResultReceipt.create(work, worker, publisher.node_id_hex).to_dict())
            contributions.append({"peer_id": worker.node_id_hex, "start_token": start, "token_count": 65,
                                  "gradient_id": imported.artifact_id, "result": work.to_dict()})
        from dataclasses import asdict
        candidate_request = {"action": "candidate", "device": "cpu", "cpu_threads": 2,
                             "deadline": time.time() + 45, "output_directory": str(root / "candidate"),
                             "public_directory": str(root / "public"), "store_quota": 32 * 1024**2,
                             "trust_policy": asdict(policy), "base_release": genesis.to_dict(),
                             "requester_node_id": publisher.node_id_hex,
                             "trusted_workers": [worker.node_id_hex for worker in workers],
                             "dataset_id": dataset_id, "contributions": contributions}
        candidate_process = subprocess.run(command, input=json.dumps(candidate_request), text=True, capture_output=True, timeout=45, check=True)
        candidate = json.loads(candidate_process.stdout)
        if not candidate["accepted"] or candidate["release"]["signatures"] or registry.active().release_id != genesis.release_id:
            raise ValueError("Frozen evaluator did not return an unsigned improvement without promotion")
    from src.learning.swarm_foundation import make_foundation_model,predict
    from src.learning.swarm_distributed_job import batches
    with contextlib.redirect_stdout(sys.stderr):
        own=make_foundation_model().train()
    inputs,labels,tokens=next(batches([b'Own public fixture.'],'cpu'))
    _,own_loss,_=predict(own,inputs,labels);own_loss.backward()
    if not torch.isfinite(own_loss) or tokens!=18 or not any(p.grad is not None for p in own.parameters()):
        raise ValueError('Frozen own-swarm backward did not produce a valid gradient')
    return {"success": True, "frozen": bool(getattr(sys, "frozen", False)), "torch_version": torch.__version__,
            "cuda_runtime": False, "inference_logits": 256, "gradient_tokens": 64,
            "candidate_evaluated": True, "candidate_signed_or_promoted": False,
            "candidate_loss_before": candidate["report"]["before_loss"],
            "candidate_loss_after": candidate["report"]["after_loss"],
            "test_fixture_signers": 3, "personal_state_loaded": False,
            "job_process_signing_keys_loaded": False,'own_swarm_backward_verified':True,
            'own_swarm_parameters':sum(p.numel() for p in own.parameters())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--report", type=Path)
    options = parser.parse_args()
    if options.self_test:
        report = self_test()
        if options.report:
            options.report.parent.mkdir(parents=True, exist_ok=True)
            options.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        return
    raw = sys.stdin.buffer.read(64 * 1024 + 1)
    if len(raw) > 64 * 1024:
        raise ValueError("Engine request exceeds its bound")
    request = json.loads(raw)
    if len(raw)>16*1024 and request.get('action') not in {'own-candidate','own-audit'}:
        raise ValueError('Engine request exceeds its operation bound')
    if request.get("device") != "cpu":
        raise ValueError("This locally installed package supports CPU work only")
    if request.get('action') == 'own-validate':
        from src.learning.swarm_bootstrap_job import run_request as validate_own
        return validate_own(request)
    if request.get('action') == 'own-audit':
        from src.learning.swarm_state_audit import run_request as audit_own
        return audit_own(request)
    if request.get('action') in {'own-gradient','own-candidate'}:
        from src.learning.swarm_distributed_job import run_request as run_own
        return run_own(request)
    # The parent node chooses verified artifacts, deadlines and permissions.
    # Tensor imports and computation stay exclusively in this child process.
    from src.network.reference_job import run_request
    run_request(request)


if __name__ == "__main__":
    main()
