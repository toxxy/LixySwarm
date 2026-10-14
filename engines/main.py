"""CPU-only engine entry point; no network listener or signing identity."""
from __future__ import annotations

import argparse
import contextlib
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
    from src.network.reference_model import REFERENCE_CONFIG, make_reference_model
    from src.network.training_worker import validate_gradient_artifact

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
        np.save(tokens, np.arange(65, dtype=np.int64))
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
    return {"success": True, "frozen": bool(getattr(sys, "frozen", False)), "torch_version": torch.__version__,
            "cuda_runtime": False, "inference_logits": 256, "gradient_tokens": 64,
            "personal_state_loaded": False, "network_identity_loaded": False}


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
    raw = sys.stdin.buffer.read(16 * 1024 + 1)
    if len(raw) > 16 * 1024:
        raise ValueError("Engine request exceeds its bound")
    request = json.loads(raw)
    if request.get("device") != "cpu":
        raise ValueError("This locally installed package supports CPU work only")
    # The parent node chooses verified artifacts, deadlines and permissions.
    # Tensor imports and computation stay exclusively in this child process.
    from src.network.reference_job import run_request
    run_request(request)


if __name__ == "__main__":
    main()
