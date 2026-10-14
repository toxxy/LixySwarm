"""One supervised reference job; this process has no network identity."""
from __future__ import annotations
import contextlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main():
    raw=sys.stdin.buffer.read(16*1024+1)
    if len(raw)>16*1024:
        raise ValueError("Compute request exceeds its bound")
    request=json.loads(raw)
    import numpy as np
    import torch
    from src.network.reference_model import load_reference_checkpoint, reference_logits
    from src.network.artifact_store import ArtifactStore
    from src.network.training_worker import TrainingWorker
    torch.set_num_threads(request["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261014)
    device=request["device"]
    if device=="cuda":
        torch.cuda.set_per_process_memory_fraction(request["gpu_fraction"])
    with contextlib.redirect_stdout(sys.stderr):
        model=load_reference_checkpoint(Path(request["model_path"]),device)
    payload=request["payload"]
    if request["action"]=="inference":
        output={"model_id":payload["model_artifact_id"],
                "next_byte_logits":reference_logits(model,payload["prompt"].encode())}
    elif request["action"]=="training":
        tokens=np.load(request["dataset_path"],mmap_mode="r",allow_pickle=False)
        start,count=payload["start_token"],payload["token_count"]
        if tokens.ndim!=1 or tokens.dtype.kind not in {"i","u"} or start+count>len(tokens):
            raise ValueError("Invalid public token range")
        batch=np.asarray(tokens[start:start+count],dtype=np.int64)
        if batch.min()<0 or batch.max()>=256:
            raise ValueError("Invalid byte vocabulary")
        store=ArtifactStore(Path(request["output_directory"])/"artifacts",max_artifact_bytes=16*1024**2,max_total_bytes=32*1024**2)
        class Coordinator:
            def register_handler(self,*_): pass
        worker=TrainingWorker(Coordinator(),SimpleNamespace(store=store),model,
            model_artifact_id=payload["model_artifact_id"],device=device,max_batch_tokens=65)
        def check():
            if time.time()>=request["deadline"]:
                raise RuntimeError("Job deadline reached")
        execution=SimpleNamespace(raise_if_cancelled=check)
        x=torch.from_numpy(batch[:-1].copy()).unsqueeze(0).to(device)
        y=torch.from_numpy(batch[1:].copy()).unsqueeze(0).to(device)
        output=worker._compute_gradient(x,y,payload["dataset_artifact_id"],start,count,execution)
        output["gradient_path"]=str(store.object_path(output["gradient"]["artifact_id"]))
    else:
        raise ValueError("Unsupported reference job")
    print(json.dumps(output,allow_nan=False),flush=True)


if __name__=="__main__":
    main()
