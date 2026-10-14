"""A language-inference child without a transport identity or personal store."""
from __future__ import annotations
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def run(request):
    from src.learning.public_bases import BASES,base_hash
    def digest(path):
        h=hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda:stream.read(1024**2),b""):h.update(block)
        return h.hexdigest()
    for name,expected in BASES[request["base_name"]].items():
        if digest(Path(request["base_directory"])/name)!=expected:raise ValueError("Pinned public base mismatch")
    if digest(request["adapter_file"])!=request["adapter_id"]:raise ValueError("Pinned local adapter mismatch")
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,StoppingCriteria,StoppingCriteriaList
    from src.learning.language_adapter import install_adapter,load_adapter
    torch.set_num_threads(request["cpu_threads"]);torch.manual_seed(20261014)
    device=request.get("device","cuda")
    if device not in {"cpu","cuda"}:raise ValueError("Unknown language device")
    if device=="cuda":torch.cuda.set_per_process_memory_fraction(request["gpu_fraction"])
    dtype=torch.bfloat16 if device=="cpu" or torch.cuda.is_bf16_supported() else torch.float16
    tokenizer=AutoTokenizer.from_pretrained(request["base_directory"],local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(request["base_directory"],local_files_only=True,trust_remote_code=False,
                                             dtype=dtype,device_map={"":device}).eval()
    install_adapter(model);load_adapter(model,request["adapter_file"],base_hash(request["base_name"]))
    text=tokenizer.apply_chat_template(request["messages"],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    inputs=tokenizer(text,return_tensors="pt").to(device)
    if inputs.input_ids.shape[1]>1024:raise ValueError("Conversation exceeds the bounded model context")
    class Deadline(StoppingCriteria):
        def __call__(self,*_,**__):return time.time()>=request["deadline"]
    with torch.inference_mode():output=model.generate(**inputs,max_new_tokens=128,do_sample=False,use_cache=True,stopping_criteria=StoppingCriteriaList([Deadline()]))
    if time.time()>=request["deadline"]:raise RuntimeError("Language deadline expired")
    tokens=output[0,inputs.input_ids.shape[1]:]
    answer=tokenizer.decode(tokens,skip_special_tokens=True).strip()
    if request.get("metrics_file"):
        import resource
        Path(request["metrics_file"]).write_text(json.dumps({"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=="darwin" else 1024),
            "gpu_allocated_bytes":torch.cuda.max_memory_allocated() if device=="cuda" else 0},indent=2)+"\n")
    print(json.dumps({"model_id":request["adapter_id"],"answer":answer,"output_tokens":int(tokens.numel())},ensure_ascii=False,allow_nan=False),flush=True)


def main():
    raw=sys.stdin.buffer.read(16*1024+1)
    if len(raw)>16*1024:raise ValueError("Language job exceeds its bound")
    run(json.loads(raw))


if __name__=="__main__":main()
