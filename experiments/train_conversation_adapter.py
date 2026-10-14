"""Bounded public SFT research gate, before any network model activation.

Training data, generated checkpoints and reports stay outside Dropbox. This
local experiment does not claim distributed learning or conversational quality.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model-directory",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--steps",type=int,default=60)
    parser.add_argument("--retention-weight",type=float,default=1.0)
    parser.add_argument("--model-size",choices=["0.6b","1.7b"],default="0.6b")
    parser.add_argument("--direct-device-load",action="store_true")
    options=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    if options.output.resolve().is_relative_to(repo) or not 1<=options.steps<=200 or not 0<=options.retention_weight<=10:
        parser.error("Bounded research outputs must be outside the checkout")
    options.output.mkdir(parents=True,exist_ok=True)
    from experiments.evaluate_conversation_seed import PINNED_FILES,MODEL_REVISION,LARGER_MODEL,CASES,file_hash
    repository="Qwen/Qwen3-0.6B";revision=MODEL_REVISION;pinned=PINNED_FILES
    if options.model_size=="1.7b":repository,revision,pinned=LARGER_MODEL["repository"],LARGER_MODEL["revision"],LARGER_MODEL["files"]
    base_hash=PINNED_FILES["model.safetensors"] if options.model_size=="0.6b" else hashlib.sha256(json.dumps(pinned,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    from src.learning.public_conversation import public_corpus,corpus_digest
    corpus=public_corpus()
    if any(row["prompt"]==case[1] for row in corpus["train"] for case in CASES):
        raise ValueError("Training must not include the frozen instruction gate")
    (options.output/"public-dataset.json").write_text(json.dumps(corpus,ensure_ascii=False,sort_keys=True)+"\n")
    for name,expected in pinned.items():
        if file_hash(options.model_directory/name)!=expected:raise ValueError(f"Pinned base mismatch: {name}")
    print(json.dumps({"event":"public_inputs_verified","corpus":corpus_digest(corpus),"train":len(corpus["train"]),"evaluation":len(corpus["evaluation"])}),flush=True)
    import torch
    import psutil
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from src.learning.language_adapter import install_adapter,save_adapter,adapter_parameters,adapters_enabled
    torch.set_num_threads(4);torch.manual_seed(20261014)
    torch.cuda.set_per_process_memory_fraction(0.25);torch.cuda.reset_peak_memory_stats()
    tokenizer=AutoTokenizer.from_pretrained(options.model_directory,local_files_only=True,trust_remote_code=False)
    loading={"device_map":{"":"cuda"}} if options.direct_device_load else {}
    model=AutoModelForCausalLM.from_pretrained(options.model_directory,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,**loading).to("cuda").eval()
    parameters=install_adapter(model)
    def encode(row):
        text=tokenizer.apply_chat_template([{"role":"user","content":row["prompt"]}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        prompt=tokenizer.encode(text,add_special_tokens=False)
        answer=tokenizer.encode(row["answer"],add_special_tokens=False)+[tokenizer.eos_token_id]
        if len(prompt)+len(answer)>192:raise ValueError("Public example exceeds the bounded sequence")
        x=torch.tensor([prompt+answer],device="cuda")
        y=torch.tensor([[-100]*len(prompt)+answer],device="cuda")
        return x,y,len(answer)
    encoded={split:[encode(row) for row in corpus[split]] for split in ["train","evaluation"]}
    anchors=[tokenizer(text,return_tensors="pt",add_special_tokens=False).input_ids.to("cuda") for text in corpus["retention"]]
    def evaluate():
        model.eval();total=count=0
        with torch.inference_mode():
            for x,y,n in encoded["evaluation"]:
                total+=float(model(input_ids=x,labels=y,use_cache=False).loss)*n;count+=n
        torch.manual_seed(20261014);answers=[]
        for name,prompt,pattern in CASES:
            text=tokenizer.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            inputs=tokenizer(text,return_tensors="pt").to("cuda")
            with torch.inference_mode():output=model.generate(**inputs,max_new_tokens=96,do_sample=False,use_cache=True)
            answer=tokenizer.decode(output[0,inputs.input_ids.shape[1]:],skip_special_tokens=True).strip()
            answers.append({"case":name,"answer":answer,"passed":re.search(pattern,answer) is not None})
        return {"heldout_nll":total/count,"heldout_tokens":count,"instruction_passed":sum(row["passed"] for row in answers),"cases":answers}
    baseline=evaluate();(options.output/"baseline.json").write_text(json.dumps(baseline,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"event":"baseline","nll":baseline["heldout_nll"],"gate":baseline["instruction_passed"]}),flush=True)
    model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    optimizer=torch.optim.AdamW(parameters.values(),lr=0.0002,weight_decay=0)
    rng=random.Random(20261014);order=list(range(len(encoded["train"])));rng.shuffle(order)
    losses=[];began=time.monotonic();process=psutil.Process();peak_rss=process.memory_info().rss
    for step in range(options.steps):
        model.train();optimizer.zero_grad(set_to_none=True)
        indices=[order[(step*4+i)%len(order)] for i in range(4)]
        total=sum(encoded["train"][index][2] for index in indices);loss_sum=0
        for index in indices:
            x,y,n=encoded["train"][index]
            loss=model(input_ids=x,labels=y,use_cache=False).loss
            if not torch.isfinite(loss):raise ValueError("Nonfinite public training loss")
            (loss*n/total).backward();loss_sum+=float(loss.detach())*n/total
        if options.retention_weight:
            anchor=anchors[step%len(anchors)]
            with torch.no_grad(),adapters_enabled(model,False):
                teacher=model(input_ids=anchor,use_cache=False).logits.detach().float()
                teacher_prob=torch.softmax(teacher,dim=-1)
            student=model(input_ids=anchor,use_cache=False).logits.float()
            retention=torch.nn.functional.kl_div(torch.log_softmax(student,dim=-1),teacher_prob,
                                                reduction="sum")/anchor.shape[1]
            if not torch.isfinite(retention):raise ValueError("Nonfinite retention loss")
            (retention*options.retention_weight).backward()
        torch.nn.utils.clip_grad_norm_(parameters.values(),1,error_if_nonfinite=True);optimizer.step()
        losses.append(loss_sum);peak_rss=max(peak_rss,process.memory_info().rss)
        if (step+1)%5==0:print(json.dumps({"event":"training","step":step+1,"loss":loss_sum,"elapsed_s":round(time.monotonic()-began,1)}),flush=True)
    model.gradient_checkpointing_disable();model.config.use_cache=True
    candidate=evaluate();path=options.output/"adapter.safetensors"
    save_adapter(model,path,base_hash)
    required={"es_instruction","en_instruction","es_public_context","en_public_context"}
    previous_correct={row["case"] for row in baseline["cases"] if row["passed"]}
    regressions=[row["case"] for row in candidate["cases"] if row["case"] in previous_correct and not row["passed"]]
    accepted=(candidate["heldout_nll"]<baseline["heldout_nll"]-0.001 and candidate["instruction_passed"]>=11
              and not regressions and all(row["passed"] for row in candidate["cases"] if row["case"] in required))
    report={"schema_version":1,"scope":"local public SFT research; not network promotion", "base_repository":repository,
            "base_revision":revision,"base_model_hash":base_hash,
            "dataset_hash":corpus_digest(corpus),"private_inputs":False,"training_examples":len(corpus["train"]),
            "evaluation_examples":len(corpus["evaluation"]),"steps":options.steps,"gradient_accumulation":4,
            "learning_rate":0.0002,"trainable_parameters":sum(p.numel() for p in parameters.values()),
            "retention_weight":options.retention_weight,"regressions":regressions,
            "direct_device_load":options.direct_device_load,
            "adapter_sha256":file_hash(path),"adapter_bytes":path.stat().st_size,"baseline":baseline,"candidate":candidate,
            "minimum_nll_improvement":0.001,"minimum_instruction_passed":11,"eligible_for_integration":accepted,
            "peak_gpu_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_observed_rss_bytes":peak_rss,
            "training_seconds":time.monotonic()-began,"losses":losses,
            "limitations":["Same templates with disjoint values in the 72-example NLL evaluation", "Only an elementary 12-case instruction gate", "No distributed-learning, privacy-isolation or broad-conversation claim"]}
    (options.output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"event":"candidate","nll":candidate["heldout_nll"],"gate":candidate["instruction_passed"],"eligible":accepted,"adapter_bytes":path.stat().st_size}),flush=True)


if __name__=="__main__":main()
