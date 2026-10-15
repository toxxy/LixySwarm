"""Typed public-data gradient child; no network identity or personal store."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def run(request):
    from src.learning.language_policy import policy_id,limits,retention_plan,dataset_id
    actions={'gradient':1,'gradient-batch':2,'gradient-half':3,'gradient-math':4}
    if request.get('action') not in actions:raise ValueError('Unknown public gradient action')
    policy_version=actions[request['action']]
    if policy_version>=2 and request.get('training_policy_id')!=policy_id(policy_version):
        raise ValueError('Unknown public batch training policy')
    from src.learning.public_bases import BASES,base_hash
    from src.learning.public_conversation import public_corpus,corpus_digest
    from src.learning.weights_io import digest_file
    corpus=public_corpus();start=request['start'];count=request['count']
    if policy_version==4:
        from src.learning.public_math import load_snapshot,prompt as math_prompt
        if request['base_name']!='qwen3-1.7b' or not Path(request.get('training_file','')).is_absolute():
            raise ValueError('Math training requires its declared expert and locally approved dataset path')
        snapshot=load_snapshot(request['training_file'],'train')
        corpus={'train':[{'prompt':math_prompt(row),'answer':row['answer']} for row in snapshot['rows']]}
    if type(start) is not int or type(count) is not int or not 1<=count<=limits(policy_version)[0] or not 0<=start<len(corpus['train']) or start+count>len(corpus['train']):
        raise ValueError('Invalid public language shard')
    if request['dataset_id']!=dataset_id(policy_version):raise ValueError('Only the exact public corpus is accepted')
    name=request['base_name']
    for filename,identifier in BASES[name].items():
        if digest_file(Path(request['base_directory'])/filename)[0]!=identifier:raise ValueError('Pinned public base mismatch')
    if digest_file(request['adapter_file'])[0]!=request['adapter_id']:raise ValueError('Pinned public adapter mismatch')
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from src.learning.language_adapter import install_adapter,load_adapter,adapter_parameters,adapters_enabled
    from src.learning.language_gradient import save_gradient
    torch.set_num_threads(request['cpu_threads']);torch.manual_seed(20261014)
    torch.cuda.set_per_process_memory_fraction(request['gpu_fraction'])
    tokenizer=AutoTokenizer.from_pretrained(request['base_directory'],local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(request['base_directory'],local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16,device_map={'':'cuda'})
    parameters=install_adapter(model);load_adapter(model,request['adapter_file'],base_hash(name))
    model.config.use_cache=False;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.train()
    if policy_version==4:
        from src.learning.language_math_gradient import backward
        loss_sum,retention_loss,tokens,replay_tokens=backward(model,tokenizer,corpus['train'][start:start+count],'cuda',request['deadline'])
    else:
        encoded=[]
        for row in corpus['train'][start:start+count]:
            text=tokenizer.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            prompt=tokenizer.encode(text,add_special_tokens=False);answer=tokenizer.encode(row['answer'],add_special_tokens=False)+[tokenizer.eos_token_id]
            if len(prompt)+len(answer)>192:raise ValueError('Public language example exceeds its bound')
            encoded.append((torch.tensor([prompt+answer],device='cuda'),torch.tensor([[-100]*len(prompt)+answer],device='cuda'),len(answer)))
        tokens=sum(row[2] for row in encoded);loss_sum=0
        for inputs,labels,n in encoded:
            if time.time()>=request['deadline']:raise TimeoutError('Language gradient deadline exceeded')
            loss=model(input_ids=inputs,labels=labels,use_cache=False).loss
            (loss*n/tokens).backward();loss_sum+=float(loss.detach())*n/tokens
        anchors=retention_plan(start,[row[2] for row in encoded]) if policy_version>=2 else [(start%len(corpus['retention']),1)]
        retention_loss=0
        for anchor_index,weight in anchors:
            if time.time()>=request['deadline']:raise TimeoutError('Language retention deadline exceeded')
            anchor=tokenizer(corpus['retention'][anchor_index],return_tensors='pt',add_special_tokens=False).input_ids.to('cuda')
            with torch.no_grad(),adapters_enabled(model,False):teacher=model(input_ids=anchor,use_cache=False).logits.detach().float()
            student=model(input_ids=anchor,use_cache=False).logits.float()
            retention=torch.nn.functional.kl_div(torch.log_softmax(student,dim=-1),torch.softmax(teacher,dim=-1),reduction='sum')/anchor.shape[1]
            (retention*weight).backward();retention_loss+=float(retention.detach())*weight
        torch.nn.utils.clip_grad_norm_(parameters.values(),1,error_if_nonfinite=True)
    if time.time()>=request['deadline']:raise TimeoutError('Language gradient deadline exceeded')
    path=Path(request['output_file']);path.parent.mkdir(parents=True,exist_ok=True)
    precision=save_gradient(model,path,base_hash=base_hash(name),adapter_id=request['adapter_id'],dataset_id=request['dataset_id'],start=start,count=count,tokens=tokens,policy_version=policy_version)
    return {**({'replay_tokens':replay_tokens} if policy_version==4 else {}),**({'training_policy_id':policy_id(policy_version)} if policy_version>=2 else {}),**precision,'base_hash':base_hash(name),'adapter_id':request['adapter_id'],'dataset_id':request['dataset_id'],
            'start':start,'count':count,'tokens':tokens,'loss':loss_sum,'retention_loss':retention_loss,
            'gradient_id':digest_file(path)[0],'gradient_bytes':path.stat().st_size}


def main():
    raw=sys.stdin.buffer.read(16*1024+1)
    if len(raw)>16*1024:raise ValueError('Language gradient request exceeds its bound')
    print(json.dumps(run(json.loads(raw)),allow_nan=False),flush=True)


if __name__=='__main__':main()
