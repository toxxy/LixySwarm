"""Bounded offline learned routing over two unchanged public Qwen experts."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from .language_selector_protocol import definition,protocol_id
from .language_retention import PARENT_MODEL,PARENT_RELEASE

os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'


def validate_request(request):
    fields={'action','protocol_id','compute_dtype','streamed_base','base_directory','adapter_file','model_id',
        'parent_release_id','training_file','evaluation_file','output_directory','device','cpu_threads','gpu_fraction','deadline'}
    if (not isinstance(request,dict) or set(request)!=fields or request['action']!=definition()['action']
            or request['protocol_id']!=protocol_id() or request['model_id']!=PARENT_MODEL
            or request['parent_release_id']!=PARENT_RELEASE or request['compute_dtype']!='float32'
            or request['streamed_base'] is not True or request['device'] not in {'cuda','cpu'}):
        raise ValueError('Unknown public expert-selector protocol')
    if (type(request['cpu_threads']) is not int or not 1<=request['cpu_threads']<=16
            or type(request['gpu_fraction']) not in (int,float) or not 0<request['gpu_fraction']<=1
            or type(request['deadline']) not in (int,float) or not math.isfinite(request['deadline'])
            or not time.time()<request['deadline']<=time.time()+definition()['maximum_seconds']):
        raise ValueError('Invalid expert-selector resource bounds')
    for key in ['base_directory','adapter_file','training_file','evaluation_file','output_directory']:
        if not isinstance(request[key],str) or not Path(request[key]).is_absolute():raise ValueError('Selector paths are locally selected absolute paths')
    output=Path(request['output_directory'])
    if output.exists() or output.is_symlink() or output.resolve().is_relative_to(Path(request['base_directory']).resolve()):
        raise ValueError('Selector research cannot replace models or earlier evidence')


def encode_rows(tokenizer,rows):
    encoded=[]
    for row in rows:
        text=tokenizer.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=False,
            add_generation_prompt=True,enable_thinking=False)
        prefix=tokenizer.encode(text,add_special_tokens=False)
        answer=tokenizer.encode(row['answer'],add_special_tokens=False)+[tokenizer.eos_token_id]
        if not prefix or len(prefix)+len(answer)>1024 or len(answer)>512:raise ValueError('Selector cannot truncate public examples')
        encoded.append((prefix,answer))
    return encoded


def expert_losses(model,encoded,device,check,progress):
    import torch
    from .language_adapter import adapters_enabled
    result=[]
    with torch.inference_mode():
        for index,(prefix,answer) in enumerate(encoded):
            check();inputs=torch.tensor([prefix+answer],device=device)
            labels=torch.tensor([[-100]*len(prefix)+answer],device=device)
            losses=[]
            for enabled in [True,False]:
                with adapters_enabled(model,enabled):
                    value=float(model(input_ids=inputs,labels=labels,use_cache=False).loss)
                if not math.isfinite(value) or value<0:raise ValueError('Invalid frozen expert NLL')
                losses.append(value)
            result.append(losses);progress(index+1,len(encoded))
    return torch.tensor(result,dtype=torch.float64)


def run(request):
    validate_request(request);started=time.monotonic();protocol=definition()
    from .public_bases import BASES,base_hash
    from .public_math import load_snapshot,canonical,prompt
    from .public_conversation import public_corpus
    from .weights_io import digest_file
    from .language_retention_job import training_sets
    for name,expected in BASES['qwen3-1.7b'].items():
        if digest_file(Path(request['base_directory'])/name)[0]!=expected:raise ValueError('Selector upstream base mismatch')
    parent_digest=digest_file(request['adapter_file'])
    if parent_digest[0]!=PARENT_MODEL:raise ValueError('Selector accepted parent mismatch')
    corpus=public_corpus();training=corpus['train']+training_sets(load_snapshot(request['training_file'],'train'))['math_train192']
    if len(training)!=392:raise ValueError('Selector TRAIN coverage changed')
    output=Path(request['output_directory']);output.mkdir(parents=True,exist_ok=False)
    def progress(phase,done=0,total=0):
        (output/'progress.json').write_bytes(canonical({'phase':phase,'completed':done,'total':total,'timestamp':time.time()}))
    def check():
        if time.time()>=request['deadline']:raise TimeoutError('Expert-selector deadline exceeded')
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from .language_adapter import install_adapter,load_adapter
    from .language_streamed_float import install_streamed_float32
    from .language_selector import train_selectors,save_selector,load_selector,LanguageExpertSelector,RoutedFrozenExperts,tensor_digest
    torch.set_num_threads(request['cpu_threads']);torch.manual_seed(protocol['seed'])
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if request['device']=='cuda':torch.cuda.set_per_process_memory_fraction(request['gpu_fraction']);torch.cuda.reset_peak_memory_stats()
    progress('load-pinned-experts')
    tokenizer=AutoTokenizer.from_pretrained(request['base_directory'],local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(request['base_directory'],local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16,device_map={'':request['device']})
    parameters=install_adapter(model);load_adapter(model,request['adapter_file'],base_hash('qwen3-1.7b'))
    base_execution=install_streamed_float32(model);model.eval();model.config.use_cache=False
    originals={k:v.detach().clone() for k,v in parameters.items()}
    for p in parameters.values():p.requires_grad_(False)
    try:
        encoded=encode_rows(tokenizer,training)
        teacher=expert_losses(model,encoded,request['device'],check,lambda d,t:progress('train-expert-nll',d,t))
        variants,fit,optimizers=train_selectors([p for p,a in encoded],teacher,model.config.vocab_size,protocol,
            device=request['device'],check=check,progress=progress)
        binding=dict(protocol_id=protocol_id(),base_hash=base_hash('qwen3-1.7b'),tokenizer_id=protocol['tokenizer_id'],
            dolphin_state_id=fit['dolphin_state_id'],feature_space=fit['memory_feature_space'],memory_scope='public-training-only')
        artifacts={};frozen={}
        for name,selector in variants.items():
            check();metadata={**binding,'variant':name};path=output/(name+'.safetensors')
            save_selector(selector,path,metadata);artifacts[path.name]=dict(zip(['sha256','bytes'],digest_file(path)))
            reloaded=LanguageExpertSelector(model.config.vocab_size,support_count=len(fit['support_indices'])).to(request['device'])
            load_selector(reloaded,path,metadata);reloaded.eval()
            if tensor_digest(reloaded.state_dict())!=tensor_digest(selector.state_dict()):raise ValueError('Selector artifact reload changed learned state')
            del reloaded;frozen[name]=tensor_digest(selector.state_dict())
        # Optimizer state is local resumable research evidence, never executable
        # inference content or a peer-selected program.
        torch.save(optimizers,output/'optimizer-state.pt');artifacts['optimizer-state.pt']=dict(zip(['sha256','bytes'],digest_file(output/'optimizer-state.pt')))
        if sum(row['bytes'] for row in artifacts.values())>protocol['disk_reservation_gb']*1024**3:raise ValueError('Selector state exceeded its disk reservation')
        progress('train-frozen-before-heldout')
        (output/'training.json').write_bytes(canonical({'fit':fit,'teacher_nll':teacher.tolist(),'artifacts':artifacts,'binding':binding}))
        # Held-out files are first opened after training, freezing and saving.
        evaluation=load_snapshot(request['evaluation_file'],'evaluation')
        heldout=corpus['evaluation']+[{'prompt':prompt(r),'answer':r['answer']} for r in evaluation['rows']]
        if {r['prompt'] for r in training}&{r['prompt'] for r in heldout}:raise ValueError('Public selector TRAIN and held-out prompts overlap')
        encoded_eval=encode_rows(tokenizer,heldout)
        losses=expert_losses(model,encoded_eval,request['device'],check,lambda d,t:progress('heldout-expert-nll',d,t))
        tokens=torch.tensor([len(a) for p,a in encoded_eval],dtype=torch.float64)
        reports={}
        def summary(values,choices):
            return {key:{'nll':float((values[sl]*tokens[sl]).sum()/tokens[sl].sum()),
                'examples':len(values[sl]),'tokens':int(tokens[sl].sum()),'upstream_base_choices':int(choices[sl].sum())}
                for key,sl in [('historical',slice(0,len(corpus['evaluation']))),('math',slice(len(corpus['evaluation']),None))]}
        for expert,name in enumerate(['accepted-parent','upstream-base']):
            reports[name]=summary(losses[:,expert],torch.full((len(heldout),),expert))
        reports['oracle-only-diagnostic']=summary(losses.min(dim=-1).values,losses.argmin(dim=-1))
        for name,selector in variants.items():
            check();routes=[selector.choose([p],use_memory=name!='dolphin') for p,a in encoded_eval]
            choices=torch.cat([r[0] for r in routes]);probabilities=torch.cat([r[1] for r in routes])
            choices=choices.cpu();expected=losses[torch.arange(len(heldout)),choices]
            actual=[];routed=RoutedFrozenExperts(model,selector,use_memory=name!='dolphin').eval()
            with torch.inference_mode():
                for i,(prefix,answer) in enumerate(encoded_eval):
                    check();inputs=torch.tensor([prefix+answer],device=request['device'])
                    labels=torch.tensor([[-100]*len(prefix)+answer],device=request['device'])
                    actual.append(float(routed(input_ids=inputs,labels=labels,use_cache=False).loss))
                    progress('heldout-routed-'+name,i+1,len(encoded_eval))
            if not torch.equal(torch.tensor(actual,dtype=torch.float64),expected):raise ValueError('Actual selected expert execution differs from declared route')
            if tensor_digest(selector.state_dict())!=frozen[name]:raise ValueError('Held-out evaluation changed the selector')
            reports[name]={**summary(expected,choices),'choices':choices.tolist(),'probabilities':probabilities.cpu().tolist(),
                'single_selected_expert_executed_per_request':True,'prefix_only_routing_verified':True}
        if not all(torch.equal(p,originals[n]) for n,p in parameters.items()):raise ValueError('Frozen language expert parameters changed')
    finally:
        with torch.no_grad():
            for n,p in parameters.items():p.copy_(originals[n]);p.requires_grad_(True)
    if digest_file(request['adapter_file'])!=parent_digest:raise ValueError('Accepted language adapter file changed')
    result={'success':True,'protocol_id':protocol_id(),'model_id':PARENT_MODEL,'parent_release_id':PARENT_RELEASE,
        'base_execution':base_execution,'fit':fit,'selector_artifacts':artifacts,'binding':binding,'evaluations':reports,
        'heldout_expert_nll':losses.tolist(),'heldout_answer_tokens':tokens.tolist(),
        'held_out_data_used_in_compute':True,'held_out_data_used_for_training_or_selection':False,'generation_performed':False,
        'parent_parameters_restored':True,'updated_model_files_written':False,'language_expert_files_written':False,
        'selector_state_files_written':True,'new_model_publication_authorized':False,'native_learning_cycles_added':0,
        'eligible_for_native_promotion':False,'full_generation_gates_evaluated':False,
        'seconds':time.monotonic()-started,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated() if request['device']=='cuda' else 0}
    raw=canonical(result)
    if len(raw)>1024**2:raise ValueError('Selector report exceeds its bound')
    (output/'diagnostic.json').write_bytes(raw);progress('complete')
    return {'success':True,'protocol_id':protocol_id(),'report_id':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}


def main():
    raw=sys.stdin.buffer.read(16*1024+1)
    if len(raw)>16*1024:raise ValueError('Selector request exceeds its bound')
    print(json.dumps(run(json.loads(raw)),allow_nan=False),flush=True)


if __name__=='__main__':main()
