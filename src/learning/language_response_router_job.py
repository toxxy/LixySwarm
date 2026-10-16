"""Fit utility pings/public memory using pinned TRAIN drafts; never open TEST."""
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from .language_response_router import definition,protocol_id,feature_id,training_rows,response_messages,select_secondary
from .language_retention import PARENT_MODEL,PARENT_RELEASE
from .language_repair_comparison_job import durable_json
from .public_math import canonical


def validate_request(request):
    p=definition();fields={'action','protocol_id','base_directory','adapter_file','model_id','parent_release_id',
        'training_file','outcome_files','repair_files','output_directory','device','cpu_threads','gpu_fraction','deadline'}
    if (not isinstance(request,dict) or set(request)!=fields or request['action']!=p['action']
            or request['protocol_id']!=protocol_id() or request['model_id']!=PARENT_MODEL
            or request['parent_release_id']!=PARENT_RELEASE or request['device'] not in {'cpu','cuda'}):
        raise ValueError('Response utility fit requires only the pinned public TRAIN program')
    if (type(request['cpu_threads']) is not int or not 1<=request['cpu_threads']<=16
            or type(request['gpu_fraction']) not in (int,float) or not 0<request['gpu_fraction']<=1
            or type(request['deadline']) not in (int,float) or not math.isfinite(request['deadline'])
            or not time.time()<request['deadline']<=time.time()+p['maximum_seconds']):
        raise ValueError('Invalid public utility resource bounds')
    paths=[request[k] for k in ['base_directory','adapter_file','training_file','output_directory']]
    for key in ['outcome_files','repair_files']:
        if not isinstance(request[key],list) or len(request[key])!=2:raise ValueError('Both physical TRAIN artifacts must be selected locally')
        paths+=request[key]
    if any(not isinstance(v,str) or not Path(v).is_absolute() for v in paths):raise ValueError('Only explicit absolute local inputs are permitted')
    output=Path(request['output_directory']).resolve()
    if (output.exists() or output.is_relative_to(Path.home()/'Dropbox')
            or output.is_relative_to(Path(__file__).resolve().parents[2])
            or any(output.is_relative_to(Path(v).resolve()) or Path(v).resolve().is_relative_to(output)
                for v in paths if v!=request['output_directory'])):
        raise ValueError('Response utility fit must preserve separate fresh external state')


def encode_response(tokenizer,row):
    p=definition();messages=response_messages(row)
    if sum(len(v['content'].encode()) for v in messages)>p['maximum_messages_bytes']:
        raise ValueError('Response representation cannot truncate its public context')
    prefix=tokenizer.encode(tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False,
        enable_thinking=False),add_special_tokens=False)
    question=tokenizer.encode(tokenizer.apply_chat_template(messages[:1],tokenize=False,add_generation_prompt=True,
        enable_thinking=False),add_special_tokens=False)
    tokens=row['draft_output_tokens']
    if (not 1<=len(prefix)<=p['maximum_input_tokens'] or not 1<=len(question)<=p['maximum_input_tokens']
            or type(tokens) is not int or not 1<=tokens<=384):
        raise ValueError('Response or token-length feature exceeds its declared bound')
    return prefix,[len(question)/1024,tokens/384]


def _run(request):
    p=definition();began=time.monotonic()
    from .public_math import load_snapshot
    from .public_bases import BASES,base_hash
    from .weights_io import digest_file
    base=Path(request['base_directory'])
    for name,identifier in BASES['qwen3-1.7b'].items():
        if digest_file(base/name)[0]!=identifier:raise ValueError('Response encoder base differs from the public parent')
    if digest_file(request['adapter_file'])[0]!=PARENT_MODEL:raise ValueError('Response encoder parent is not accepted model 2')
    snapshot=load_snapshot(request['training_file'],'train')
    rows=training_rows(snapshot,request['outcome_files'],request['repair_files'])
    output=Path(request['output_directory']);output.mkdir(mode=0o700,parents=True)
    def check():
        if time.time()>=request['deadline']:raise TimeoutError('Public response utility fit exceeded its deadline')
    def progress(phase,done=0,total=0):
        check();durable_json(output/'progress.json',{'phase':phase,'done':done,'total':total,'timestamp':time.time()})
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from safetensors.torch import save_file
    from .language_adapter import install_adapter,load_adapter
    from .language_streamed_float import install_streamed_float32
    from .language_response_selector import fit_projection,project,train,ResponseSelector,binding,save,load
    torch.set_num_threads(request['cpu_threads']);torch.manual_seed(p['seed']);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if request['device']=='cuda':torch.cuda.set_per_process_memory_fraction(request['gpu_fraction']);torch.cuda.reset_peak_memory_stats()
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16,device_map={'':request['device']}).eval()
    parameters=install_adapter(model);load_adapter(model,request['adapter_file'],base_hash('qwen3-1.7b'))
    execution=install_streamed_float32(model);before={k:v.detach().clone() for k,v in parameters.items()}
    for parameter in model.parameters():parameter.requires_grad_(False)
    hidden=[];lengths=[];feature_started=time.monotonic()
    with torch.inference_mode():
        for index,row in enumerate(rows):
            check();prefix,length=encode_response(tokenizer,row);ids=torch.tensor([prefix],device=request['device'])
            result=model.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True)
            vector=result.last_hidden_state[0,-1].float().cpu().contiguous()
            if vector.shape!=(2048,) or not torch.isfinite(vector).all():raise ValueError('Public response feature space changed')
            hidden.append(vector);lengths.append(length);progress('public-train-response-features',index+1,len(rows))
    hidden=torch.stack(hidden).clone();lengths=torch.tensor(lengths,dtype=torch.float32)
    targets=torch.tensor([v['target'] for v in rows],dtype=torch.float32)
    feature_seconds=time.monotonic()-feature_started
    common={'protocol_id':protocol_id(),'feature_space_id':feature_id(),'scope':'public-training-only'}
    save_file({'hidden':hidden,'lengths':lengths,'targets':targets},str(output/'train-features.safetensors'),
        metadata={'format':'lixyswarm-public-response-features-v1','binding':canonical(common).decode()})
    projection=fit_projection(hidden,lengths)
    save_file(projection,str(output/'projection.safetensors'),
        metadata={'format':'lixyswarm-public-response-projection-v1','binding':canonical(common).decode()})
    projection_id=digest_file(output/'projection.safetensors')[0]
    features=project(hidden,lengths,projection).to(request['device'])
    variants,fit,optimizers=train(features,targets.to(request['device']),check=check,progress=progress)
    torch.save(optimizers,output/'optimizer-state.pt')
    artifacts={};bindings={};predictions={};maximum_error=0.
    for name,net in variants.items():
        metadata=binding(net,name,projection_id);path=output/(name+'.safetensors');save(net,path,metadata);bindings[name]=metadata
        restored=ResponseSelector(len(net.public_memory)).to(request['device']);load(restored,path,metadata);restored.eval()
        with torch.no_grad():
            expected=net(features,use_memory=name!='dolphin');actual=restored(features,use_memory=name!='dolphin')
            error=float((expected-actual).abs().max());maximum_error=max(maximum_error,error)
            if error>1e-6:raise ValueError('Learned response state failed its fresh reload')
            predictions[name]=actual.cpu().tolist()
    for name in ['train-features.safetensors','projection.safetensors','optimizer-state.pt',*[(v+'.safetensors') for v in variants]]:
        identifier,size=digest_file(output/name);artifacts[name]={'sha256':identifier,'bytes':size}
    if sum(v['bytes'] for v in artifacts.values())>8*1024**2:raise ValueError('Learned response state exceeded its declared bound')
    if (not all(torch.equal(v,before[k]) for k,v in parameters.items())
            or digest_file(request['adapter_file'])[0]!=PARENT_MODEL):raise ValueError('Response utility fit changed its frozen language parent')
    report={'success':True,'protocol_id':protocol_id(),'feature_space_id':feature_id(),'base_execution':execution,
        'training_indices':[v['index'] for v in rows],'case_ids':[v['case'] for v in rows],
        'artifacts':artifacts,'bindings':bindings,'fit':fit,'train_predictions':predictions,
        'maximum_reload_error':maximum_error,'feature_seconds':feature_seconds,'seconds':time.monotonic()-began,
        'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated() if request['device']=='cuda' else 0,
        'fresh_evaluation_performed':False,'useful_routing_proven':False,'language_experts_updated':False,
        'model_promoted':False,'native_cycles_added':0}
    durable_json(output/'diagnostic.json',report);raw=(output/'diagnostic.json').read_bytes()
    if len(raw)>256*1024:raise ValueError('Response fit evidence exceeded its bounded report size')
    return {'success':True,'protocol_id':protocol_id(),'report_id':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}


def run(request):
    validate_request(request)
    try:return _run(request)
    except Exception as error:
        # Only the output path checked absent and external above may receive
        # this local diagnostic. Never publish it as training or model data.
        output=Path(request['output_directory'])
        if output.is_dir():
            import traceback
            durable_json(output/'failure.json',{'error_type':type(error).__name__,'error':str(error)[:4096],
                'traceback':traceback.format_exc(limit=12)[-16*1024:]})
        raise


def main():
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    raw=sys.stdin.buffer.read(16*1024+1)
    if len(raw)>16*1024:raise ValueError('Public response utility request exceeds its bound')
    with contextlib.redirect_stdout(sys.stderr):response=run(json.loads(raw))
    print(json.dumps(response,allow_nan=False),flush=True)


if __name__=='__main__':main()
