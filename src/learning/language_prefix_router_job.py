"""Fit a small public selector, commit new routes, then measure fresh answers."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from .language_prefix_router import definition,protocol_id,feature_id,verify_predictions,verify_report
from .language_retention import PARENT_MODEL,PARENT_RELEASE
from .language_specialist_answers import SPECIALIST_ID
from .language_outcome_selector import OUTCOME_IDS,dataset
from .public_math import canonical


def validate_request(request):
    from .language_selector_job import validate_request as common_validate
    extras={'outcome_files','specialist_file'}
    if not isinstance(request,dict) or not extras<=set(request) or 'evaluation_file' in request:
        raise ValueError('Prefix component requires exact TRAIN inputs and forbids original EVAL')
    common={k:v for k,v in request.items() if k not in extras};common['evaluation_file']=common.get('training_file')
    common_validate(common,protocol=definition(),identifier=protocol_id())
    if (not isinstance(request['outcome_files'],list) or len(request['outcome_files'])!=2
            or any(not isinstance(v,str) or not Path(v).is_absolute() for v in [request['specialist_file'],*request['outcome_files']])):
        raise ValueError('Prefix calibration paths must be selected locally')
    output=Path(request['output_directory']).resolve()
    inputs=[request[k] for k in ['base_directory','adapter_file','training_file','specialist_file']]+request['outcome_files']
    if (output.is_relative_to(Path.home()/'Dropbox') or output.is_relative_to(Path(__file__).resolve().parents[2])
            or any(output.is_relative_to(Path(v).resolve()) or Path(v).resolve().is_relative_to(output) for v in inputs)):
        raise ValueError('Prefix component requires separate state outside the checkout and Dropbox')


def encode_prefixes(tokenizer,rows,*,vocab_size):
    from .public_math import prompt
    prefixes=[]
    for row in rows:
        text=tokenizer.apply_chat_template([{'role':'user','content':prompt(row)}],tokenize=False,
            add_generation_prompt=True,enable_thinking=False)
        ids=tokenizer.encode(text,add_special_tokens=False)
        if not ids or len(ids)>1024 or any(type(t) is not int or not 0<=t<vocab_size for t in ids):
            raise ValueError('Frozen prefix exceeds its exact untruncated bound')
        prefixes.append(ids)
    return prefixes


def extract_prefix_features(model,prefixes,*,dimensions,device,check,progress):
    import torch
    values=[]
    with torch.inference_mode():
        for index,prefix in enumerate(prefixes):
            check();inputs=torch.tensor([prefix],device=device)
            result=model.model(input_ids=inputs,attention_mask=torch.ones_like(inputs),use_cache=False,return_dict=True)
            vector=result.last_hidden_state[0,-1].detach().float().cpu().contiguous()
            if vector.shape!=(dimensions,) or not torch.isfinite(vector).all():raise ValueError('Frozen query representation differs from its declared space')
            values.append(vector);progress(index+1,len(prefixes))
    return torch.stack(values)


def generate_prefix(model,tokenizer,prefix,*,device,deadline,maximum):
    import torch
    from transformers import StoppingCriteria,StoppingCriteriaList
    class Deadline(StoppingCriteria):
        def __call__(self,*_a,**_kw):return time.time()>=deadline
    if time.time()>=deadline:raise TimeoutError('Frozen prefix generation deadline exceeded')
    inputs=torch.tensor([prefix],device=device)
    with torch.inference_mode():
        if str(device).startswith('cuda'):torch.cuda.synchronize()
        started=time.monotonic()
        result=model.generate(input_ids=inputs,attention_mask=torch.ones_like(inputs),do_sample=False,
            max_new_tokens=maximum,use_cache=True,stopping_criteria=StoppingCriteriaList([Deadline()]))
        if str(device).startswith('cuda'):torch.cuda.synchronize()
        elapsed=time.monotonic()-started
    if time.time()>=deadline:raise TimeoutError('Frozen prefix generation deadline exceeded')
    answer=tokenizer.decode(result[0,len(prefix):],skip_special_tokens=True).strip()
    tokens=int(result.shape[1]-len(prefix))
    if not 1<=tokens<=maximum or len(answer.encode())>8192:raise ValueError('Frozen prefix answer exceeds its bound')
    return {'answer':answer,'output_tokens':tokens,'generation_seconds':elapsed}


def run(request):
    validate_request(request);p=definition();began=time.monotonic()
    from .public_bases import BASES,base_hash
    from .public_math import load_snapshot
    from .weights_io import digest_file
    from src.network.telemetry import atomic_json
    base=Path(request['base_directory'])
    for name,identifier in BASES['qwen3-1.7b'].items():
        if digest_file(base/name)[0]!=identifier:raise ValueError('Prefix representation base differs from its public pin')
    if digest_file(request['adapter_file'])[0]!=PARENT_MODEL or digest_file(request['specialist_file'])[0]!=SPECIALIST_ID:
        raise ValueError('Prefix component cannot substitute the frozen experts')
    snapshot=load_snapshot(request['training_file'],'train');calibration=[]
    for path,identifier in zip(request['outcome_files'],OUTCOME_IDS):
        if Path(path).stat().st_size>128*1024 or digest_file(path)[0]!=identifier:raise ValueError('Prefix calibration outcomes differ from their physical artifacts')
        calibration.append(json.loads(Path(path).read_text()))
    rows=dataset(calibration,snapshot)
    output=Path(request['output_directory']);output.mkdir(parents=True,exist_ok=False,mode=0o700)
    def check():
        if time.time()>=request['deadline']:raise TimeoutError('Frozen prefix component exceeded its deadline')
    def progress(phase,done=0,total=0):
        check();atomic_json(output/'progress.json',{'phase':phase,'done':done,'total':total,'timestamp':time.time()})
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from safetensors.torch import save_file
    from .language_adapter import install_adapter,load_adapter
    from .language_streamed_float import install_streamed_float32
    from .language_specialist import FrozenAdapterBank
    from .language_dense_selector import train_dense_selectors,DenseExpertSelector,binding,save_dense,load_dense
    from .language_selector import tensor_digest
    torch.set_num_threads(request['cpu_threads']);torch.manual_seed(p['seed']);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if request['device']=='cuda':torch.cuda.set_per_process_memory_fraction(request['gpu_fraction']);torch.cuda.reset_peak_memory_stats()
    progress('load-frozen-query-encoder')
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16,device_map={'':request['device']}).eval()
    parameters=install_adapter(model);load_adapter(model,request['adapter_file'],base_hash('qwen3-1.7b'))
    execution=install_streamed_float32(model);model.config.use_cache=False
    parent={name:v.detach().clone() for name,v in parameters.items()}
    try:
        load_adapter(model,request['specialist_file'],base_hash('qwen3-1.7b'))
        specialist={name:v.detach().clone() for name,v in parameters.items()}
    finally:
        with torch.no_grad():
            for name,v in parameters.items():v.copy_(parent[name]);v.requires_grad_(False)
    bank=FrozenAdapterBank(model,parameters,[parent,specialist])
    selected=[snapshot['rows'][i] for i in p['fit_indices']+p['prediction_indices']]
    prefixes=encode_prefixes(tokenizer,selected,vocab_size=model.config.vocab_size)
    progress('frozen-prefix-features',0,len(prefixes));feature_started=time.monotonic()
    features=extract_prefix_features(model,prefixes,dimensions=p['feature_definition']['dimensions'],device=request['device'],
        check=check,progress=lambda d,t:progress('frozen-prefix-features',d,t))
    feature_seconds=time.monotonic()-feature_started
    feature_binding={'feature_space_id':feature_id(),'training_artifact_id':p['training_artifact_id'],
        'case_ids':[r['case'] for r in selected],'answer_tokens_used':False}
    save_file({'features':features},str(output/'prefix-features.safetensors'),metadata={'format':'lixyswarm-public-prefix-features-v1',
        'binding':canonical(feature_binding).decode()})
    targets=torch.tensor([r['target'] for r in rows],dtype=torch.float32,device=request['device'])
    variants,fit,optimizers=train_dense_selectors(features[:128].to(request['device']),targets,p,check=check,progress=progress)
    artifacts={}
    def record(path):artifacts[path.name]=dict(zip(['sha256','bytes'],digest_file(path)))
    record(output/'prefix-features.safetensors');frozen={}
    for name,selector in variants.items():
        save_dense(selector,output/(name+'.safetensors'),binding(selector,name));record(output/(name+'.safetensors'))
        frozen[name]=tensor_digest(selector.state_dict())
    torch.save(optimizers,output/'optimizer-state.pt');record(output/'optimizer-state.pt')
    predictions={};bindings={}
    for name,selector in variants.items():
        metadata=binding(selector,name);bindings[name]=metadata
        restored=DenseExpertSelector(features.shape[1],support_count=len(fit['support_indices'])).to(request['device'])
        load_dense(restored,output/(name+'.safetensors'),metadata);restored.eval();restored.requires_grad_(False)
        if tensor_digest(restored.state_dict())!=frozen[name]:raise ValueError('Reload changed a learned prefix selector')
        choices,probabilities=restored.choose(features[128:].to(request['device']),use_memory=name!='dolphin')
        predictions[name]=[{'index':i,'choice':c,'probabilities':prob} for i,c,prob in
            zip(p['prediction_indices'],choices.cpu().tolist(),probabilities.cpu().tolist())]
        if tensor_digest(restored.state_dict())!=frozen[name]:raise ValueError('Prediction mutated public prefix memory')
    commitment={'protocol_id':protocol_id(),'feature_space_id':feature_id(),'expert_ids':p['expert_ids'],
        'learned_artifacts':{k:v for k,v in artifacts.items() if k!='prefix-features.safetensors'},'predictions':predictions}
    verify_predictions(commitment);commitment_path=output/'prediction-commitment.json'
    atomic_json(commitment_path,commitment);record(commitment_path)
    with commitment_path.open('rb') as committed_file:os.fsync(committed_file.fileno())
    if os.name=='posix':
        fd=os.open(output,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    commitment_id=hashlib.sha256(canonical(commitment)).hexdigest();committed_at=time.time()
    if sum(v['bytes'] for v in artifacts.values())>p['disk_reservation_gb']*1024**3:raise ValueError('Prefix state exceeded its reserved storage')
    progress('new-routes-frozen-before-fresh-generation',128,128)
    # No fresh answer is generated or scored before the durable commitment.
    # Run the primary expert first for every query, then its counterfactual.
    outcomes={'accepted-parent':[],'research-specialist':[]};order=[]
    for offset,index in enumerate(p['prediction_indices']):
        choice=predictions[p['primary_selector']][offset]['choice'];order.append([choice,1-choice])
        for expert in order[-1]:
            check()
            with bank.activate(expert) as active:
                result=generate_prefix(active,tokenizer,prefixes[128+offset],device=request['device'],deadline=request['deadline'],maximum=p['maximum_new_tokens'])
            name=['accepted-parent','research-specialist'][expert]
            outcomes[name].append({'case':snapshot['rows'][index]['case'],**result})
        progress('fresh-outcomes-after-commitment',offset+1,128)
        atomic_json(output/'partial-outcomes.json',{'commitment_id':commitment_id,'completed':offset+1,'outcomes':outcomes})
    if (not all(torch.equal(v,parent[name]) for name,v in parameters.items())
            or any(tensor_digest(net.state_dict())!=frozen[name] for name,net in variants.items())
            or json.loads(commitment_path.read_text())!=commitment):raise ValueError('Fresh generation changed a frozen state or commitment')
    if digest_file(request['adapter_file'])[0]!=PARENT_MODEL or digest_file(request['specialist_file'])[0]!=SPECIALIST_ID:
        raise ValueError('Prefix component changed a frozen input file')
    report={'success':True,'protocol_id':protocol_id(),'feature_space_id':feature_id(),'commitment':commitment,
        'commitment_id':commitment_id,'committed_at':committed_at,'committed_before_fresh_generation':True,
        'feature_binding':feature_binding,'feature_extraction_seconds':feature_seconds,'base_execution':execution,
        'fit':fit,'bindings':bindings,'artifacts':artifacts,'outcomes':outcomes,'generation_expert_order':order,
        'original_evaluation_opened':False,'native_promotion_authorized':False,'language_experts_updated':False,
        'parent_parameters_restored':True,'new_native_cycles':0,'native_chat_enabled':False,
        'seconds':time.monotonic()-began,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated() if request['device']=='cuda' else 0}
    report['verification']=verify_report(report,snapshot)
    raw=canonical(report);(output/'diagnostic.json').write_bytes(raw);progress('complete',128,128)
    return {'success':True,'report_id':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}


def main():
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    raw=sys.stdin.buffer.read(16*1024+1)
    if len(raw)>16*1024:raise ValueError('Prefix component request exceeds its bound')
    with contextlib.redirect_stdout(sys.stderr):response=run(json.loads(raw))
    print(json.dumps(response,allow_nan=False),flush=True)


if __name__=='__main__':main()
