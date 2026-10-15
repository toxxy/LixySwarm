"""Prospective specialist TRAIN, learned routing and original quality gates."""
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from .language_specialist_protocol import definition,protocol_id,necessary_loss_gate
from .language_retention import PARENT_MODEL,PARENT_RELEASE
from .language_selector_job import encode_rows,validate_request as validate_selector_request

os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'


def validate_request(request):
    validate_selector_request(request,protocol=definition(),identifier=protocol_id())


def expert_losses(bank,encoded,device,check,progress):
    import torch
    values=[]
    with torch.inference_mode():
        for index,(prefix,answer) in enumerate(encoded):
            check();inputs=torch.tensor([prefix+answer],device=device)
            labels=torch.tensor([[-100]*len(prefix)+answer],device=device);row=[]
            for choice in [0,1]:
                with bank.activate(choice) as model:value=float(model(input_ids=inputs,labels=labels,use_cache=False).loss)
                if not math.isfinite(value) or value<0:raise ValueError('Invalid frozen adapter expert loss')
                row.append(value)
            values.append(row);progress(index+1,len(encoded))
    return torch.tensor(values,dtype=torch.float64)


def run(request):
    validate_request(request);started=time.monotonic();protocol=definition()
    from .public_bases import BASES,base_hash
    from .public_math import load_snapshot,canonical,prompt,ARTIFACTS,compact_evaluation,verify_math_transition,MAX_TRACE_BYTES
    from .public_conversation import public_corpus
    from .weights_io import digest_file
    for filename,expected in BASES['qwen3-1.7b'].items():
        if digest_file(Path(request['base_directory'])/filename)[0]!=expected:raise ValueError('Specialist public base mismatch')
    parent_digest=digest_file(request['adapter_file'])
    if parent_digest[0]!=PARENT_MODEL:raise ValueError('Specialist accepted generalist mismatch')
    corpus=public_corpus();snapshot=load_snapshot(request['training_file'],'train')
    math_rows=[{'prompt':prompt(r),'answer':r['answer']} for r in snapshot['rows']]
    boundary=protocol['specialist_training_start'];end=boundary+protocol['specialist_training_examples']
    if len(corpus['train'])!=protocol['historical_training_examples'] or boundary!=protocol['math_training_examples'] or end!=len(math_rows):
        raise ValueError('Specialist and calibration must partition the complete public math TRAIN snapshot')
    training=math_rows[boundary:end];calibration=corpus['train']+math_rows[:boundary]
    if {r['prompt'] for r in training}&{r['prompt'] for r in calibration}:raise ValueError('Specialist math TRAIN overlaps route calibration')
    output=Path(request['output_directory']);output.mkdir(parents=True,exist_ok=False)
    def progress(phase,done=0,total=0):
        (output/'progress.json').write_bytes(canonical({'phase':phase,'completed':done,'total':total,'timestamp':time.time()}))
    def check():
        if time.time()>=request['deadline']:raise TimeoutError('Specialist research deadline exceeded')
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from .language_adapter import install_adapter,load_adapter,save_adapter
    from .language_streamed_float import install_streamed_float32
    from .language_specialist import fit_specialist,FrozenAdapterBank,RoutedAdapterExperts
    from .language_selector import train_selectors,save_selector,load_selector,LanguageExpertSelector,tensor_digest
    torch.set_num_threads(request['cpu_threads']);torch.manual_seed(protocol['seed'])
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if request['device']=='cuda':torch.cuda.set_per_process_memory_fraction(request['gpu_fraction']);torch.cuda.reset_peak_memory_stats()
    progress('load-pinned-generalist')
    tokenizer=AutoTokenizer.from_pretrained(request['base_directory'],local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(request['base_directory'],local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16,device_map={'':request['device']})
    parameters=install_adapter(model);load_adapter(model,request['adapter_file'],base_hash('qwen3-1.7b'))
    base_execution=install_streamed_float32(model);model.config.use_cache=False
    parent={name:p.detach().clone() for name,p in parameters.items()};artifacts={};generation_performed=False;quality={}
    def artifact(path):
        artifacts[path.name]=dict(zip(['sha256','bytes'],digest_file(path)))
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        trained,optimizer=fit_specialist(model,parameters,encode_rows(tokenizer,training),encode_rows(tokenizer,corpus['train']),
            protocol,device=request['device'],check=check,progress=progress)
        specialist={name:p.detach().clone() for name,p in parameters.items()}
        save_adapter(model,output/'specialist.safetensors',base_hash('qwen3-1.7b'));artifact(output/'specialist.safetensors')
        torch.save(optimizer,output/'specialist-optimizer.pt');artifact(output/'specialist-optimizer.pt');del optimizer
        specialist_id=artifacts['specialist.safetensors']['sha256']
        with torch.no_grad():
            for name,p in parameters.items():p.copy_(parent[name]);p.requires_grad_(False)
        model.gradient_checkpointing_disable();model.eval()
        bank=FrozenAdapterBank(model,parameters,[parent,specialist])
        encoded=encode_rows(tokenizer,calibration)
        teacher=expert_losses(bank,encoded,request['device'],check,lambda d,t:progress('calibration-expert-nll',d,t))
        variants,fit,optimizers=train_selectors([p for p,a in encoded],teacher,model.config.vocab_size,protocol,
            device=request['device'],check=check,progress=progress)
        binding=dict(protocol_id=protocol_id(),base_hash=base_hash('qwen3-1.7b'),tokenizer_id=protocol['tokenizer_id'],
            dolphin_state_id=fit['dolphin_state_id'],feature_space=fit['memory_feature_space'],memory_scope='public-training-only',
            expert_ids=[PARENT_MODEL,specialist_id])
        frozen={}
        for name,selector in variants.items():
            check();metadata={**binding,'variant':name};path=output/(name+'.safetensors')
            save_selector(selector,path,metadata);artifact(path)
            reloaded=LanguageExpertSelector(model.config.vocab_size,support_count=len(fit['support_indices'])).to(request['device'])
            load_selector(reloaded,path,metadata);reloaded.eval()
            if tensor_digest(reloaded.state_dict())!=tensor_digest(selector.state_dict()):raise ValueError('Specialist selector reload changed its learned state')
            del reloaded;frozen[name]=tensor_digest(selector.state_dict())
        torch.save(optimizers,output/'optimizer-state.pt');artifact(output/'optimizer-state.pt');del optimizers
        (output/'training.json').write_bytes(canonical({'specialist_training':trained,'route_fit':fit,'teacher_nll':teacher.tolist(),
            'binding':binding,'artifacts':artifacts}))
        progress('all-states-frozen-before-heldout')
        evaluation=load_snapshot(request['evaluation_file'],'evaluation')
        heldout=corpus['evaluation']+[{'prompt':prompt(r),'answer':r['answer']} for r in evaluation['rows']]
        if {r['prompt'] for r in training+calibration}&{r['prompt'] for r in heldout}:raise ValueError('Specialist training data intersects held-out prompts')
        encoded_eval=encode_rows(tokenizer,heldout)
        losses=expert_losses(bank,encoded_eval,request['device'],check,lambda d,t:progress('heldout-adapter-nll',d,t))
        tokens=torch.tensor([len(a) for p,a in encoded_eval],dtype=torch.float64);reports={}
        def summary(values,choices):
            return {key:{'nll':float((values[sl]*tokens[sl]).sum()/tokens[sl].sum()),'examples':len(values[sl]),
                'tokens':int(tokens[sl].sum()),'specialist_choices':int(choices[sl].sum())}
                for key,sl in [('historical',slice(0,len(corpus['evaluation']))),('math',slice(len(corpus['evaluation']),None))]}
        for expert,name in enumerate(['accepted-parent','trained-math-specialist']):
            reports[name]=summary(losses[:,expert],torch.full((len(heldout),),expert))
        reports['oracle-only-diagnostic']=summary(losses.min(dim=-1).values,losses.argmin(dim=-1))
        for name,selector in variants.items():
            check();routes=[selector.choose([p],use_memory=name!='dolphin') for p,a in encoded_eval]
            choices=torch.cat([r[0] for r in routes]).cpu();probabilities=torch.cat([r[1] for r in routes]).cpu()
            expected=losses[torch.arange(len(heldout)),choices];actual=[]
            routed=RoutedAdapterExperts(bank,selector,use_memory=name!='dolphin').eval()
            with torch.inference_mode():
                for i,(prefix,answer) in enumerate(encoded_eval):
                    check();inputs=torch.tensor([prefix+answer],device=request['device'])
                    labels=torch.tensor([[-100]*len(prefix)+answer],device=request['device'])
                    actual.append(float(routed(input_ids=inputs,labels=labels,use_cache=False).loss))
                    progress('heldout-routed-'+name,i+1,len(encoded_eval))
            if not torch.equal(torch.tensor(actual,dtype=torch.float64),expected):raise ValueError('Routed specialist loss differs from the selected frozen adapter')
            reports[name]={**summary(expected,choices),'choices':choices.tolist(),'probabilities':probabilities.tolist(),
                'single_selected_expert_executed_per_request':True,'prefix_only_routing_verified':True}
        screen=necessary_loss_gate(reports['accepted-parent'],reports[protocol['primary_selector']])
        if screen['necessary_loss_gates_passed']:
            from .language_candidate_job import evaluate
            from .public_math_job import evaluate_math
            def assess(model,label):
                check();legacy=evaluate(model,tokenizer,corpus,request['device'],request['deadline'],
                    progress=lambda phase,d,t:progress(label+'-'+phase,d,t))
                math_result=evaluate_math(model,tokenizer,evaluation,request['device'],request['deadline'],
                    progress=lambda d,t:progress(label+'-math',d,t))
                return legacy,math_result
            before,math_before=assess(model,'parent-fp32')
            routed=RoutedAdapterExperts(bank,variants[protocol['primary_selector']]).eval()
            after,math_after=assess(routed,'combined-fp32')
            for values,expected in [(before,reports['accepted-parent']),(after,reports[protocol['primary_selector']])]:
                if not math.isclose(values['nll'],expected['historical']['nll'],rel_tol=1e-9,abs_tol=1e-12):raise ValueError('Quality and screening historical losses disagree')
            if not math.isclose(math_before['nll'],reports['accepted-parent']['math']['nll'],rel_tol=1e-9,abs_tol=1e-12) or not math.isclose(math_after['nll'],reports[protocol['primary_selector']]['math']['nll'],rel_tol=1e-9,abs_tol=1e-12):
                raise ValueError('Quality and screening math losses disagree')
            def compare(label,legacy,math_result):
                trace={'version':1,'evaluation_artifact_id':ARTIFACTS['evaluation'][0],'before':math_result,'after':math_after}
                raw=canonical(trace)
                if len(raw)>MAX_TRACE_BYTES:raise ValueError('Specialist full answer trace exceeds its bound')
                path=output/('math-traces-'+label+'.json');path.write_bytes(raw);artifact(path)
                value={'baseline_nll':legacy['nll'],'candidate_nll':after['nll'],'quality_before':legacy['quality'],
                    'quality_after':after['quality'],'instruction_before':legacy['instructions'],'instruction_after':after['instructions'],
                    'math_before':compact_evaluation(math_result),'math_after':compact_evaluation(math_after),
                    'math_trace_id':hashlib.sha256(raw).hexdigest(),'math_trace_bytes':len(raw)}
                value['verdict']=verify_math_transition(value,evaluation,trace);return value
            quality['parent_fp32']=compare('fp32',before,math_before)
            progress('load-parent-published-bf16-profile')
            bf16=AutoModelForCausalLM.from_pretrained(request['base_directory'],local_files_only=True,trust_remote_code=False,
                dtype=torch.bfloat16,device_map={'':request['device']}).eval()
            install_adapter(bf16);load_adapter(bf16,request['adapter_file'],base_hash('qwen3-1.7b'))
            for parameter in bf16.parameters():parameter.requires_grad_(False)
            old,math_old=assess(bf16,'parent-bf16');quality['parent_bf16']=compare('bf16',old,math_old);del bf16
            generation_performed=True
        for name,selector in variants.items():
            if tensor_digest(selector.state_dict())!=frozen[name]:raise ValueError('Held-out scoring changed a learned selector')
        if not all(torch.equal(p,parent[name]) for name,p in parameters.items()):raise ValueError('Frozen generalist state changed')
    finally:
        with torch.no_grad():
            for name,p in parameters.items():p.copy_(parent[name]);p.requires_grad_(True)
    check()
    if digest_file(request['adapter_file'])!=parent_digest:raise ValueError('Accepted generalist file changed')
    if sum(r['bytes'] for r in artifacts.values())>protocol['disk_reservation_gb']*1024**3:raise ValueError('Specialist state exceeds the declared disk reservation')
    eligible=generation_performed and all(q['verdict']['eligible_for_research_promotion'] for q in quality.values())
    result={'success':True,'protocol_id':protocol_id(),'model_id':PARENT_MODEL,'parent_release_id':PARENT_RELEASE,'base_execution':base_execution,
        'specialist_training':trained,'fit':fit,'binding':binding,'selector_artifacts':artifacts,'evaluations':reports,'loss_screen':screen,
        'heldout_expert_nll':losses.tolist(),'heldout_answer_tokens':tokens.tolist(),'quality':quality,
        'held_out_data_used_in_compute':True,'held_out_data_used_for_training_or_selection':False,'generation_performed':generation_performed,
        'parent_parameters_restored':True,'updated_model_files_written':True,'language_expert_files_written':True,
        'generalist_files_written':False,'selector_state_files_written':True,'full_generation_gates_evaluated':generation_performed,
        'eligible_for_research_promotion':bool(eligible),'eligible_for_native_promotion':False,'new_model_publication_authorized':False,
        'native_learning_cycles_added':0,'new_remote_contributions':0,'seconds':time.monotonic()-started,
        'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated() if request['device']=='cuda' else 0}
    raw=canonical(result)
    if len(raw)>1024**2:raise ValueError('Specialist report exceeds its bound')
    (output/'diagnostic.json').write_bytes(raw);progress('complete')
    return {'success':True,'protocol_id':protocol_id(),'report_id':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}


def main():
    raw=sys.stdin.buffer.read(16*1024+1)
    if len(raw)>16*1024:raise ValueError('Specialist request exceeds its bound')
    with contextlib.redirect_stdout(sys.stderr):response=run(json.loads(raw))
    print(json.dumps(response,allow_nan=False),flush=True)


if __name__=='__main__':main()
