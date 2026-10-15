"""Offline four-step public TRAIN block and paired held-out evaluation."""
import hashlib,json,os,sys,time
from pathlib import Path
from .language_block_protocol import definition,protocol_id
from .language_retention import PARENT_MODEL,PARENT_RELEASE

os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'


def validate_request(request):
    from .language_retention import protocol_id as prior_id
    from .language_retention_job import validate_request as prior_validate
    if (not isinstance(request,dict) or request.get('action')!='public-block-quality-v1'
            or request.get('protocol_id')!=protocol_id() or request.get('compute_dtype')!='float32'
            or request.get('streamed_base') is not True or not isinstance(request.get('evaluation_file'),str)
            or not Path(request['evaluation_file']).is_absolute()):
        raise ValueError('Unknown bounded quality-block protocol')
    prior_validate({**{k:v for k,v in request.items() if k not in {'streamed_base','evaluation_file'}},
        'action':'public-retention-diagnostic-v3','protocol_id':prior_id('float32')},maximum_seconds=definition()['maximum_seconds'])


def run(request):
    validate_request(request);started=time.monotonic();protocol=definition()
    from .public_bases import BASES,base_hash
    from .public_math import load_snapshot,canonical,compact_evaluation,verify_math_transition,ARTIFACTS,MAX_TRACE_BYTES
    from .public_conversation import public_corpus
    from .weights_io import digest_file
    from .language_retention_job import training_sets,training_statistics
    for filename,expected in BASES['qwen3-1.7b'].items():
        if digest_file(Path(request['base_directory'])/filename)[0]!=expected:raise ValueError('Quality-block base mismatch')
    parent_digest=digest_file(request['adapter_file'])
    if parent_digest[0]!=PARENT_MODEL:raise ValueError('Quality-block parent mismatch')
    for row in request['contributions']:
        if digest_file(row['path'])!=(row['gradient_id'],row['gradient_bytes']):raise ValueError('Quality-block gradient mismatch')
    corpus=public_corpus();old_rows=corpus['train']
    math_rows=training_sets(load_snapshot(request['training_file'],'train'))['math_train192']
    if len(old_rows)!=200 or len(math_rows)!=192:raise ValueError('Quality block training groups changed')
    evaluation=load_snapshot(request['evaluation_file'],'evaluation')
    output=Path(request['output_directory']);output.mkdir(parents=True,exist_ok=False)
    def progress(phase,done=0,total=0):
        (output/'progress.json').write_bytes(canonical({'phase':phase,'completed':done,'total':total,'timestamp':time.time()}))
    def check():
        if time.time()>=request['deadline']:raise TimeoutError('Quality block deadline exceeded')
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from .language_adapter import install_adapter,load_adapter
    from .language_streamed_float import install_streamed_float32
    from .language_gradient import combine_gradients
    from .language_group_direction import project_groups
    from .language_finite_step import public_training_measurement
    from .language_block import protected_block
    from .language_candidate_job import evaluate
    from .public_math_job import evaluate_math
    torch.set_num_threads(request['cpu_threads']);torch.manual_seed(20261014)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if request['device']=='cuda':
        torch.cuda.set_per_process_memory_fraction(request['gpu_fraction']);torch.cuda.reset_peak_memory_stats()
    progress('load-pinned-model')
    tokenizer=AutoTokenizer.from_pretrained(request['base_directory'],local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(request['base_directory'],local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16,device_map={'':request['device']})
    parameters=install_adapter(model);load_adapter(model,request['adapter_file'],base_hash('qwen3-1.7b'))
    base_execution=install_streamed_float32(model)
    model.config.use_cache=False;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    source=combine_gradients(model,request['contributions'],policy_version=4)
    originals={name:p.detach().clone() for name,p in parameters.items()};passes=0
    groups={'old_prefix64':old_rows[:64],'old_full200':old_rows}
    groups.update({'old_kind_'+kind:[row for row in old_rows if row['kind']==kind] for kind in sorted({r['kind'] for r in old_rows})})
    if set(groups)!=set(protocol['protected']):raise ValueError('Quality block protected groups changed')
    def measure():
        nonlocal passes
        passes+=1;progress('train-loss-pass-'+str(passes))
        return public_training_measurement(model,tokenizer,old_rows,math_rows,request['device'],request['deadline'])
    def prepare(direction,index):
        references={};statistics={}
        for name,rows in groups.items():
            progress('step-'+str(index)+'-reference-'+name);check()
            statistics[name],references[name]=training_statistics(model,tokenizer,rows,request['device'],request['deadline'],gradient=True)
        projected,geometry=project_groups(direction,references,maximum_sweeps=protocol['maximum_projection_sweeps'],
            relative_tolerance=protocol['projection_relative_tolerance'],relative_margin=protocol['projection_relative_margin'])
        geometry['within_direction_norm_bound']=geometry['projected_norm']<=protocol['maximum_projected_direction_norm']
        geometry['converged']=geometry['converged'] and geometry['within_direction_norm_bound']
        geometry['reference_training_statistics']=statistics
        return projected,geometry
    def assess(side):
        progress(side+'-historical')
        legacy=evaluate(model,tokenizer,corpus,request['device'],request['deadline'],
            progress=lambda phase,done,total:progress(side+'-'+phase,done,total))
        math=evaluate_math(model,tokenizer,evaluation,request['device'],request['deadline'],
            progress=lambda done,total:progress(side+'-math',done,total))
        return legacy,math
    try:
        before,math_before=assess('before')
        (output/'quality-before.json').write_bytes(canonical({'legacy':before,'math':math_before}))
        if not all(torch.equal(p,originals[n]) for n,p in parameters.items()):raise ValueError('Baseline evaluation changed parent parameters')
        block=protected_block(parameters,source,measure,prepare,maximum_steps=protocol['maximum_steps'],check=check,
            guard_options={k:protocol[k] for k in ['protected','target','rates','absolute_tolerance','relative_tolerance','minimum_absolute_gain','minimum_relative_gain']})
        selected={name:p.detach().clone() for name,p in parameters.items()}
        if block['accepted_steps']:after,math_after=assess('after')
        else:after,math_after=before,math_before
        if not all(torch.equal(p,selected[n]) for n,p in parameters.items()):raise ValueError('Quality assessment changed the TRAIN-selected state')
    finally:
        with torch.no_grad():
            for name,p in parameters.items():p.copy_(originals[name])
        if not all(torch.equal(p,originals[n]) for n,p in parameters.items()):raise ValueError('Quality block did not restore parent')
    check()
    if digest_file(request['adapter_file'])!=parent_digest:raise ValueError('Quality-block parent file changed')
    trace={'version':1,'evaluation_artifact_id':ARTIFACTS['evaluation'][0],'before':math_before,'after':math_after}
    raw_trace=canonical(trace)
    if len(raw_trace)>MAX_TRACE_BYTES:raise ValueError('Quality block trace exceeds its bound')
    (output/'math-traces.json').write_bytes(raw_trace)
    quality={'baseline_nll':before['nll'],'candidate_nll':after['nll'],'quality_before':before['quality'],
        'quality_after':after['quality'],'instruction_before':before['instructions'],'instruction_after':after['instructions'],
        'math_before':compact_evaluation(math_before),'math_after':compact_evaluation(math_after),
        'math_trace_id':hashlib.sha256(raw_trace).hexdigest(),'math_trace_bytes':len(raw_trace)}
    quality['math_verdict']=verify_math_transition(quality,evaluation,trace)
    result={'success':True,'protocol_id':protocol_id(),'model_id':PARENT_MODEL,'parent_release_id':PARENT_RELEASE,
        'base_execution':base_execution,'block':block,'quality':quality,'training_loss_passes':passes,
        'held_out_data_used_in_compute':True,'held_out_data_used_for_training_or_selection':False,'generation_performed':True,
        'parent_parameters_restored':True,'updated_model_files_written':False,'model_published':False,
        'new_promotion_authorized':False,'native_learning_cycles_added':0,'new_remote_contributions':0,
        'seconds':time.monotonic()-started,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated() if request['device']=='cuda' else 0}
    raw=canonical(result)
    if len(raw)>1024**2:raise ValueError('Quality block report exceeds its bound')
    (output/'diagnostic.json').write_bytes(raw);progress('complete')
    return {'success':True,'protocol_id':protocol_id(),'report_id':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}


def main():
    raw=sys.stdin.buffer.read(16*1024+1)
    if len(raw)>16*1024:raise ValueError('Quality-block request exceeds its bound')
    print(json.dumps(run(json.loads(raw)),allow_nan=False),flush=True)


if __name__=='__main__':main()
