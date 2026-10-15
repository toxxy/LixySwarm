"""Offline, prospective finite-step check of one saved two-PC direction."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from .language_finite_protocol import definition,protocol_id
from .language_retention import PARENT_MODEL,PARENT_RELEASE

os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'


def validate_request(request):
    from .language_retention import protocol_id as prior_id
    from .language_retention_job import validate_request as prior_validate
    if (not isinstance(request,dict) or request.get('action')!='public-finite-step-v1'
            or request.get('protocol_id')!=protocol_id(request.get('compute_dtype'))):
        raise ValueError('Unknown prospective finite-step protocol')
    # Same pinned public inputs and resource/path constraints, a distinct action.
    prior_validate({**request,'action':'public-retention-diagnostic-v3',
                    'protocol_id':prior_id(request['compute_dtype'])})


def run(request):
    validate_request(request);started=time.monotonic();protocol=definition(request['compute_dtype'])
    from .public_bases import BASES,base_hash
    from .public_math import load_snapshot,canonical
    from .public_conversation import public_corpus
    from .weights_io import digest_file
    from .language_retention_job import training_sets
    for filename,identifier in BASES['qwen3-1.7b'].items():
        if digest_file(Path(request['base_directory'])/filename)[0]!=identifier:raise ValueError('Finite-step public base mismatch')
    parent_digest=digest_file(request['adapter_file'])
    if parent_digest[0]!=PARENT_MODEL:raise ValueError('Finite-step parent mismatch')
    for row in request['contributions']:
        if digest_file(row['path'])!=(row['gradient_id'],row['gradient_bytes']):raise ValueError('Finite-step gradient hash mismatch')
    old_rows=public_corpus()['train']
    math_rows=training_sets(load_snapshot(request['training_file'],'train'))['math_train192']
    if len(old_rows)!=200 or len(math_rows)!=192:raise ValueError('Finite-step public training groups changed')
    output=Path(request['output_directory']);output.mkdir(parents=True,exist_ok=False)
    def progress(phase):
        (output/'progress.json').write_bytes(canonical({'phase':phase,'timestamp':time.time()}))
    def check():
        if time.time()>=request['deadline']:raise TimeoutError('Finite-step research deadline exceeded')
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from .language_adapter import install_adapter,load_adapter
    from .language_gradient import combine_gradients
    from .language_finite_step import safeguarded_step,public_training_measurement
    torch.set_num_threads(request['cpu_threads']);torch.manual_seed(20261014)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if request['device']=='cuda':
        torch.cuda.set_per_process_memory_fraction(request['gpu_fraction']);torch.cuda.reset_peak_memory_stats()
    progress('load-pinned-model')
    tokenizer=AutoTokenizer.from_pretrained(request['base_directory'],local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(request['base_directory'],local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16 if request['compute_dtype']=='bfloat16' else torch.float32,device_map={'':request['device']})
    parameters=install_adapter(model);load_adapter(model,request['adapter_file'],base_hash('qwen3-1.7b'))
    model.config.use_cache=False;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.train()
    direction=combine_gradients(model,request['contributions'],policy_version=4)
    originals={name:p.detach().clone() for name,p in parameters.items()};passes=0
    def measure():
        nonlocal passes
        passes+=1;progress('training-loss-pass-'+str(passes))
        return public_training_measurement(model,tokenizer,old_rows,math_rows,request['device'],request['deadline'])
    try:
        guard=safeguarded_step(parameters,direction,measure,check=check,**{key:protocol[key] for key in
            ['protected','target','rates','absolute_tolerance','relative_tolerance','minimum_absolute_gain','minimum_relative_gain']})
    finally:
        with torch.no_grad():
            for name,p in parameters.items():p.copy_(originals[name])
        if not all(torch.equal(p,originals[n]) for n,p in parameters.items()):raise ValueError('Experimental finite-step rollback failed')
    check()
    if digest_file(request['adapter_file'])!=parent_digest:raise ValueError('Finite-step parent file changed')
    result={'success':True,'protocol_id':protocol_id(request['compute_dtype']),'model_id':PARENT_MODEL,'parent_release_id':PARENT_RELEASE,
        'model_compute_dtype':request['compute_dtype'],'float32_matmul_precision':'highest','cuda_tf32':False,
        'guard':guard,'training_loss_passes':passes,'no_update_control_passed':True,
        'loss_execution_mode':'train-with-autograd-for-every-loss-pass; no backward or generation',
        'held_out_data_used_in_compute':False,'generation_performed':False,'parent_parameters_restored':True,
        'updated_model_files_written':False,'model_published':False,'new_promotion_authorized':False,
        'seconds':time.monotonic()-started,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated() if request['device']=='cuda' else 0}
    raw=canonical(result);(output/'diagnostic.json').write_bytes(raw);progress('complete')
    return {'success':True,'protocol_id':protocol_id(request['compute_dtype']),'report_id':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}


def main():
    raw=sys.stdin.buffer.read(16*1024+1)
    if len(raw)>16*1024:raise ValueError('Finite-step request exceeds its bound')
    print(json.dumps(run(json.loads(raw)),allow_nan=False),flush=True)


if __name__=='__main__':main()
