"""Offline training-loss diagnostic, using public inputs and no generation."""
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from .language_retention import PARENT_MODEL, PARENT_RELEASE, GRADIENTS, protocol_id

os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'


def validate_request(request):
    fields = {'action', 'protocol_id', 'base_directory', 'adapter_file', 'model_id', 'parent_release_id',
              'training_file', 'contributions', 'output_directory', 'device', 'cpu_threads', 'gpu_fraction', 'deadline'}
    if (not isinstance(request, dict) or set(request) != fields or request['action'] != 'public-retention-diagnostic-v2'
            or request['protocol_id'] != protocol_id() or request['model_id'] != PARENT_MODEL
            or request['parent_release_id'] != PARENT_RELEASE or request['device'] not in {'cpu', 'cuda'}):
        raise ValueError('Unknown bounded retention diagnostic or model')
    if (type(request['cpu_threads']) is not int or not 1 <= request['cpu_threads'] <= 16
            or type(request['gpu_fraction']) not in (int, float) or not 0 < request['gpu_fraction'] <= 1
            or type(request['deadline']) not in (int, float) or not math.isfinite(request['deadline'])
            or not time.time() < request['deadline'] <= time.time()+1800):
        raise ValueError('Invalid retention diagnostic resources')
    if any(not isinstance(request[k], str) or not Path(request[k]).is_absolute()
           for k in ['base_directory', 'adapter_file', 'training_file', 'output_directory']):
        raise ValueError('Retention diagnostic requires locally selected absolute paths')
    output = Path(request['output_directory'])
    if output.exists() or output.is_symlink() or output.resolve().is_relative_to(Path(request['base_directory']).resolve()):
        raise ValueError('Retention diagnostic cannot replace previous output or model files')
    from .language_candidate_job import GRADIENT_FIELDS
    from .language_policy import dataset_id, HALF_POLICY
    from .public_bases import base_hash
    rows = request['contributions']
    if not isinstance(rows, list) or len(rows) != 3:raise ValueError('Retention diagnosis requires its three fixed signed directions')
    for row, expected in zip(rows, GRADIENTS):
        if (not isinstance(row, dict) or set(row) != GRADIENT_FIELDS | {'quantization_relative_l2_error'}
                or any(type(row[k]) is not int for k in ['gradient_bytes','start','count','tokens'])
                or tuple(row.get(k) for k in ['gradient_id', 'gradient_bytes', 'start', 'count', 'tokens']) != expected
                or row['adapter_id'] != PARENT_MODEL or row['base_hash'] != base_hash('qwen3-1.7b')
                or row['dataset_id'] != dataset_id(4) or not isinstance(row['path'], str) or not Path(row['path']).is_absolute()
                or type(row['quantization_relative_l2_error']) not in (int, float)
                or not math.isfinite(row['quantization_relative_l2_error'])
                or not 0 <= row['quantization_relative_l2_error'] <= HALF_POLICY['maximum_relative_l2_error']):
            raise ValueError('Retention diagnosis cannot substitute data, directions or precision')


def training_sets(snapshot):
    from .public_conversation import public_corpus
    from .public_math import prompt
    old = public_corpus()['train']
    return {'old_prefix64': old[:64], 'old_full200': old,
            'math_train192': [{'prompt': prompt(r), 'answer': r['answer']} for r in snapshot['rows'][:192]]}


def training_statistics(model, tokenizer, rows, device, deadline, *, gradient=False, maximum=1024, answer_maximum=512):
    import torch
    from .language_adapter import adapter_parameters
    if time.time() >= deadline:raise TimeoutError('Retention training diagnostic deadline exceeded')
    if not rows:raise ValueError('Training diagnostic needs public training examples')
    encoded=[]
    for row in rows:
        text=tokenizer.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=False,
                                           add_generation_prompt=True,enable_thinking=False)
        prefix=tokenizer.encode(text,add_special_tokens=False)
        target=tokenizer.encode(row['answer'],add_special_tokens=False)+[tokenizer.eos_token_id]
        if len(prefix)+len(target)>maximum or len(target)>answer_maximum:raise ValueError('Training diagnostic cannot truncate examples')
        encoded.append((prefix,target))
    tokens=sum(len(target) for _prefix,target in encoded);total=0.;previous=model.training
    # Keep the forward path identical when only the backward calculation differs.
    # BF16 attention/loss paths may differ between training and no-grad inference.
    model.train();model.zero_grad(set_to_none=True)
    try:
        with torch.enable_grad():
            for prefix,target in encoded:
                if time.time()>=deadline:raise TimeoutError('Retention training diagnostic deadline exceeded')
                inputs=torch.tensor([prefix+target],device=device)
                labels=torch.tensor([[-100]*len(prefix)+target],device=device)
                loss=model(input_ids=inputs,labels=labels,use_cache=False).loss
                value=float(loss.detach())
                if not math.isfinite(value) or value<0:raise ValueError('Invalid public training loss')
                total+=value*len(target)
                if gradient:(loss*len(target)/tokens).backward()
        result={'nll':total/tokens,'tokens':tokens,'examples':len(rows)}
        gradients={name:p.grad.detach().float().cpu().clone() for name,p in adapter_parameters(model).items()} if gradient else None
        return result,gradients
    finally:
        model.zero_grad(set_to_none=True);model.train(previous)


def direction_statistics(direction, reference, *, rate=.001):
    import torch
    if not direction or set(direction)!=set(reference):raise ValueError('Direction and reference parameters differ')
    if any(direction[n].shape!=reference[n].shape or not torch.isfinite(direction[n]).all()
           or not torch.isfinite(reference[n]).all() for n in direction):raise ValueError('Invalid diagnostic direction tensors')
    norm2=sum(float(t.double().square().sum()) for t in direction.values())
    ref2=sum(float(t.double().square().sum()) for t in reference.values())
    dot=sum(float((direction[n].double()*reference[n].double()).sum()) for n in direction)
    if not all(math.isfinite(x) for x in [norm2,ref2,dot]):raise ValueError('Nonfinite diagnostic direction')
    return {'dot':dot,'direction_norm':math.sqrt(norm2),'reference_norm':math.sqrt(ref2),
            'cosine':dot/math.sqrt(norm2*ref2) if norm2 and ref2 else None,
            'predicted_first_order_loss_change':-rate*dot}


@contextlib.contextmanager
def temporary_step(parameters, direction, rate=.001):
    import torch
    if set(parameters)!=set(direction):raise ValueError('Temporary diagnostic step has different parameters')
    originals={name:p.detach().clone() for name,p in parameters.items()}
    try:
        with torch.no_grad():
            for name,p in parameters.items():
                p.add_(direction[name].to(p.device),alpha=-rate)
                if not torch.isfinite(p).all():raise ValueError('Nonfinite temporary diagnostic step')
        yield
    finally:
        with torch.no_grad():
            for name,p in parameters.items():p.copy_(originals[name])
        if not all(torch.equal(p,originals[name]) for name,p in parameters.items()):raise ValueError('Diagnostic did not restore parent parameters')


def run(request):
    validate_request(request);started=time.monotonic()
    from .public_bases import BASES,base_hash
    from .public_math import load_snapshot,canonical
    from .weights_io import digest_file
    for filename,identifier in BASES['qwen3-1.7b'].items():
        if digest_file(Path(request['base_directory'])/filename)[0]!=identifier:raise ValueError('Retention diagnostic public base mismatch')
    if digest_file(request['adapter_file'])[0]!=PARENT_MODEL:raise ValueError('Retention diagnostic parent mismatch')
    for row in request['contributions']:
        if digest_file(row['path'])!=(row['gradient_id'],row['gradient_bytes']):raise ValueError('Retention diagnostic gradient hash mismatch')
    sets=training_sets(load_snapshot(request['training_file'],'train'))
    output=Path(request['output_directory']);output.mkdir(parents=True,exist_ok=False)
    def progress(phase):
        (output/'progress.json').write_bytes(canonical({'phase':phase,'timestamp':time.time()}))
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from .language_adapter import install_adapter,load_adapter,adapter_parameters
    from .language_gradient import combine_gradients,load_gradient,save_gradient
    from .language_math_gradient import backward
    torch.set_num_threads(request['cpu_threads']);torch.manual_seed(20261014)
    if request['device']=='cuda':
        torch.cuda.set_per_process_memory_fraction(request['gpu_fraction']);torch.cuda.reset_peak_memory_stats()
    tokenizer=AutoTokenizer.from_pretrained(request['base_directory'],local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(request['base_directory'],local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16 if request['device']=='cuda' else torch.float32,device_map={'':request['device']})
    parameters=install_adapter(model);load_adapter(model,request['adapter_file'],base_hash('qwen3-1.7b'))
    model.config.use_cache=False;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.train()
    baseline={};references={}
    for name,rows in sets.items():
        progress('baseline-'+name)
        baseline[name],reference=training_statistics(model,tokenizer,rows,request['device'],request['deadline'],gradient=name!='math_train192')
        if reference is not None:references[name]=reference
    directions={}
    for row in request['contributions']:
        fields={k:row[k] for k in ['base_hash','adapter_id','dataset_id','start','count','tokens','quantization_relative_l2_error']}
        directions['saved_start'+str(row['start'])]=load_gradient(model,row['path'],policy_version=4,**fields)
    combined=combine_gradients(model,request['contributions'],policy_version=4);directions['saved_token_weighted']=combined
    progress('reconstruct-first-desktop-shard');model.train();model.zero_grad(set_to_none=True)
    loss,replay_loss,tokens,replay_tokens=backward(model,tokenizer,sets['math_train192'][:64],request['device'],request['deadline'])
    reconstructed={name:p.grad.detach().float().cpu().clone() for name,p in parameters.items()}
    directions['reconstructed_float32_start0']=reconstructed
    row=request['contributions'][0];fields={k:row[k] for k in ['base_hash','adapter_id','dataset_id','start','count','tokens']}
    if tokens!=row['tokens']:raise ValueError('Reconstructed training tokens differ from the signed source')
    path=output/'reconstructed-root-gradient.safetensors';precision=save_gradient(model,path,**fields,policy_version=4)
    exact=digest_file(path)==(row['gradient_id'],row['gradient_bytes'])
    directions['reconstructed_decoded_start0']=load_gradient(model,path,**fields,policy_version=4,**precision)
    original=directions['saved_start0'];current=directions['reconstructed_decoded_start0']
    norm=math.sqrt(sum(float(t.double().square().sum()) for t in original.values()))
    difference=math.sqrt(sum(float((original[n].double()-current[n].double()).square().sum()) for n in original))
    concordance={**direction_statistics(original,current),'relative_l2_difference':difference/norm if norm else None}
    geometry={name:{key:direction_statistics(value,reference) for key,reference in references.items()}
              for name,value in directions.items()}
    progress('no-update-control')
    control={name:training_statistics(model,tokenizer,rows,request['device'],request['deadline'])[0] for name,rows in sets.items()}
    control_changes={name:control[name]['nll']-baseline[name]['nll'] for name in sets}
    control_passed=all(abs(value)<=max(1e-7,abs(baseline[name]['nll'])*1e-6) for name,value in control_changes.items())
    progress('temporary-signed-update')
    with temporary_step(parameters,combined):
        after={name:training_statistics(model,tokenizer,rows,request['device'],request['deadline'])[0] for name,rows in sets.items()}
    deltas={name:{'actual_loss_change':after[name]['nll']-baseline[name]['nll']} for name in sets}
    for name in references:
        predicted=geometry['saved_token_weighted'][name]['predicted_first_order_loss_change']
        deltas[name].update(predicted_first_order_loss_change=predicted,
            finite_step_residual=deltas[name]['actual_loss_change']-predicted)
    if time.time()>=request['deadline']:raise TimeoutError('Retention diagnostic deadline exceeded')
    result={'success':True,'protocol_id':protocol_id(),'model_id':PARENT_MODEL,'parent_release_id':PARENT_RELEASE,
        'baseline':baseline,'after_signed_update':after,'loss_changes':deltas,'direction_geometry':geometry,
        'loss_execution_mode':'train-with-autograd-for-every-loss-pass; backward-only-when-requested',
        'no_update_control':control,'no_update_loss_changes':control_changes,'no_update_control_passed':control_passed,
        'prefix_vs_full_reference_alignment':direction_statistics(references['old_prefix64'],references['old_full200']),
        'reconstruction':{'original_half_artifact_reproduced_exactly':exact,'artifact_id':digest_file(path)[0],
            'math_training_nll':loss,'replay_training_nll':replay_loss,'tokens':tokens,'replay_tokens':replay_tokens,**precision,
            'decoded_direction_concordance_with_saved':concordance,
            'causal_scope':'Source-vs-codec comparison of the original artifact only if exact reproduction succeeded'},
        'held_out_data_used_in_compute':False,'generation_performed':False,'parent_parameters_restored':True,
        'updated_model_files_written':False,'model_published':False,'new_promotion_authorized':False,
        'seconds':time.monotonic()-started,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated() if request['device']=='cuda' else 0}
    raw=canonical(result);path=output/'diagnostic.json';path.write_bytes(raw);progress('complete')
    return {'success':True,'protocol_id':protocol_id(),'report_id':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}


def main():
    raw=sys.stdin.buffer.read(16*1024+1)
    if len(raw)>16*1024:raise ValueError('Retention request exceeds its bound')
    print(json.dumps(run(json.loads(raw)),allow_nan=False),flush=True)


if __name__=='__main__':main()
