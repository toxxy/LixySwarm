"""Offline GPU child collecting exact answers from two frozen public adapters."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from .language_outcomes import definition, protocol_id, partition, verify_outcomes, score_training_answers
from .language_retention import PARENT_MODEL,PARENT_RELEASE
from .language_specialist_answers import SPECIALIST_ID
from .public_math import canonical


def validate_request(request):
    from .language_selector_job import validate_request as validate_common
    extras = {'specialist_file','shard'}
    if not isinstance(request,dict) or not extras <= set(request): raise ValueError('Outcome calibration needs its frozen specialist and shard')
    # The common schema normally binds EVAL as well. Here it is forbidden.
    if 'evaluation_file' in request: raise ValueError('Outcome collection cannot open an evaluation snapshot')
    common = {k:v for k,v in request.items() if k not in extras}
    common['evaluation_file'] = common.get('training_file')
    validate_common(common,protocol=definition(),identifier=protocol_id())
    if (type(request['shard']) is not int or request['shard'] not in (0,1)
            or not isinstance(request['specialist_file'],str) or not Path(request['specialist_file']).is_absolute()):
        raise ValueError('Outcome collection needs a bounded shard and local specialist path')


def run(request):
    validate_request(request); started = time.monotonic()
    from .public_bases import BASES,base_hash
    from .public_math import load_snapshot
    from .weights_io import digest_file
    for name,expected in BASES['qwen3-1.7b'].items():
        if digest_file(Path(request['base_directory'])/name)[0] != expected: raise ValueError('Outcome base differs from its public pin')
    if digest_file(request['adapter_file'])[0] != PARENT_MODEL or digest_file(request['specialist_file'])[0] != SPECIALIST_ID:
        raise ValueError('Outcome collection cannot substitute learned weights')
    snapshot = load_snapshot(request['training_file'],'train'); selected,indices = partition(snapshot,request['shard'])
    output = Path(request['output_directory']); output.mkdir(mode=0o700,parents=True,exist_ok=False)
    from src.network.telemetry import atomic_json
    def progress(label,done,total): atomic_json(output/'progress.json',{'phase':label,'completed':done,'total':total,'timestamp':time.time()})
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from .language_adapter import install_adapter,load_adapter
    from .language_streamed_float import install_streamed_float32
    from .language_specialist import FrozenAdapterBank
    from .public_math_job import evaluate_math
    torch.set_num_threads(request['cpu_threads']); torch.manual_seed(definition()['seed'])
    torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    if request['device']=='cuda': torch.cuda.set_per_process_memory_fraction(request['gpu_fraction']); torch.cuda.reset_peak_memory_stats()
    progress('load-frozen-experts',0,128)
    tokenizer = AutoTokenizer.from_pretrained(request['base_directory'],local_files_only=True,trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(request['base_directory'],local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16,device_map={'':request['device']}).eval()
    parameters = install_adapter(model); load_adapter(model,request['adapter_file'],base_hash('qwen3-1.7b'))
    execution = install_streamed_float32(model)
    parent = {n:p.detach().clone() for n,p in parameters.items()}
    try:
        load_adapter(model,request['specialist_file'],base_hash('qwen3-1.7b'))
        specialist = {n:p.detach().clone() for n,p in parameters.items()}
    finally:
        with torch.no_grad():
            for n,p in parameters.items(): p.copy_(parent[n]); p.requires_grad_(False)
    bank = FrozenAdapterBank(model,parameters,[parent,specialist]); outcomes = {}
    for choice,name in enumerate(definition()['experts']):
        progress(name,0,64)
        with bank.activate(choice) as active:
            outcomes[name] = evaluate_math(active,tokenizer,selected,request['device'],request['deadline'],
                progress=lambda done,total,name=name:progress(name,done,total),record_generation_seconds=True,
                score=score_training_answers)
    if not all(torch.equal(p,parent[n]) for n,p in parameters.items()): raise ValueError('Frozen outcome generation changed the parent')
    if digest_file(request['adapter_file'])[0] != PARENT_MODEL or digest_file(request['specialist_file'])[0] != SPECIALIST_ID:
        raise ValueError('Outcome generation changed a learned input file')
    value = {'version':1,'protocol_id':protocol_id(),'expert_ids':[PARENT_MODEL,SPECIALIST_ID],
        'shard':request['shard'],'training_indices':indices,'experts':outcomes,'evaluation_data_used':False,'optimizer_updates':False}
    verified = verify_outcomes(value,snapshot); raw = canonical(value)
    if len(raw)>4*1024**2: raise ValueError('Outcome calibration exceeds its public artifact bound')
    (output/'outcomes.json').write_bytes(raw)
    report = {**verified,'success':True,'base_execution':execution,'outcome_artifact_id':hashlib.sha256(raw).hexdigest(),
        'outcome_bytes':len(raw),'seconds':time.monotonic()-started,
        'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated() if request['device']=='cuda' else 0,
        'parent_parameters_restored':True,'updated_model_files_written':False,'native_learning_cycles_added':0}
    encoded = canonical(report); (output/'diagnostic.json').write_bytes(encoded); progress('complete',128,128)
    return {'success':True,'protocol_id':protocol_id(),'report_id':hashlib.sha256(encoded).hexdigest(),'bytes':len(encoded)}


def main():
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    raw = sys.stdin.buffer.read(16*1024+1)
    if len(raw)>16*1024: raise ValueError('Outcome request exceeds its bound')
    with contextlib.redirect_stdout(sys.stderr): response = run(json.loads(raw))
    print(json.dumps(response,allow_nan=False),flush=True)


if __name__=='__main__': main()
