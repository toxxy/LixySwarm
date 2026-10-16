"""Offline GPU child; only questions and prior generated drafts reach generate."""
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from .language_repair_calibration import (definition,protocol_id,load_outcomes,selected_rows,
    commitment,verify_calibration,needs_second_call,repair_messages)
from .language_retention import PARENT_MODEL,PARENT_RELEASE
from .public_math import canonical
from .language_repair_comparison_job import durable_json


def validate_request(request):
    p=definition()
    fields={'action','protocol_id','compute_dtype','streamed_base','base_directory','adapter_file',
        'model_id','parent_release_id','training_file','outcome_files','shard','output_directory',
        'device','cpu_threads','gpu_fraction','deadline'}
    if (not isinstance(request,dict) or set(request)!=fields or request['action']!=p['action']
            or request['protocol_id']!=protocol_id() or request['model_id']!=PARENT_MODEL
            or request['parent_release_id']!=PARENT_RELEASE or request['compute_dtype']!='float32'
            or request['streamed_base'] is not True or request['device'] not in {'cpu','cuda'}
            or type(request['shard']) is not int or request['shard'] not in (0,1)):
        raise ValueError('Only the pinned public TRAIN repair calibration may run')
    if (type(request['cpu_threads']) is not int or not 1<=request['cpu_threads']<=16
            or type(request['gpu_fraction']) not in (int,float) or not 0<request['gpu_fraction']<=1
            or type(request['deadline']) not in (int,float) or not math.isfinite(request['deadline'])
            or not time.time()<request['deadline']<=time.time()+p['maximum_seconds']):
        raise ValueError('Invalid repair calibration resource bounds')
    paths=[request[k] for k in ['base_directory','adapter_file','training_file','output_directory']]
    if (not isinstance(request['outcome_files'],list) or len(request['outcome_files'])!=2
            or any(not isinstance(v,str) or not Path(v).is_absolute() for v in [*paths,*request['outcome_files']])):
        raise ValueError('Repair calibration inputs must be selected locally with absolute paths')
    output=Path(request['output_directory']).resolve()
    if (output.exists() or output.is_relative_to(Path.home()/'Dropbox')
            or output.is_relative_to(Path(__file__).resolve().parents[2])
            or any(output.is_relative_to(Path(v).resolve()) or Path(v).resolve().is_relative_to(output)
                   for v in [*paths[:-1],*request['outcome_files']])):
        raise ValueError('Fresh external evidence must be separate from all calibration inputs')


def collect(snapshot,values,shard,generate,output,progress):
    frozen=commitment(snapshot,values,shard)
    durable_json(output/'prospective-commitment.json',frozen)
    records=[]
    for offset,(_,row,draft) in enumerate(selected_rows(snapshot,values,shard)):
        record=dict(frozen['cases'][offset])
        record['repair']=generate(repair_messages(row['question'],draft)) if needs_second_call(draft) else None
        records.append(record)
        durable_json(output/'partial-repairs.json',{'protocol_id':protocol_id(),'records':records})
        progress(offset+1,64)
    value={'version':1,'protocol_id':protocol_id(),'shard':shard,'commitment':frozen,'records':records,
           'evaluation_data_used':False,'optimizer_updates':False}
    verify_calibration(value,snapshot,values)
    return value


def run(request):
    validate_request(request);p=definition();began=time.monotonic()
    from .public_math import load_snapshot
    from .public_bases import BASES,base_hash
    from .weights_io import digest_file
    base=Path(request['base_directory'])
    for name,identifier in BASES['qwen3-1.7b'].items():
        if digest_file(base/name)[0]!=identifier:raise ValueError('Repair calibration base differs from its public pin')
    if digest_file(request['adapter_file'])[0]!=PARENT_MODEL:raise ValueError('Repair calibration parent is not accepted model 2')
    snapshot=load_snapshot(request['training_file'],'train')
    values=load_outcomes(request['outcome_files'],snapshot)
    output=Path(request['output_directory']);output.mkdir(mode=0o700,parents=True)
    def check():
        if time.time()>=request['deadline']:raise TimeoutError('Repair calibration deadline expired')
    def progress(done,total):
        check();durable_json(output/'progress.json',{'phase':'collect-public-train-repairs','done':done,'total':total,'timestamp':time.time()})
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from .language_adapter import install_adapter,load_adapter
    from .language_streamed_float import install_streamed_float32
    from .language_prefix_router_job import generate_prefix
    torch.set_num_threads(request['cpu_threads']);torch.manual_seed(p['seed']);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if request['device']=='cuda':torch.cuda.set_per_process_memory_fraction(request['gpu_fraction']);torch.cuda.reset_peak_memory_stats()
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16,device_map={'':request['device']}).eval()
    parameters=install_adapter(model);load_adapter(model,request['adapter_file'],base_hash('qwen3-1.7b'))
    execution=install_streamed_float32(model)
    before={k:v.detach().clone() for k,v in parameters.items()}
    for parameter in parameters.values():parameter.requires_grad_(False)
    def generate(messages):
        check()
        if sum(len(row['content'].encode()) for row in messages)>p['maximum_messages_bytes_per_call']:
            raise ValueError('Calibration context cannot be truncated')
        text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        prefix=tokenizer.encode(text,add_special_tokens=False)
        if not 1<=len(prefix)<=p['maximum_input_tokens_per_call']:
            raise ValueError('Calibration input token bound exceeded; no truncation permitted')
        return {**generate_prefix(model,tokenizer,prefix,device=request['device'],deadline=request['deadline'],
            maximum=p['maximum_new_tokens_per_call']),'input_tokens':len(prefix)}
    value=collect(snapshot,values,request['shard'],generate,output,progress)
    if (not all(torch.equal(v,before[k]) for k,v in parameters.items())
            or digest_file(request['adapter_file'])[0]!=PARENT_MODEL):raise ValueError('Read-only calibration changed its parent')
    raw=canonical(value);path=output/'calibration.json'
    if len(raw)>1024**2:raise ValueError('Calibration artifact exceeds its public bound')
    with path.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    report={'success':True,'protocol_id':protocol_id(),'shard':request['shard'],
        'artifact_id':hashlib.sha256(raw).hexdigest(),'artifact_bytes':len(raw),
        'verification':verify_calibration(value,snapshot,values),'base_execution':execution,
        'seconds':time.monotonic()-began,'parent_parameters_unchanged':True,
        'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated() if request['device']=='cuda' else 0}
    durable_json(output/'diagnostic.json',report);encoded=(output/'diagnostic.json').read_bytes()
    return {'success':True,'protocol_id':protocol_id(),'report_id':hashlib.sha256(encoded).hexdigest(),'bytes':len(encoded)}


def main():
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    raw=sys.stdin.buffer.read(16*1024+1)
    if len(raw)>16*1024:raise ValueError('Calibration request exceeds its bound')
    with contextlib.redirect_stdout(sys.stderr):response=run(json.loads(raw))
    print(json.dumps(response,allow_nan=False),flush=True)


if __name__=='__main__':main()
