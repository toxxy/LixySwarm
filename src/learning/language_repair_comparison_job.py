"""Read-only, counterbalanced fresh generation with two frozen adapters."""
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from .language_repair_comparison import (definition, protocol_id, load_snapshot,
    needs_second_call, repair_messages, verify_report, SNAPSHOT_ID)
from .language_retention import PARENT_MODEL, PARENT_RELEASE
from .language_specialist_answers import SPECIALIST_ID
from .public_math import canonical, prompt


def durable_json(path, value):
    from src.network.telemetry import atomic_json
    atomic_json(path, value)
    with path.open('rb') as stream: os.fsync(stream.fileno())
    if os.name == 'posix':
        fd = os.open(path.parent, os.O_RDONLY|os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)


def validate_request(request):
    p = definition()
    fields = {'action','protocol_id','compute_dtype','streamed_base','base_directory','adapter_file',
        'model_id','parent_release_id','snapshot_file','specialist_file','output_directory','device',
        'cpu_threads','gpu_fraction','deadline'}
    if (not isinstance(request,dict) or set(request) != fields or request['action'] != p['action']
            or request['protocol_id'] != protocol_id() or request['model_id'] != PARENT_MODEL
            or request['parent_release_id'] != PARENT_RELEASE or request['compute_dtype'] != 'float32'
            or request['streamed_base'] is not True or request['device'] not in {'cuda','cpu'}):
        raise ValueError('Unknown prospective format comparison request')
    if (type(request['cpu_threads']) is not int or not 1 <= request['cpu_threads'] <= 16
            or type(request['gpu_fraction']) not in (int,float) or not 0 < request['gpu_fraction'] <= 1
            or type(request['deadline']) not in (int,float) or not math.isfinite(request['deadline'])
            or not time.time() < request['deadline'] <= time.time()+p['maximum_seconds']):
        raise ValueError('Invalid format comparison resource bounds')
    keys = ['base_directory','adapter_file','snapshot_file','specialist_file','output_directory']
    if any(not isinstance(request[k],str) or not Path(request[k]).is_absolute() for k in keys):
        raise ValueError('Comparison paths must be locally selected absolute paths')
    output = Path(request['output_directory']).resolve()
    if (output.exists() or output.is_relative_to(Path.home()/'Dropbox')
            or output.is_relative_to(Path(__file__).resolve().parents[2])
            or any(output.is_relative_to(Path(request[k]).resolve()) or Path(request[k]).resolve().is_relative_to(output) for k in keys[:-1])):
        raise ValueError('Comparison must retain separate fresh external evidence')


def collect(rows, generate, output, progress):
    """The generator receives question/draft messages, never reference answers."""
    records = []
    durable_json(output/'prospective-commitment.json', {'protocol_id':protocol_id(),
        'snapshot_id':SNAPSHOT_ID,'cases':[row['case'] for row in rows],
        'rule':definition()['rule'],'second_call_order':definition()['second_call_order']})
    for index, row in enumerate(rows):
        messages = [{'role':'user','content':prompt({'question':row['question']})}]
        parent = generate(messages, 0)
        second = needs_second_call(parent['answer'])
        order = ['repair','specialist'] if index%2 == 0 else ['specialist','repair']
        decision = {'case':row['case'],'parent_answer_id':hashlib.sha256(parent['answer'].encode()).hexdigest(),
            'second_requested':second,'second_order':order}
        durable_json(output/f'decision-{index:03}.json', decision)
        responses = {}
        if second:
            for name in order:
                responses[name] = generate(repair_messages(row['question'],parent['answer']) if name == 'repair' else messages,
                    0 if name == 'repair' else 1)
        records.append({'case':row['case'],'parent':parent,'second_requested':second,
            'second_order':order,'responses':responses})
        durable_json(output/'partial-outcomes.json', {'protocol_id':protocol_id(),'records':records})
        progress(index+1,len(rows))
    return records


def run(request):
    validate_request(request); p = definition(); started = time.monotonic()
    from .public_bases import BASES, base_hash
    from .weights_io import digest_file
    from src.network.telemetry import atomic_json
    base = Path(request['base_directory'])
    for name, identifier in BASES['qwen3-1.7b'].items():
        if digest_file(base/name)[0] != identifier: raise ValueError('Comparison base differs from public provenance')
    if digest_file(request['adapter_file'])[0] != PARENT_MODEL or digest_file(request['specialist_file'])[0] != SPECIALIST_ID:
        raise ValueError('Comparison substituted a frozen expert')
    snapshot = load_snapshot(request['snapshot_file']); output = Path(request['output_directory']); output.mkdir(mode=0o700)
    def check():
        if time.time() >= request['deadline']: raise TimeoutError('Prospective comparison deadline expired')
    def progress(done,total):
        check(); atomic_json(output/'progress.json', {'phase':'fresh-format-comparison','done':done,'total':total,'timestamp':time.time()})
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .language_adapter import install_adapter, load_adapter
    from .language_streamed_float import install_streamed_float32
    from .language_specialist import FrozenAdapterBank
    from .language_prefix_router_job import generate_prefix
    torch.set_num_threads(request['cpu_threads']); torch.manual_seed(p['seed']); torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if request['device'] == 'cuda':
        torch.cuda.set_per_process_memory_fraction(request['gpu_fraction']); torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(base,local_files_only=True,trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(base,local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16,device_map={'':request['device']}).eval()
    parameters = install_adapter(model); load_adapter(model,request['adapter_file'],base_hash('qwen3-1.7b'))
    execution = install_streamed_float32(model); model.config.use_cache = False
    parent = {name:v.detach().clone() for name,v in parameters.items()}
    try:
        load_adapter(model,request['specialist_file'],base_hash('qwen3-1.7b'))
        specialist = {name:v.detach().clone() for name,v in parameters.items()}
    finally:
        with torch.no_grad():
            for name, value in parameters.items(): value.copy_(parent[name]); value.requires_grad_(False)
    bank = FrozenAdapterBank(model,parameters,[parent,specialist])
    def generate(messages, expert):
        check()
        if sum(len(row['content'].encode()) for row in messages) > p['maximum_messages_bytes_per_call']:
            raise ValueError('Comparison message context cannot be truncated')
        text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids = tokenizer.encode(text,add_special_tokens=False)
        if not 1 <= len(ids) <= p['maximum_input_tokens_per_call']:
            raise ValueError('Comparison prefix cannot be truncated')
        with bank.activate(expert) as active:
            result = generate_prefix(active,tokenizer,ids,device=request['device'],deadline=request['deadline'],maximum=p['maximum_new_tokens_per_call'])
        return {**result,'input_tokens':len(ids)}
    records = collect(snapshot['rows'],generate,output,progress)
    if (not all(torch.equal(v,parent[name]) for name,v in parameters.items())
            or digest_file(request['adapter_file'])[0] != PARENT_MODEL
            or digest_file(request['specialist_file'])[0] != SPECIALIST_ID):
        raise ValueError('Read-only comparison mutated its experts')
    report = {'success':True,'protocol_id':protocol_id(),'snapshot_id':SNAPSHOT_ID,
        'records':records,'base_execution':execution,'training_performed':False,'model_promoted':False,
        'seconds':time.monotonic()-started,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated() if request['device']=='cuda' else 0}
    report['verification'] = verify_report(report,snapshot)
    raw = canonical(report)
    if len(raw) > 4*1024**2: raise ValueError('Comparison evidence exceeds its reserved bound')
    durable_json(output/'diagnostic.json',report);progress(128,128)
    return {'success':True,'protocol_id':protocol_id(),'report_id':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}


def main():
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    raw = sys.stdin.buffer.read(16*1024+1)
    if len(raw)>16*1024: raise ValueError('Comparison request exceeds its bound')
    with contextlib.redirect_stdout(sys.stderr): result = run(json.loads(raw))
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__ == '__main__': main()
