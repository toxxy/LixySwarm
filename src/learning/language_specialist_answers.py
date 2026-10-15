"""Read-only answer diagnostic of the saved specialist; never retries training."""
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from .language_retention import PARENT_MODEL,PARENT_RELEASE
from .public_math import ARTIFACTS,gate_id

SPECIALIST_ID='b0127aeb99fae82990f8e97cdb450af2c827c4d89a8b4b29eb1caa8ee340568a'
BASELINE_REPORT_ID='4a0bfc91aaf9e8d8bd754ad9df456cda5ec0901963b63eabb875c06776e72c3a'


def definition():
    return dict(version=1,action='public-specialist-answer-diagnostic-v1',parent_model_id=PARENT_MODEL,
        parent_release_id=PARENT_RELEASE,specialist_id=SPECIALIST_ID,cached_parent_report_id=BASELINE_REPORT_ID,
        evaluation_artifact_id=ARTIFACTS['evaluation'][0],gate_id=gate_id(),compute_dtype='float32',streamed_base=True,
        ram_gb=10,disk_reservation_gb=.01,maximum_seconds=1800,training_performed=False,
        selection_or_hyperparameter_updates=False,full_global_quality_gate_evaluated=False,
        original_combined_system_rejection_retained=True,native_publication_authorized=False,
        scope='Read-only mathematical answer measurement after the separate system rejection, no promotion or training retry')


def protocol_id():
    return hashlib.sha256(json.dumps(definition(),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def run(request):
    from .language_selector_job import validate_request
    extras={'specialist_file','cached_baseline_file'}
    if not isinstance(request,dict) or not extras<=set(request):raise ValueError('Specialist answer diagnostic needs its bound public artifacts')
    validate_request({k:v for k,v in request.items() if k not in extras},protocol=definition(),identifier=protocol_id())
    if any(not isinstance(request[k],str) or not Path(request[k]).is_absolute() for k in extras):raise ValueError('Specialist diagnostic paths must be locally selected')
    from .weights_io import digest_file
    from .public_bases import BASES,base_hash
    from .public_math import load_snapshot,canonical,score_answers,compact_evaluation
    from .public_math_job import evaluate_math
    for name,expected in BASES['qwen3-1.7b'].items():
        if digest_file(Path(request['base_directory'])/name)[0]!=expected:raise ValueError('Specialist diagnostic base mismatch')
    if digest_file(request['adapter_file'])[0]!=PARENT_MODEL or digest_file(request['specialist_file'])[0]!=SPECIALIST_ID:
        raise ValueError('Specialist answer diagnostic cannot substitute learned weights')
    if digest_file(request['cached_baseline_file'])[0]!=BASELINE_REPORT_ID:raise ValueError('Specialist diagnostic cached parent differs')
    cached=json.loads(Path(request['cached_baseline_file']).read_text())
    baseline=cached['full_public_math_trace']['before'];evaluation=load_snapshot(request['evaluation_file'],'evaluation')
    source=cached['research_report']['diagnostic']
    if source['model_id']!=PARENT_MODEL or source['parent_release_id']!=PARENT_RELEASE or cached['full_public_math_trace']['evaluation_artifact_id']!=ARTIFACTS['evaluation'][0]:
        raise ValueError('Cached parent has another model, profile or evaluation')
    before=score_answers(baseline['answers'],evaluation)
    if sum(before.values())!=baseline['correct']:raise ValueError('Cached parent answer count does not reconstruct')
    output=Path(request['output_directory']);output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    def progress(done,total):
        (output/'progress.json').write_bytes(canonical({'phase':'specialist-math-answers','completed':done,'total':total,'timestamp':time.time()}))
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from .language_adapter import install_adapter,load_adapter
    from .language_streamed_float import install_streamed_float32
    torch.set_num_threads(request['cpu_threads']);torch.manual_seed(20261015)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if request['device']=='cuda':torch.cuda.set_per_process_memory_fraction(request['gpu_fraction']);torch.cuda.reset_peak_memory_stats()
    tokenizer=AutoTokenizer.from_pretrained(request['base_directory'],local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(request['base_directory'],local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16,device_map={'':request['device']}).eval()
    parameters=install_adapter(model);load_adapter(model,request['specialist_file'],base_hash('qwen3-1.7b'))
    execution=install_streamed_float32(model);initial={n:p.detach().clone() for n,p in parameters.items()}
    result=evaluate_math(model,tokenizer,evaluation,request['device'],request['deadline'],progress=progress)
    if not all(torch.equal(p,initial[n]) for n,p in parameters.items()):raise ValueError('Read-only generation changed the specialist')
    if digest_file(request['specialist_file'])[0]!=SPECIALIST_ID or digest_file(request['adapter_file'])[0]!=PARENT_MODEL:
        raise ValueError('A read-only diagnostic input file changed')
    after=score_answers(result['answers'],evaluation);regressions=sorted(k for k in before if before[k] and not after[k])
    trace={'version':1,'evaluation_artifact_id':ARTIFACTS['evaluation'][0],'before':baseline,'after':result}
    raw_trace=canonical(trace)
    if len(raw_trace)>4*1024**2:raise ValueError('Specialist answer evidence exceeds its trace bound')
    (output/'math-traces.json').write_bytes(raw_trace)
    gain=sum(after.values())-sum(before.values())
    report={'success':True,'protocol_id':protocol_id(),'model_id':PARENT_MODEL,'parent_release_id':PARENT_RELEASE,
        'evaluated_specialist_id':SPECIALIST_ID,'base_execution':execution,'cached_parent_report_id':BASELINE_REPORT_ID,
        'math_before':compact_evaluation(baseline),'math_after':compact_evaluation(result),
        'correct_before':sum(before.values()),'correct_after':sum(after.values()),'accuracy_gain':gain,'regressions':regressions,
        'math_only_conditions':{'gain_at_least_four':gain>=4,'no_regressions':not regressions,
            'nll_improves_at_least_one_percent':result['nll']<baseline['nll']-max(1e-5,baseline['nll']*.01)},
        'training_performed':False,'selection_or_hyperparameter_updates':False,'generation_performed':True,
        'full_global_quality_gate_evaluated':False,
        'held_out_data_used_in_compute':True,'updated_model_files_written':False,'parent_parameters_restored':True,
        'original_combined_system_rejection_retained':True,'native_publication_authorized':False,'native_learning_cycles_added':0,
        'math_trace_id':hashlib.sha256(raw_trace).hexdigest(),'math_trace_bytes':len(raw_trace),
        'seconds':time.monotonic()-started,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated() if request['device']=='cuda' else 0}
    raw=canonical(report);(output/'diagnostic.json').write_bytes(raw);progress(len(after),len(after))
    return {'success':True,'protocol_id':protocol_id(),'report_id':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}


def main():
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    raw=sys.stdin.buffer.read(16*1024+1)
    if len(raw)>16*1024:raise ValueError('Specialist answer request exceeds its bound')
    with contextlib.redirect_stdout(sys.stderr):response=run(json.loads(raw))
    print(json.dumps(response,allow_nan=False),flush=True)


if __name__=='__main__':main()
