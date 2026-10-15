"""Offline public-language candidate and paired evaluation; never publishes."""
import json
import math
import re
import sys
import time
from pathlib import Path

from .public_bases import BASES,base_hash
from .public_conversation import public_corpus,corpus_digest
from .weights_io import digest_file
from .language_policy import policy_from_id,limits,HALF_POLICY

HEX=re.compile(r'[0-9a-f]{64}')
FIELDS={'action','base_name','base_directory','adapter_file','adapter_id','dataset_id',
        'parent_release_id','contributions','learning_rate','output_file','cpu_threads',
        'gpu_fraction','device','deadline'}
GRADIENT_FIELDS={'path','gradient_id','gradient_bytes','base_hash','adapter_id','dataset_id',
                 'start','count','tokens'}


def validate_request(request):
    """A local supervisor supplies paths; peers cannot choose this request."""
    policy_version=policy_from_id(request.get('training_policy_id')) if isinstance(request,dict) else 1
    expected=FIELDS|({'training_policy_id'} if policy_version>=2 else set())
    if not isinstance(request,dict) or set(request)!=expected or request['action']!='candidate':
        raise ValueError('Invalid language candidate request')
    if request['base_name'] not in BASES or request['device'] not in {'cpu','cuda'}:
        raise ValueError('Unknown candidate base or device')
    if any(not isinstance(request[k],str) or not HEX.fullmatch(request[k])
           for k in ['adapter_id','parent_release_id','dataset_id']):
        raise ValueError('Invalid candidate content identity')
    if (type(request['learning_rate']) not in (int,float) or not 1e-6<=request['learning_rate']<=1e-3
            or type(request['cpu_threads']) is not int or not 1<=request['cpu_threads']<=16
            or type(request['gpu_fraction']) not in (int,float) or not 0<request['gpu_fraction']<=1
            or type(request['deadline']) not in (int,float) or not math.isfinite(request['deadline'])
            or not time.time()<request['deadline']<=time.time()+3600):
        raise ValueError('Invalid bounded candidate resources or deadline')
    for key in ['base_directory','adapter_file','output_file']:
        if not isinstance(request[key],str) or not Path(request[key]).is_absolute():
            raise ValueError('Candidate inputs and output must be locally selected absolute paths')
    output=Path(request['output_file'])
    if output.exists() or output.is_symlink() or output==Path(request['adapter_file']):
        raise ValueError('Candidate output must be new and cannot replace the parent')
    corpus=public_corpus()
    if request['dataset_id']!=corpus_digest(corpus):raise ValueError('Candidate requires the exact public corpus')
    rows=request['contributions']
    if not isinstance(rows,list) or not 1<=len(rows)<=16:raise ValueError('Invalid number of candidate contributions')
    for row in rows:
        if (not isinstance(row,dict) or set(row)!=(GRADIENT_FIELDS|({'quantization_relative_l2_error'} if policy_version==3 else set()))
                or row['base_hash']!=base_hash(request['base_name']) or row['adapter_id']!=request['adapter_id']
                or row['dataset_id']!=request['dataset_id'] or not isinstance(row['gradient_id'],str)
                or not HEX.fullmatch(row['gradient_id']) or type(row['gradient_bytes']) is not int
                or not 1<=row['gradient_bytes']<=32*1024**2
                or type(row['start']) is not int or type(row['count']) is not int
                or not 1<=row['count']<=limits(policy_version)[0] or not 0<=row['start']<=len(corpus['train'])-row['count']
                or type(row['tokens']) is not int or not 1<=row['tokens']<=limits(policy_version)[1]
                or policy_version>=2 and row['tokens']>192*row['count']
                or not isinstance(row['path'],str) or not Path(row['path']).is_absolute()):
            raise ValueError('Candidate contribution has incompatible version, data, shard or bounds')
        if policy_version==3:
            error=row['quantization_relative_l2_error']
            if type(error) not in (int,float) or not math.isfinite(error) or not 0<=error<=HALF_POLICY['maximum_relative_l2_error']:
                raise ValueError('Candidate half-precision error exceeds its bound')
    ranges=sorted((row['start'],row['start']+row['count']) for row in rows)
    if any(a[1]>b[0] for a,b in zip(ranges,ranges[1:])):raise ValueError('Candidate shards overlap')
    if len({row['gradient_id'] for row in rows})!=len(rows):raise ValueError('Candidate repeats a gradient')
    return corpus


def apply_update(model,contributions,learning_rate,*,policy_version=1):
    """One token-weighted SGD step on compatible adapter parameters only."""
    import torch
    from .language_adapter import adapter_parameters
    from .language_gradient import combine_gradients
    if type(learning_rate) not in (int,float) or not 1e-6<=learning_rate<=1e-3:
        raise ValueError('Invalid bounded candidate learning rate')
    combined=combine_gradients(model,contributions,policy_version=policy_version)
    with torch.no_grad():
        for name,parameter in adapter_parameters(model).items():
            parameter.add_(combined[name].to(parameter.device),alpha=-learning_rate)
            if not torch.isfinite(parameter).all():raise ValueError('Nonfinite candidate adapter')


def evaluate(model,tokenizer,corpus,device,deadline):
    import torch
    from transformers import StoppingCriteria,StoppingCriteriaList
    from .public_quality import quality_cases
    from .public_language_gate import cases
    def check():
        if time.time()>=deadline:raise TimeoutError('Language candidate evaluation deadline exceeded')
    class Deadline(StoppingCriteria):
        def __call__(self,*_a,**_kw):return time.time()>=deadline
    def prompt(messages):
        text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        return tokenizer.encode(text,add_special_tokens=False)
    model.eval();total=tokens=0
    with torch.inference_mode():
        for row in corpus['evaluation']:
            check()
            prefix=prompt([{'role':'user','content':row['prompt']}])
            answer=tokenizer.encode(row['answer'],add_special_tokens=False)+[tokenizer.eos_token_id]
            if len(prefix)+len(answer)>192:raise ValueError('Public evaluation exceeds its bounded sequence')
            inputs=torch.tensor([prefix+answer],device=device)
            labels=torch.tensor([[-100]*len(prefix)+answer],device=device)
            loss=float(model(input_ids=inputs,labels=labels,use_cache=False).loss)
            if not math.isfinite(loss) or loss<0:raise ValueError('Invalid candidate evaluation loss')
            total+=loss*len(answer);tokens+=len(answer)
        def generate(name,messages):
            check();prefix=prompt(messages)
            if len(prefix)>1024:raise ValueError('Public gate exceeds its bounded context')
            inputs=torch.tensor([prefix],device=device)
            output=model.generate(input_ids=inputs,attention_mask=torch.ones_like(inputs),max_new_tokens=128,
                do_sample=False,use_cache=True,stopping_criteria=StoppingCriteriaList([Deadline()]))
            check();answer=tokenizer.decode(output[0,len(prefix):],skip_special_tokens=True).strip()
            if len(answer.encode('utf-8'))>1024:raise ValueError('Public evaluation answer exceeds its bound')
            return {'case':name,'answer':answer}
        quality=[generate(row['case'],[{'role':'user','content':row['prompt']}]) for row in quality_cases()]
        instructions=[generate(name,messages) for name,messages,_pattern in cases()]
    return {'nll':total/tokens,'tokens':tokens,'quality':quality,'instructions':instructions}


def run(request):
    corpus=validate_request(request);started=time.monotonic()
    for filename,identifier in BASES[request['base_name']].items():
        if time.time()>=request['deadline']:raise TimeoutError('Candidate input verification deadline exceeded')
        if digest_file(Path(request['base_directory'])/filename)[0]!=identifier:raise ValueError('Pinned public candidate base mismatch')
    if digest_file(request['adapter_file'])[0]!=request['adapter_id']:raise ValueError('Pinned candidate parent mismatch')
    for row in request['contributions']:
        if digest_file(row['path'])!=(row['gradient_id'],row['gradient_bytes']):raise ValueError('Candidate gradient content identity mismatch')
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from .language_adapter import install_adapter,load_adapter,save_adapter
    torch.set_num_threads(request['cpu_threads']);torch.manual_seed(20261014)
    device=request['device']
    if device=='cuda':
        torch.cuda.set_per_process_memory_fraction(request['gpu_fraction']);torch.cuda.reset_peak_memory_stats()
    tokenizer=AutoTokenizer.from_pretrained(request['base_directory'],local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(request['base_directory'],local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16 if device=='cuda' else torch.float32,device_map={'':device}).eval()
    install_adapter(model);load_adapter(model,request['adapter_file'],base_hash(request['base_name']))
    before=evaluate(model,tokenizer,corpus,device,request['deadline'])
    policy_version=policy_from_id(request.get('training_policy_id'))
    apply_update(model,request['contributions'],request['learning_rate'],policy_version=policy_version)
    after=evaluate(model,tokenizer,corpus,device,request['deadline'])
    if time.time()>=request['deadline']:raise TimeoutError('Candidate evaluation deadline exceeded')
    path=Path(request['output_file']);save_adapter(model,path,base_hash(request['base_name']))
    identifier,size=digest_file(path)
    from .public_quality import preservation_verdict as factual_verdict
    from .public_language_gate import preservation_verdict as instruction_verdict
    baseline,value=before['nll'],after['nll']
    factual=factual_verdict(before['quality'],after['quality'])
    instruction=instruction_verdict(before['instructions'],after['instructions'])
    return {'version':policy_version,'private_inputs':False,'base_name':request['base_name'],'base_hash':base_hash(request['base_name']),
        'dataset_id':request['dataset_id'],'parent_release_id':request['parent_release_id'],
        'adapter_id':identifier,'adapter_bytes':size,'baseline_nll':baseline,'candidate_nll':value,
        'quality_before':before['quality'],'quality_after':after['quality'],
        'instruction_before':before['instructions'],'instruction_after':after['instructions'],
        'heldout_tokens':after['tokens'],'contribution_tokens':sum(row['tokens'] for row in request['contributions']),
        'learning_rate':request['learning_rate'],'factual_verdict':factual,'instruction_verdict':instruction,
        'numerical_gates_passed':value<baseline-max(1e-7,baseline*.001) and factual['eligible_for_promotion'] and instruction['eligible_for_promotion'],
        'physical_worker_gate_checked':False,'published':False,'seconds':round(time.monotonic()-started,3),
        'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0}


def main():
    raw=sys.stdin.buffer.read(16*1024+1)
    if len(raw)>16*1024:raise ValueError('Candidate request exceeds its bound')
    print(json.dumps(run(json.loads(raw)),ensure_ascii=False,allow_nan=False),flush=True)


if __name__=='__main__':main()
