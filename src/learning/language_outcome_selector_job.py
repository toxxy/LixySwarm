"""Learn only the selector; reuse verified public TRAIN answer measurements."""
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from .language_outcome_selector import definition,protocol_id,dataset,fold_indices,verify_report,OUTCOME_IDS
from .public_math import canonical

os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'


def validate_request(request):
    fields={'action','protocol_id','training_file','outcome_files','base_directory','output_directory','device','cpu_threads','gpu_fraction','deadline'}
    if (not isinstance(request,dict) or set(request)!=fields or request['action']!=definition()['action']
            or request['protocol_id']!=protocol_id() or request['device'] not in {'cpu','cuda'}
            or type(request['cpu_threads']) is not int or not 1<=request['cpu_threads']<=8
            or type(request['gpu_fraction']) not in (int,float) or not 0<request['gpu_fraction']<=1
            or type(request['deadline']) not in (int,float) or not math.isfinite(request['deadline'])
            or not time.time()<request['deadline']<=time.time()+definition()['maximum_seconds']):
        raise ValueError('Unknown bounded outcome selector request')
    if not isinstance(request['outcome_files'],list) or len(request['outcome_files'])!=2:raise ValueError('Two exact outcome files required')
    paths=[request[k] for k in ['training_file','base_directory','output_directory']]+request['outcome_files']
    if any(not isinstance(p,str) or not Path(p).is_absolute() for p in paths):raise ValueError('Outcome paths must be locally selected absolute paths')
    output=Path(request['output_directory']).resolve()
    if (output.exists() or output.is_relative_to(Path.home()/'Dropbox')
            or output.is_relative_to(Path(__file__).resolve().parents[2])
            or any(output.is_relative_to(Path(p).resolve()) or Path(p).resolve().is_relative_to(output) for p in paths if p!=request['output_directory'])):
        raise ValueError('Outcome research must use a fresh external directory separate from its inputs')


def run(request):
    validate_request(request);started=time.monotonic();p=definition()
    from .public_bases import BASES
    from .public_math import load_snapshot,prompt
    from .weights_io import digest_file
    from src.network.telemetry import atomic_json
    snapshot=load_snapshot(request['training_file'],'train');outcomes=[]
    for path,identifier in zip(request['outcome_files'],OUTCOME_IDS):
        path=Path(path)
        if path.stat().st_size>128*1024 or digest_file(path)[0]!=identifier:raise ValueError('Physical outcome artifact was substituted')
        outcomes.append(json.loads(path.read_text()))
    rows=dataset(outcomes,snapshot);base=Path(request['base_directory'])
    for name in ['config.json','tokenizer.json','tokenizer_config.json']:
        if digest_file(base/name)[0]!=BASES['qwen3-1.7b'][name]:raise ValueError('Outcome selector tokenizer differs from its pinned public expert')
    output=Path(request['output_directory']);output.mkdir(mode=0o700,parents=True)
    def check():
        if time.time()>=request['deadline']:raise TimeoutError('Outcome selector deadline exceeded')
    def progress(phase,done=0,total=0):
        check();atomic_json(output/'progress.json',{'phase':phase,'done':done,'total':total,'timestamp':time.time()})
    import torch
    from transformers import AutoTokenizer
    from .language_selector import train_selectors,save_selector,load_selector,LanguageExpertSelector,tensor_digest
    torch.set_num_threads(request['cpu_threads']);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if request['device']=='cuda':torch.cuda.set_per_process_memory_fraction(request['gpu_fraction']);torch.cuda.reset_peak_memory_stats()
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True,trust_remote_code=False)
    vocab=json.loads((base/'config.json').read_text())['vocab_size'];queries=[]
    for row in snapshot['rows'][:128]:
        text=tokenizer.apply_chat_template([{'role':'user','content':prompt(row)}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        tokens=tokenizer.encode(text,add_special_tokens=False)
        if not tokens or len(tokens)>1024 or any(not 0<=t<vocab for t in tokens):raise ValueError('Outcome query prefix changed its token bound')
        queries.append(tokens)
    report={'protocol_id':protocol_id(),'folds':[],'predictions':{name:[None]*128 for name in ['dolphin',*p['memory_modes']]},
        'original_evaluation_opened':False,'native_promotion_authorized':False,'language_experts_updated':False,
        'answer_generation_performed':False,'artifacts':{},'targets_id':hashlib.sha256(canonical(rows)).hexdigest()}
    for fold in range(8):
        check();train,test=fold_indices(fold);folder=output/str(fold);folder.mkdir()
        targets=torch.tensor([rows[i]['target'] for i in train],dtype=torch.float32)
        variants,fit,optimizers=train_selectors([queries[i] for i in train],None,vocab,p,teacher_targets=targets,
            device=request['device'],check=check,progress=lambda phase,d,t:progress(str(fold)+'/'+phase,d,t))
        before={name:tensor_digest(model.state_dict()) for name,model in variants.items()}
        binding={'protocol_id':protocol_id(),'base_hash':p['base_hash'],'tokenizer_id':p['tokenizer_id'],
            'dolphin_state_id':fit['dolphin_state_id'],'feature_space':p['memory_feature_space'],
            'memory_scope':'public-training-only','expert_ids':p['expert_ids']}
        def record(path):report['artifacts'][str(path.relative_to(output))]=dict(zip(['sha256','bytes'],digest_file(path)))
        for name,model in variants.items():
            metadata={**binding,'variant':name};path=folder/(name+'.safetensors');save_selector(model,path,metadata);record(path)
            restored=LanguageExpertSelector(vocab,support_count=len(fit['support_indices'])).to(request['device'])
            load_selector(restored,path,metadata);restored.eval()
            if tensor_digest(restored.state_dict())!=before[name]:raise ValueError('Outcome selector reload changed its state')
            choices,probabilities=restored.choose([queries[i] for i in test],use_memory=name!='dolphin')
            for i,choice,probability in zip(test,choices.cpu().tolist(),probabilities.cpu().tolist()):
                report['predictions'][name][i]={'index':i,'choice':choice,'probabilities':probability}
            if tensor_digest(restored.state_dict())!=before[name]:raise ValueError('Prediction wrote query history into public memory')
            del restored
        path=folder/'optimizer-state.pt';torch.save(optimizers,path);record(path)
        report['folds'].append({'training_indices':train,'prediction_indices':test,'fit':fit,'binding':binding})
        if sum(v['bytes'] for v in report['artifacts'].values())>p['disk_reservation_gb']*1024**3:raise ValueError('Outcome learned state exceeded its reserved storage')
        atomic_json(output/'partial-report.json',report);del model,variants,optimizers
        if request['device']=='cuda':torch.cuda.empty_cache()
    report['verification']=verify_report(report,rows)
    report.update(success=True,seconds=time.monotonic()-started,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if request['device']=='cuda' else 0)
    raw=canonical(report);(output/'diagnostic.json').write_bytes(raw);progress('complete',8,8)
    return {'success':True,'report_id':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}


def main():
    raw=sys.stdin.buffer.read(16*1024+1)
    if len(raw)>16*1024:raise ValueError('Outcome selector request exceeds its bound')
    with contextlib.redirect_stdout(sys.stderr):response=run(json.loads(raw))
    print(json.dumps(response,allow_nan=False),flush=True)


if __name__=='__main__':main()
