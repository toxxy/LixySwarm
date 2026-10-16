"""Bounded own-weight pretraining child, using only pinned public documents."""
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from .swarm_foundation_protocol import definition, protocol_id
from .public_math import load_snapshot
from .language_repair_comparison_job import durable_json


def documents(snapshot):
    return [('Question: '+row['question']+'\nAnswer: '+row['answer']+'\n\n').encode()
            for row in snapshot['rows']]


def evaluate(model, docs, device):
    import torch
    import torch.nn.functional as F
    from .swarm_foundation import predict
    p = definition(); total = 0.; correct = tokens = 0
    model.eval()
    with torch.inference_mode():
        for doc in docs:
            doc = doc[:p['evaluation_maximum_bytes_per_document']]
            for start in range(0, len(doc)-1, p['block_size']):
                segment = doc[start:start+p['block_size']+1]
                if len(segment) < 2: continue
                ids = torch.tensor(list(segment), dtype=torch.long, device=device)
                logits, _, _ = predict(model, ids[:-1].unsqueeze(0))
                labels = ids[1:]
                total += float(F.cross_entropy(logits[0], labels, reduction='sum'))
                correct += int((logits[0].argmax(-1) == labels).sum())
                tokens += len(labels)
    if not tokens: raise ValueError('Held-out foundation documents are empty')
    return {'nll': total/tokens, 'byte_accuracy': correct/tokens, 'predicted_bytes': tokens,
            'documents': len(docs)}


def causal_check(model, device):
    import torch
    from .swarm_foundation import predict
    p=definition(); model.eval()
    a=torch.tensor([[41,42,43,44,45,46,47,48]], device=device)
    b=a.clone();b[:,4:]=torch.tensor([91,92,93,94], device=device)
    with torch.inference_mode():
        left=predict(model,a)[0][:,:4];right=predict(model,b)[0][:,:4]
    return {'max_prefix_logits_difference':float((left-right).abs().max()),
        'passed':bool(torch.allclose(left,right,atol=p['causality_atol'],rtol=p['causality_rtol']))}


def run(request):
    p=definition()
    fields={'action','protocol_id','training_file','evaluation_file','output_directory',
            'device','cpu_threads','gpu_fraction','deadline'}
    if (not isinstance(request,dict) or set(request)!=fields or request['action']!=p['action']
            or request['protocol_id']!=protocol_id() or request['device'] not in {'cpu','cuda'}
            or type(request['cpu_threads']) is not int or not 1<=request['cpu_threads']<=16
            or type(request['gpu_fraction']) not in (float,int) or not 0<request['gpu_fraction']<=1
            or type(request['deadline']) not in (float,int) or not math.isfinite(request['deadline'])
            or not time.time()<request['deadline']<=time.time()+p['maximum_seconds']):
        raise ValueError('Only the declared bounded own-foundation experiment is permitted')
    paths=[Path(request[k]) for k in ['training_file','evaluation_file','output_directory']]
    output=paths[-1]
    if (any(not v.is_absolute() or v.is_symlink() for v in paths) or output.exists()
            or output.resolve().is_relative_to(Path.home()/'Dropbox')
            or output.resolve().is_relative_to(Path(__file__).resolve().parents[2])):
        raise ValueError('Own foundation requires explicit public inputs and fresh external output')
    training=load_snapshot(paths[0],'train');heldout=load_snapshot(paths[1],'evaluation')
    if {v['question'] for v in training['rows']} & {v['question'] for v in heldout['rows']}:
        raise ValueError('Training and held-out foundation documents overlap')
    output.mkdir(parents=True,mode=0o700)
    try:
        return _train(request,output,documents(training),documents(heldout))
    except BaseException as error:
        import traceback
        durable_json(output/'failure.json',{'error_type':type(error).__name__,
            'error':str(error)[:4096],'traceback':traceback.format_exc(limit=12)[-16*1024:]})
        raise


def _train(request,output,training,heldout):
    import torch
    from safetensors.torch import save_file,load_file
    from .swarm_foundation import make_foundation_model,predict
    from src.network.artifact_store import digest_file
    p=definition();device=request['device'];began=time.monotonic()
    torch.set_num_threads(request['cpu_threads'])
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if device=='cuda':
        torch.cuda.set_per_process_memory_fraction(request['gpu_fraction'])
        torch.cuda.reset_peak_memory_stats()
    def check():
        if time.time()>=request['deadline']:raise TimeoutError('Own foundation deadline expired')
    def save_weights(model,name):
        save_file({k:v.detach().cpu().clone().contiguous() for k,v in model.state_dict().items()},
            str(output/name),metadata={'format':'lixyswarm-own-causal-byte-v1','protocol_id':protocol_id()})
    model=make_foundation_model(seed=p['initialization_seed'],device=device)
    save_weights(model,'initial.safetensors')
    initial_causality=causal_check(model,device)
    if not initial_causality['passed']:raise ValueError('Own foundation is not prefix causal')
    check();before=evaluate(model,heldout,device);check()
    stream=torch.tensor(list(b''.join(training)),dtype=torch.uint8,device='cpu')
    rng=torch.Generator(device='cpu').manual_seed(p['sample_seed'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'])
    history=[];train_started=time.monotonic();model.train()
    for step in range(p['steps']):
        check()
        starts=torch.randint(0,len(stream)-p['block_size'],(p['batch_size'],),generator=rng)
        batch=torch.stack([stream[int(v):int(v)+p['block_size']+1] for v in starts]).long().to(device)
        for group in optimizer.param_groups:
            group['lr']=p['learning_rate']*min(1.,(step+1)/p['warmup_steps'])
        optimizer.zero_grad(set_to_none=True)
        _,loss,_=predict(model,batch[:,:-1],batch[:,1:])
        if not torch.isfinite(loss):raise ValueError('Own foundation loss is nonfinite')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),p['gradient_clip_norm'],error_if_nonfinite=True)
        optimizer.step()
        if (step+1)%64==0 or step==0:
            item={'step':step+1,'loss':float(loss.detach()),'loss_terms':dict(model._last_loss_terms)}
            history.append(item);durable_json(output/'progress.json',{'phase':'own-byte-training',**item,'timestamp':time.time()})
    train_seconds=time.monotonic()-train_started
    save_weights(model,'model.safetensors')
    optimizer_state=optimizer.state_dict()
    # Store optimizer tensors on CPU for portable resumption on another PC.
    for state in optimizer_state['state'].values():
        for key,value in state.items():
            if isinstance(value,torch.Tensor):state[key]=value.detach().cpu()
    torch.save({'protocol_id':protocol_id(),'step':p['steps'],'optimizer':optimizer_state,
                'sampler_state':rng.get_state()},output/'optimizer.pt')
    check();after=evaluate(model,heldout,device);final_causality=causal_check(model,device)
    if not final_causality['passed']:raise ValueError('Learned foundation ceased to be prefix causal')
    restored=make_foundation_model(seed=p['initialization_seed'],device='cpu').eval()
    restored.load_state_dict(load_file(str(output/'model.safetensors')),strict=True)
    probe=torch.tensor([list(b'Question: What is a swarm?\nAnswer:')],device=device)
    with torch.inference_mode():
        expected=predict(model,probe)[0].cpu();actual=predict(restored,probe.cpu())[0]
    if not torch.allclose(expected,actual,atol=1e-5,rtol=1e-4):
        raise ValueError('Own foundation did not reload its learned predictions')
    artifacts={}
    for name in ['initial.safetensors','model.safetensors','optimizer.pt']:
        identifier,size=digest_file(output/name);artifacts[name]={'sha256':identifier,'bytes':size}
        if size>16*1024**2:raise ValueError('Own foundation artifact exceeds its LSP research bound')
    optimizer_steps=sorted({int(v['step']) for v in optimizer_state['state'].values()})
    if optimizer_steps!=[p['steps']]:raise ValueError('Own foundation optimizer steps differ from declared training')
    report={'success':True,'protocol_id':protocol_id(),'initial_causality':initial_causality,
        'final_causality':final_causality,'before':before,'after':after,
        'relative_heldout_nll_gain':(before['nll']-after['nll'])/before['nll'],
        'baseline_learning_gate_passed':after['nll']<=before['nll']*(1-p['minimum_relative_heldout_nll_gain']),
        'parameters':sum(v.numel() for v in model.parameters()),'artifacts':artifacts,
        'optimizer_states':len(optimizer_state['state']),'optimizer_steps':optimizer_steps,
        'training_history':history,'training_predicted_bytes':p['steps']*p['batch_size']*p['block_size'],
        'train_seconds':train_seconds,'seconds':time.monotonic()-began,
        'maximum_cpu_reload_error':float((expected-actual).abs().max()),
        'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 0,
        'pretrained_weights_loaded':False,'teacher_used':False,'distributed_training_performed':False,
        'model_promoted':False,'native_cycles_added':0,'general_chat_quality_proven':False,
        'scope':p['scope']}
    durable_json(output/'diagnostic.json',report)
    identifier,size=digest_file(output/'diagnostic.json')
    return {'success':True,'protocol_id':protocol_id(),'report_id':identifier,'bytes':size}


def main():
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    raw=sys.stdin.buffer.read(16*1024+1)
    if len(raw)>16*1024:raise ValueError('Own foundation request exceeds its bound')
    with contextlib.redirect_stdout(sys.stderr):result=run(json.loads(raw))
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=='__main__':main()
