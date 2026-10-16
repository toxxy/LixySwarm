"""Typed own-swarm tensor work. No identity, listener, downloads or private data."""
import contextlib
import json
import math
from pathlib import Path
import sys
import time

from .swarm_policy import (definition,policy_id,state_id,validate_payload,
                               verify_contributions,verdict)
from .swarm_foundation_protocol import protocol_id as initialization_protocol_id
from .swarm_foundation_job import documents,evaluate
from .public_math import canonical,load_snapshot


def binding(request):
    return {k:request[k] for k in ['training_policy_id','state_id','model_id','dataset_id','start','count']}


def batches(docs,device):
    import torch
    p=definition()
    for start in range(0,len(docs),p['batch_size']):
        rows=docs[start:start+p['batch_size']]
        data=torch.zeros((len(rows),129),dtype=torch.long,device=device)
        labels=torch.full((len(rows),128),-1,dtype=torch.long,device=device)
        for index,row in enumerate(rows):
            row=row[:p['maximum_document_bytes']]
            data[index,:len(row)]=torch.tensor(list(row),device=device)
            labels[index,:len(row)-1]=data[index,1:len(row)]
        yield data[:,:-1],labels,int((labels!=-1).sum())


def load_model(request):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    from src.network.artifact_store import digest_file
    from .swarm_foundation import make_foundation_model
    path=Path(request['model_file'])
    if digest_file(path)[0]!=request['model_id'] or path.stat().st_size>definition()['maximum_artifact_bytes']:
        raise ValueError('Own model input differs from its selected content')
    with safe_open(str(path),framework='pt',device='cpu') as f:
        if f.metadata()!={'format':'lixyswarm-own-causal-byte-v1','protocol_id':initialization_protocol_id()}:
            raise ValueError('Own model has incompatible initializer provenance')
    values=load_file(str(path))
    if not all(torch.isfinite(v).all() for v in values.values()):
        raise ValueError('Own model contains nonfinite state')
    model=make_foundation_model(device=request['device'])
    model.load_state_dict(values,strict=True)
    return model


def run(request):
    p=definition(request.get('training_policy_id',''));candidate=request.get('action')=='own-candidate'
    common={'action','training_policy_id','state_id','model_id','optimizer_id','step','dataset_id',
            'model_file','training_file','output_directory','device','cpu_threads','deadline'}
    fields=common|({'optimizer_file','evaluation_file','requester_id','contributions','gradient_files'}
                    if candidate else {'start','count'})
    if (set(request)!=fields or request['action'] not in {'own-gradient','own-candidate'}
            or state_id(request['model_id'],request['optimizer_id'],request['step'])!=request['state_id']
            or request['device']!='cpu' or type(request['cpu_threads']) is not int or not 1<=request['cpu_threads']<=16
            or type(request['deadline']) not in (float,int) or not math.isfinite(request['deadline'])
            or not time.time()<request['deadline']<=time.time()+p['maximum_seconds']):
        raise ValueError('Own-swarm child requires locally selected typed inputs and CPU limits')
    validate_payload({k:request[k] for k in ['training_policy_id','state_id','model_id','dataset_id',
        *(['contributions'] if candidate else ['start','count'])]},candidate=candidate)
    paths=[request[k] for k in ['model_file','training_file','output_directory']]
    if candidate:
        if not isinstance(request['gradient_files'],dict) or len(request['gradient_files'])!=len(request['contributions']):
            raise ValueError('Own candidate needs every locally acquired gradient file')
        paths += [request['optimizer_file'],request['evaluation_file'],*request['gradient_files'].values()]
    output=Path(request['output_directory'])
    if (any(not isinstance(v,str) or not Path(v).is_absolute() or Path(v).is_symlink() for v in paths)
            or output.exists() or output.resolve().is_relative_to(Path.home()/'Dropbox')
            or output.resolve().is_relative_to(Path(__file__).resolve().parents[2])):
        raise ValueError('Own compute inputs and fresh output must be explicit and outside the code checkout')
    output.mkdir(mode=0o700,parents=True)
    try:
        import torch
        torch.set_num_threads(request['cpu_threads']);torch.set_float32_matmul_precision('highest')
        model=load_model(request)
        snapshot=load_snapshot(request['training_file'],'train')
        def check():
            if time.time()>=request['deadline']:raise TimeoutError('Own work deadline exceeded')
        return (_candidate(request,output,model,snapshot,check) if candidate
                else _gradient(request,output,model,snapshot,check))
    except BaseException as error:
        from .language_repair_comparison_job import durable_json
        import traceback
        durable_json(output/'failure.json',{'error_type':type(error).__name__,'error':str(error)[:2048],
            'traceback':traceback.format_exc(limit=10)[-12*1024:]})
        raise


def _gradient(request,output,model,snapshot,check):
    import torch
    from safetensors.torch import save_file
    from src.network.artifact_store import digest_file
    from .swarm_foundation import predict
    rows=snapshot['rows'][request['start']:request['start']+request['count']]
    docs=documents({'rows':rows});total=sum(min(len(row),129)-1 for row in docs)
    model.train();model.zero_grad(set_to_none=True);loss_sum=0.
    for inputs,labels,tokens in batches(docs,'cpu'):
        check();_,loss,_=predict(model,inputs,labels)
        if not torch.isfinite(loss):raise ValueError('Own gradient loss is not finite')
        (loss*tokens/total).backward();loss_sum+=float(loss.detach())*tokens/total
    gradients={k:p.grad.detach().float().cpu().contiguous() for k,p in model.named_parameters() if p.grad is not None}
    if not gradients or not all(torch.isfinite(v).all() for v in gradients.values()):
        raise ValueError('Own gradient is incomplete or not finite')
    path=output/'gradient.safetensors'
    save_file(gradients,str(path),metadata={'format':'lixyswarm-own-gradient-v1',
        'binding':canonical({**binding(request),'tokens':total}).decode()})
    identifier,size=digest_file(path)
    if size>definition()['maximum_artifact_bytes']:raise ValueError('Own gradient exceeds its bound')
    return {**binding(request),'tokens':total,'loss':loss_sum,'gradient_id':identifier,
            'gradient_bytes':size,'gradient_file':str(path)}


def _candidate(request,output,model,snapshot,check):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file,save_file
    from src.network.artifact_store import digest_file
    from .language_repair_comparison_job import durable_json
    p=definition(request['training_policy_id']);check()
    contributions=verify_contributions(request['contributions'],model_id=request['model_id'],
        training_state_id=request['state_id'],requester_id=request['requester_id'],training_policy_id=request['training_policy_id'])
    if set(request['gradient_files'])!={v['gradient_id'] for v in contributions}:
        raise ValueError('Own acquired gradient set differs from signed contributions')
    if digest_file(request['optimizer_file'])[0]!=request['optimizer_id']:
        raise ValueError('Own optimizer differs from the selected training state')
    saved=torch.load(request['optimizer_file'],map_location='cpu',weights_only=True)
    if saved['protocol_id']!=initialization_protocol_id() or saved['step']!=request['step']:
        raise ValueError('Own optimizer belongs to another training history')
    optimizer=torch.optim.AdamW(model.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'])
    optimizer.load_state_dict(saved['optimizer'])
    named=list(model.named_parameters());states=optimizer.state_dict()['state']
    names={named[index][0] for index in states}
    if not states or {int(v['step']) for v in states.values()}!={request['step']}:
        raise ValueError('Own optimizer state was reset or lost')
    for group in optimizer.param_groups:group.update(lr=p['learning_rate'],weight_decay=p['weight_decay'])
    parameters=dict(named);optimizer.zero_grad(set_to_none=True)
    total=sum(v['tokens'] for v in contributions)
    for record in contributions:
        check();path=Path(request['gradient_files'][record['gradient_id']])
        if digest_file(path)!=(record['gradient_id'],record['gradient_bytes']):
            raise ValueError('Own gradient content does not match its physical receipt')
        selected=snapshot['rows'][record['start']:record['start']+record['count']]
        tokens=sum(min(len(row),129)-1 for row in documents({'rows':selected}))
        expected={k:record[k] for k in ['training_policy_id','state_id','model_id','dataset_id','start','count','tokens']}
        with safe_open(str(path),framework='pt',device='cpu') as f:
            if f.metadata()!={'format':'lixyswarm-own-gradient-v1','binding':canonical(expected).decode()}:
                raise ValueError('Own gradient tensor binding differs from its receipt')
        gradients=load_file(str(path))
        if set(gradients)!=names or tokens!=record['tokens']:
            raise ValueError('Own gradient parameter set or prediction-byte count is incomplete')
        for name,value in gradients.items():
            if value.dtype!=torch.float32 or value.shape!=parameters[name].shape or not torch.isfinite(value).all():
                raise ValueError('Own gradient tensor shape or values are invalid')
            term=value*(record['tokens']/total)
            if parameters[name].grad is None:parameters[name].grad=term
            else:parameters[name].grad.add_(term)
    heldout=documents(load_snapshot(request['evaluation_file'],'evaluation'))
    check();before=evaluate(model,heldout,'cpu')
    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),p['clip_norm'],error_if_nonfinite=True))
    optimizer.step();check();after=evaluate(model,heldout,'cpu');accepted=verdict(before,after)
    step=request['step']+1
    if {int(v['step']) for v in optimizer.state_dict()['state'].values()}!={step}:
        raise ValueError('Own optimizer did not continue its previous step')
    report={'accepted':accepted,'training_policy_id':request['training_policy_id'],'state_id':request['state_id'],
        'model_id':request['model_id'],'optimizer_id':request['optimizer_id'],'before':before,'after':after,
        'previous_optimizer_step':request['step'],'candidate_optimizer_step':step,
        'gradient_norm_before_clipping':norm,'prediction_bytes':total,'contributions':request['contributions'],
        'model_promoted':False,'native_cycles_added':0,'scope':p['scope']}
    response={k:report[k] for k in ['accepted','training_policy_id','state_id','model_id',
        'optimizer_id','before','after','previous_optimizer_step','candidate_optimizer_step','prediction_bytes']}
    if accepted:
        weights=output/'model.safetensors';opt=output/'optimizer.pt'
        save_file({k:v.detach().cpu().clone().contiguous() for k,v in model.state_dict().items()},str(weights),
            metadata={'format':'lixyswarm-own-causal-byte-v1','protocol_id':initialization_protocol_id()})
        torch.save({**saved,'step':step,'optimizer':optimizer.state_dict(),
            'own_curriculum':{'policy_id':request['training_policy_id'],'previous_state_id':request['state_id'],
                'shards':[[v['start'],v['count']] for v in contributions]}},opt)
        for name,path in [('candidate_model',weights),('candidate_optimizer',opt)]:
            identifier,size=digest_file(path)
            if size>p['maximum_artifact_bytes']:raise ValueError('Own candidate state exceeds its bound')
            response.update({name+'_id':identifier,name+'_bytes':size,name+'_file':str(path)})
        report['candidate_state_id']=state_id(response['candidate_model_id'],response['candidate_optimizer_id'],step)
    durable_json(output/'evaluation.json',report)
    identifier,size=digest_file(output/'evaluation.json')
    response.update(evaluation_id=identifier,evaluation_bytes=size,evaluation_file=str(output/'evaluation.json'))
    return response


def run_request(request):
    with contextlib.redirect_stdout(sys.stderr):result=run(request)
    print(json.dumps(result,allow_nan=False),flush=True)
