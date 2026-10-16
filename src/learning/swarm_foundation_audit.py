"""Recover own learned weights and optimizer in a separate CPU process."""
import argparse
import contextlib
import json
from pathlib import Path
import sys

from src.network.artifact_store import digest_file
from src.network.telemetry import atomic_json
from .swarm_foundation_protocol import definition,protocol_id
from .public_math import load_snapshot


def audit(args):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    from .swarm_foundation import make_foundation_model,predict
    from .swarm_foundation_job import evaluate,documents,causal_check
    p=definition();torch.set_num_threads(2)
    if digest_file(args.report)[0]!=args.report_id:
        raise ValueError('Own foundation source report changed')
    expected=json.loads(args.report.read_text())
    if expected['protocol_id']!=protocol_id() or expected['pretrained_weights_loaded'] is not False:
        raise ValueError('Own foundation report belongs to another lineage')
    for path,name in [(args.weights,'model.safetensors'),(args.optimizer,'optimizer.pt')]:
        record=expected['artifacts'][name]
        if path.is_symlink() or digest_file(path)!=(record['sha256'],record['bytes']) or record['bytes']>16*1024**2:
            raise ValueError('Own foundation received state differs from the selected public artifact')
    with safe_open(str(args.weights),framework='pt',device='cpu') as f:
        if f.metadata()!={'format':'lixyswarm-own-causal-byte-v1','protocol_id':protocol_id()}:
            raise ValueError('Own weights have incompatible protocol metadata')
    initial=make_foundation_model(seed=p['initialization_seed'])
    heldout=documents(load_snapshot(args.evaluation_file,'evaluation'))
    before=evaluate(initial,heldout,'cpu')
    state=load_file(str(args.weights));initial.load_state_dict(state,strict=True)
    after=evaluate(initial,heldout,'cpu');causality=causal_check(initial,'cpu')
    for label,observed in [('before',before),('after',after)]:
        if (observed['predicted_bytes']!=expected[label]['predicted_bytes']
                or observed['documents']!=expected[label]['documents']
                or abs(observed['nll']-expected[label]['nll'])>1e-5
                or observed['byte_accuracy']!=expected[label]['byte_accuracy']):
            raise ValueError('CPU held-out evaluation does not reproduce the learned model')
    saved=torch.load(args.optimizer,map_location='cpu',weights_only=True)
    if saved['protocol_id']!=protocol_id() or saved['step']!=p['steps']:
        raise ValueError('Own optimizer belongs to another training history')
    optimizer=torch.optim.AdamW(initial.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'])
    optimizer.load_state_dict(saved['optimizer'])
    states=optimizer.state_dict()['state']
    if len(states)!=expected['optimizer_states'] or {int(v['step']) for v in states.values()}!={p['steps']}:
        raise ValueError('Own optimizer training steps are missing')
    parameters=[parameter for group in optimizer.param_groups for parameter in group['params']]
    for index,item in states.items():
        for name in ['exp_avg','exp_avg_sq']:
            moment=item[name]
            if moment.shape!=parameters[index].shape or not torch.isfinite(moment).all():
                raise ValueError('Own optimizer moments are incomplete or nonfinite')
    # Continue one TRAIN-only step in memory, twice from the same recovered
    # state. Nothing is saved as a new model or fed back into native services.
    training=documents(load_snapshot(args.training_file,'train'))
    stream=torch.tensor(list(b''.join(training)),dtype=torch.uint8)
    rng=torch.Generator().set_state(saved['sampler_state'])
    starts=torch.randint(0,len(stream)-p['block_size'],(p['batch_size'],),generator=rng)
    batch=torch.stack([stream[int(v):int(v)+p['block_size']+1] for v in starts]).long()
    resumed=[]
    for model in [initial,make_foundation_model(seed=p['initialization_seed'])]:
        model.load_state_dict(state,strict=True);model.train()
        opt=torch.optim.AdamW(model.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'])
        # Each branch must own its moments; optimizer.load_state_dict may alias
        # already-CPU tensors, so reload the immutable file for every branch.
        opt.load_state_dict(torch.load(args.optimizer,map_location='cpu',weights_only=True)['optimizer'])
        opt.zero_grad(set_to_none=True)
        _,loss,_=predict(model,batch[:,:-1],batch[:,1:]);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),p['gradient_clip_norm'],error_if_nonfinite=True)
        opt.step()
        if {int(v['step']) for v in opt.state_dict()['state'].values()}!={p['steps']+1}:
            raise ValueError('Own optimizer did not resume the next real step')
        resumed.append({k:v.detach().clone() for k,v in model.state_dict().items()})
    if not all(torch.equal(v,resumed[1][k]) for k,v in resumed[0].items()):
        raise ValueError('Recovered own optimizer continuation is not reproducible')
    changed=sum(not torch.equal(state[k],v) for k,v in resumed[0].items())
    if not changed or not causality['passed']:
        raise ValueError('Own learned state did not support causal continued learning')
    for path,name in [(args.weights,'model.safetensors'),(args.optimizer,'optimizer.pt')]:
        if digest_file(path)[0]!=expected['artifacts'][name]['sha256']:
            raise ValueError('Own model input was modified by its audit')
    return {'success':True,'protocol_id':protocol_id(),'before':before,'after':after,
        'causality':causality,'optimizer_states':len(states),'saved_optimizer_step':p['steps'],
        'resumed_optimizer_step':p['steps']+1,'identical_resumed_branches':True,
        'changed_state_tensors_in_memory':changed,'continuation_steps_in_memory':2,
        'saved_weights_unchanged':True,'saved_optimizer_unchanged':True,
        'device':'cpu','cuda_initialized':torch.cuda.is_initialized(),
        'new_model_saved':False,'model_promoted':False,'native_cycles_added':0,
        'distributed_training_performed':False,'general_chat_quality_proven':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['weights','optimizer','report','training-file','evaluation-file','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--report-id',required=True);args=parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(Path.home()/'Dropbox'):
        raise ValueError('Use fresh CPU audit evidence outside the checkout')
    with contextlib.redirect_stdout(sys.stderr):value=audit(args)
    atomic_json(args.output,value);print(json.dumps(value,allow_nan=False),flush=True)


if __name__=='__main__':main()
