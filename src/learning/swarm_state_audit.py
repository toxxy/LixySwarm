"""Independent CPU replay of a signed, physically distributed own-state update."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import time

from src.network.artifact_store import ArtifactStore,digest_file
from src.network.own_state import verify_transition
from src.network.telemetry import atomic_json
from .swarm_policy import policy_id,definition
from .swarm_foundation_protocol import protocol_id as initialization_protocol_id


def replay_request(directory,candidate,output_directory,*,deadline,cpu_threads=2):
    """Materialize only public dependencies; no identity or private state reaches the child."""
    directory=Path(directory).resolve()
    store=ArtifactStore(directory/'public/own/artifacts',
        max_artifact_bytes=16*1024**2,max_total_bytes=1024**3)
    result=candidate['output'];evaluation_path=store.object_path(result['evaluation_id'])
    identifier=result['training_policy_id'];p=definition(identifier)
    transition=verify_transition(candidate,evaluation_path.read_bytes())
    evaluation=json.loads(evaluation_path.read_text())
    return {'action':'own-audit','candidate_result':candidate,
        'candidate_model_file':str(store.object_path(transition['model_id'])),
        'candidate_optimizer_file':str(store.object_path(transition['optimizer_id'])),
        'evaluation_report_file':str(evaluation_path),
        'training_policy_id':identifier,'state_id':result['state_id'],
        'model_id':result['model_id'],'optimizer_id':result['optimizer_id'],
        'step':result['previous_optimizer_step'],'dataset_id':p['training_dataset_id'],
        'model_file':str(store.object_path(result['model_id'])),
        'optimizer_file':str(store.object_path(result['optimizer_id'])),
        'training_file':str(directory/'public/language/artifacts/objects'/definition()['training_dataset_id'][:2]/definition()['training_dataset_id'][2:]),
        'evaluation_file':str(directory/'public/language/artifacts/objects'/definition()['evaluation_dataset_id'][:2]/definition()['evaluation_dataset_id'][2:]),
        'output_directory':str(output_directory),'device':'cpu','cpu_threads':cpu_threads,
        'deadline':deadline,'requester_id':transition['evaluator_id'],
        'contributions':evaluation['contributions'],
        'gradient_files':{v['result']['output']['gradient_id']:str(store.object_path(v['result']['output']['gradient_id']))
            for v in evaluation['contributions']}}


def run_audit(request):
    import torch
    from safetensors.torch import load_file
    from .swarm_distributed_job import run
    candidate=request['candidate_result'];result=candidate['output']
    transition=verify_transition(candidate,Path(request['evaluation_report_file']).read_bytes())
    for key in ['candidate_model','candidate_optimizer']:
        if digest_file(Path(request[key+'_file']))!=(result[key+'_id'],result[key+'_bytes']):
            raise ValueError('Recovered own state differs from its signed candidate')
    # Replay the exact received gradients and parent optimizer. Validation
    # labels never update parameters; this operation computes no new gradient.
    replay_request_value={k:v for k,v in request.items() if k not in {
        'candidate_result','candidate_model_file','candidate_optimizer_file','evaluation_report_file'}}
    replay_request_value['action']='own-candidate'
    replay=run(replay_request_value)
    if not replay['accepted'] or replay['candidate_optimizer_step']!=transition['step']:
        raise ValueError('Independent own update did not reproduce its accepted step')
    # CPU kernels may differ between machines. These tolerances are the same
    # bounds used by the previously published causal checks, fixed before replay.
    expected=load_file(request['candidate_model_file'])
    observed=load_file(replay['candidate_model_file'])
    if set(expected)!=set(observed):raise ValueError('Own replay changed the architecture')
    maximum=0.;exact=True
    for name,value in expected.items():
        if not torch.isfinite(value).all() or not torch.allclose(value,observed[name],atol=1e-6,rtol=1e-5):
            raise ValueError('Own replay does not reconstruct the received learned tensors')
        maximum=max(maximum,float((value-observed[name]).abs().max()));exact &= torch.equal(value,observed[name])
    saved=torch.load(request['candidate_optimizer_file'],map_location='cpu',weights_only=True)
    recomputed=torch.load(replay['candidate_optimizer_file'],map_location='cpu',weights_only=True)
    if (saved['protocol_id']!=initialization_protocol_id() or saved['step']!=transition['step']
            or saved['own_curriculum']!=recomputed['own_curriculum']
            or not torch.equal(saved['sampler_state'],recomputed['sampler_state'])):
        raise ValueError('Own replay changed its optimizer or curriculum history')
    states=saved['optimizer']['state'];other=recomputed['optimizer']['state']
    if states.keys()!=other.keys():raise ValueError('Own replay lost optimizer moments')
    for key,item in states.items():
        if int(item['step'])!=transition['step']:raise ValueError('Own optimizer step was reset')
        for name in ['exp_avg','exp_avg_sq']:
            if not torch.isfinite(item[name]).all() or not torch.allclose(item[name],other[key][name],atol=1e-6,rtol=1e-5):
                raise ValueError('Own optimizer moments do not reproduce both contributions')
    for metric in ['nll','byte_accuracy']:
        if abs(replay['after'][metric]-result['after'][metric])>1e-5:
            raise ValueError('Own replay does not reproduce validation quality')
    for key in ['candidate_model','candidate_optimizer']:
        if digest_file(Path(request[key+'_file']))!=(result[key+'_id'],result[key+'_bytes']):
            raise ValueError('Own state input was modified by independent verification')
    report={'success':True,'state_id':transition['state_id'],'model_id':transition['model_id'],
        'optimizer_id':transition['optimizer_id'],'training_policy_id':request['training_policy_id'],
        'gradient_replay_verified':True,'model_tensors_bitwise_equal':bool(exact),'maximum_model_tensor_error':maximum,
        'optimizer_states':len(states),'optimizer_step':saved['step'],'after':replay['after'],
        'saved_state_unchanged':True,'new_gradient_computed':False,'cuda_initialized':torch.cuda.is_initialized(),
        'model_promoted':False,'scope':'Independent replay of existing signed contributions on CPU'}
    return report


def audit(args):
    if (args.directory.exists() or args.directory.resolve().is_relative_to(Path.home()/'Dropbox')
            or args.directory.resolve().is_relative_to(Path(__file__).resolve().parents[2])):
        raise ValueError('Independent own replay needs fresh external evidence')
    args.directory.mkdir(parents=True,mode=0o700)
    request=replay_request(args.node_directory,json.loads(args.receipt.read_text()),args.directory/'engine',
        deadline=time.time()+definition()['maximum_seconds'])
    report=run_audit(request);atomic_json(args.directory/'audit.json',report);return report


def run_request(request):
    with contextlib.redirect_stdout(sys.stderr):report=run_audit(request)
    print(json.dumps(report,allow_nan=False),flush=True)


def main():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['node-directory','receipt','directory']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    with contextlib.redirect_stdout(sys.stderr):result=audit(args)
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=='__main__':main()
