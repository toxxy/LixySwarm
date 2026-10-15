"""Compare a newly frozen eager CUDA worker with the installed math worker.

Public inputs only; gradients remain research artifacts, never contributions or
model updates. This does not establish clean installation or model quality.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import signal
import threading
import time
from types import SimpleNamespace

from src.contribution import ResourceGovernor,ResourceRequirements
from src.learning.language_policy import policy_id,dataset_id
from src.learning.language_retention import PARENT_MODEL
from src.learning.public_math import ARTIFACTS
from src.learning.public_quality import quality_cases,correct_answer
from src.network.artifact_store import ArtifactStore,digest_file
from src.network.client_state import ClientSettings,NodeStateLock
from src.network.compute_process import ComputeSupervisor
from src.network.public_knowledge import managed_public_bytes
from src.network.telemetry import atomic_json
from .prepare_public_math import fingerprints,outside_checkout
from .research_artifact import retained_bytes


def full_baseline(args,directory,shared,execution,allowed):
    from src.learning.engine_qualification import reference,qualify
    from src.learning.language_retention import PARENT_RELEASE
    from src.learning.public_math import gate_id,compact_evaluation
    baseline=reference(Path(__file__).resolve().parents[1]/'experiments/reports/language_math_native_scaling_20261015.json')
    store=ArtifactStore(args.node_directory/'public/language/artifacts',max_artifact_bytes=4*1024**3)
    request={k:v for k,v in shared.items() if k!='adapter_id'}
    request.update(action='public-math-baseline-v1',model_id=PARENT_MODEL,parent_release_id=PARENT_RELEASE,
        evaluation_file=str(store.object_path(ARTIFACTS['evaluation'][0])),output_directory=str(directory/'baseline'),gate_id=gate_id())
    response=ComputeSupervisor().run([str(args.engine)],request,execution,allowed=allowed,ram_gb=10)
    path=directory/'baseline/baseline.json';raw=path.read_bytes();result=json.loads(raw)
    if (len(raw)!=response['bytes'] or digest_file(path)[0]!=response['evaluation_id']
            or result['model_id']!=PARENT_MODEL or result['training_performed'] is not False or result['model_published'] is not False):
        raise ValueError('Frozen baseline response changed its read-only scope')
    compact=compact_evaluation({k:result[k] for k in ['nll','tokens','answers','correct','total','output_tokens','at_output_limit']})
    verdict=qualify(result['legacy'],compact,baseline)
    atomic_json(directory/'qualification.json',verdict)
    if not verdict['unchanged_public_outputs']:raise ValueError('Compact worker did not preserve the complete bound public baseline')
    return {'response':response,'qualification':verdict,'peak_gpu_allocated_bytes':result['peak_gpu_allocated_bytes'],
        'seconds':result['seconds'],'model_updated':False,'new_native_contributions':0,'clean_install_proven':False}


def run(args,cancel):
    node=args.node_directory.resolve();directory=args.directory.resolve()
    research=Path.home()/'.local/state/lixyswarm-clean-gpu-20261015'
    if (directory.exists() or not directory.is_relative_to(research) or not outside_checkout(directory)
            or not args.engine.is_absolute() or not args.engine.is_file()):
        raise ValueError('Use a fresh external report directory and locally selected frozen worker')
    before=fingerprints(node);settings=ClientSettings.load(node);policy=asdict(settings.policy)
    state=json.loads((node/'diagnostics.json').read_text());pid=state['pid']
    inference=json.loads((node/'language-engine.json').read_text())
    reference=json.loads((node/'language-training.json').read_text())['command']
    if inference['adapter_id']!=PARENT_MODEL or digest_file(inference['adapter_file'])[0]!=PARENT_MODEL:
        raise ValueError('Eager comparison requires the unchanged accepted generalist')
    governor=ResourceGovernor(settings.policy,hardware=SimpleNamespace(has_gpu=True),storage_path=node)
    def allowed():
        try:
            now=ClientSettings.load(node);live=json.loads((node/'diagnostics.json').read_text())
            own=sum(p.stat().st_size for p in research.rglob('*') if p.is_file()) if research.exists() else 0
            age=time.time()*1000-live['timestamp_ms']
            checks={'not_cancelled':not cancel.is_set(),'connected':now.connected,'not_paused':not now.paused,
                'policy_unchanged':asdict(now.policy)==policy,'ram':now.policy.max_ram_gb>=10,
                'gpu':now.policy.gpu_percent>0 and now.policy.max_gpu_jobs>0,'native_unchanged':fingerprints(node)==before,
                'same_process':live['pid']==pid,'fresh_diagnostics':age<30000,
                'idle_inference':live['language']['active']==0,'idle_training':live['language_training']['active']==0,
                'system_available':not governor._system_busy(),
                'disk':managed_public_bytes(node)+retained_bytes()+own+16*1024**2<=now.policy.max_disk_gb*1024**3}
            permitted=all(checks.values())
            if not permitted and directory.exists():atomic_json(directory/'guard-failure.json',{'checks':checks,'diagnostic_age_ms':age,'timestamp':time.time()})
            return permitted
        except (OSError,ValueError,TypeError,KeyError):return False
    if not allowed():raise ValueError('Eager verification requires idle, consented physical resources')
    directory.mkdir(mode=0o700,parents=True);atomic_json(directory/'invariants-before.json',before)
    deadline=time.time()+600
    class Execution:
        def raise_if_cancelled(self):
            if cancel.is_set():raise RuntimeError('Eager worker verification cancelled')
    execution=Execution();execution.deadline=deadline
    lease,reason=governor.acquire(ResourceRequirements(kind='training',gpu_required=True,ram_gb=10,disk_gb=.016))
    if lease is None:raise ValueError('Eager verification reservation denied: '+reason)
    shared={'base_name':'qwen3-1.7b','base_directory':inference['base_directory'],'adapter_file':inference['adapter_file'],
        'adapter_id':PARENT_MODEL,'device':'cuda','cpu_threads':4,'gpu_fraction':policy['gpu_percent']/100/max(1,policy['max_gpu_jobs']),
        'deadline':deadline}
    store=ArtifactStore(node/'public/language/artifacts',max_artifact_bytes=4*1024**3)
    outputs={};started=time.monotonic()
    with lease,NodeStateLock(node/'research/public-math'):
        if args.full_baseline:
            atomic_json(directory/'status.json',{'phase':'full-readonly-baseline','success':False})
            report=full_baseline(args,directory,shared,execution,allowed)
            if fingerprints(node)!=before:raise ValueError('Full baseline changed native state')
            report.update(success=True,native_controls_unchanged=True,physical_pid=pid)
            atomic_json(directory/'report.json',report);atomic_json(directory/'status.json',{'phase':'complete','success':True})
            return report
        for name,command in [('installed',reference),('eager',[str(args.engine)])]:
            atomic_json(directory/'status.json',{'phase':name+'-inference','success':False})
            request={**shared,'action':'inference','messages':[{'role':'user','content':quality_cases()[0]['prompt']}]}
            answer=ComputeSupervisor().run(command,request,execution,allowed=allowed,ram_gb=10)
            if answer['model_id']!=PARENT_MODEL or not correct_answer(quality_cases()[0],answer['answer']):
                raise ValueError('Frozen worker did not preserve the declared public answer')
            atomic_json(directory/'status.json',{'phase':name+'-math-gradient','success':False})
            request={**shared,'action':'gradient-math','training_policy_id':policy_id(4),'dataset_id':dataset_id(4),
                'training_file':str(store.object_path(ARTIFACTS['train'][0])),'start':128,'count':2,
                'output_file':str(directory/(name+'-gradient.safetensors'))}
            gradient=ComputeSupervisor().run(command,request,execution,allowed=allowed,ram_gb=10)
            if digest_file(request['output_file'])!=(gradient['gradient_id'],gradient['gradient_bytes']):
                raise ValueError('Frozen worker gradient differs from its report')
            outputs[name]={'inference':answer,'gradient':gradient}
            atomic_json(directory/'outputs.json',outputs)
        import torch
        from safetensors.torch import load_file
        a=load_file(directory/'installed-gradient.safetensors');b=load_file(directory/'eager-gradient.safetensors')
        if set(a)!=set(b) or len(a)!=112:raise ValueError('Frozen workers changed their gradient parameter schema')
        exact=sum(torch.equal(a[k],b[k]) for k in a)
        x=torch.cat([a[k].float().flatten() for k in sorted(a)]);y=torch.cat([b[k].float().flatten() for k in sorted(b)])
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():raise ValueError('Nonfinite worker gradient')
        relative=float((x-y).norm()/x.norm())
    if fingerprints(node)!=before:raise ValueError('Eager verification changed native controls or learning progress')
    report={'success':True,'physical_node_id':state['node_id'],'physical_pid':pid,'model_id':PARENT_MODEL,
        'outputs':outputs,'bitwise_equal_gradient_tensors':exact,'relative_gradient_l2':relative,
        'equal_inference':outputs['installed']['inference']==outputs['eager']['inference'],'native_controls_unchanged':True,
        'new_native_contributions':0,'model_updated':False,'private_inputs':False,'clean_install_proven':False,
        'seconds':time.monotonic()-started,'frozen_worker_qualified':False,
        'scope':'Gradient-repeatability diagnostic; catalog qualification requires --full-baseline, never exact BF16 backward bits'}
    atomic_json(directory/'report.json',report);atomic_json(directory/'status.json',{'phase':'complete','success':True})
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node-directory',type=Path,required=True);parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--engine',type=Path,required=True)
    parser.add_argument('--full-baseline',action='store_true',help='Read-only qualification against all bound public outputs, no gradients or updates')
    cancel=threading.Event()
    for signum in (signal.SIGINT,signal.SIGTERM):signal.signal(signum,lambda *_:cancel.set())
    print(json.dumps(run(parser.parse_args(),cancel),allow_nan=False),flush=True)


if __name__=='__main__':main()
