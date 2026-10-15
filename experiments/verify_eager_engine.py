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


def run(args,cancel):
    node=args.node_directory.resolve();directory=args.directory.resolve()
    if directory.exists() or not outside_checkout(directory) or not args.engine.is_absolute() or not args.engine.is_file():
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
            own=sum(p.stat().st_size for p in directory.rglob('*') if p.is_file()) if directory.exists() else 0
            return (not cancel.is_set() and now.connected and not now.paused and asdict(now.policy)==policy
                and now.policy.max_ram_gb>=10 and now.policy.gpu_percent>0 and now.policy.max_gpu_jobs>0
                and fingerprints(node)==before and live['pid']==pid and time.time()*1000-live['timestamp_ms']<30000
                and live['language']['active']==0 and live['language_training']['active']==0 and not governor._system_busy()
                and managed_public_bytes(node)+retained_bytes()+own+16*1024**2<=now.policy.max_disk_gb*1024**3)
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
        import torch
        from safetensors.torch import load_file
        a=load_file(directory/'installed-gradient.safetensors');b=load_file(directory/'eager-gradient.safetensors')
        if (set(a)!=set(b) or len(a)!=112 or any(not torch.equal(a[k],b[k]) for k in a)
                or outputs['installed']['inference']!=outputs['eager']['inference']):
            raise ValueError('Removing the compiler changed eager inference or gradient tensors')
        for key in ['tokens','replay_tokens','loss','retention_loss']:
            if outputs['installed']['gradient'][key]!=outputs['eager']['gradient'][key]:
                raise ValueError('Frozen math supervision changed its counts or losses')
    if fingerprints(node)!=before:raise ValueError('Eager verification changed native controls or learning progress')
    report={'success':True,'physical_node_id':state['node_id'],'physical_pid':pid,'model_id':PARENT_MODEL,
        'outputs':outputs,'bitwise_equal_gradient_tensors':112,'equal_inference':True,'native_controls_unchanged':True,
        'new_native_contributions':0,'model_updated':False,'private_inputs':False,'clean_install_proven':False,
        'seconds':time.monotonic()-started,'scope':'One public answer and a fixed public math/replay gradient, packaging compatibility only'}
    atomic_json(directory/'report.json',report);atomic_json(directory/'status.json',{'phase':'complete','success':True})
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node-directory',type=Path,required=True);parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--engine',type=Path,required=True)
    cancel=threading.Event()
    for signum in (signal.SIGINT,signal.SIGTERM):signal.signal(signum,lambda *_:cancel.set())
    print(json.dumps(run(parser.parse_args(),cancel),allow_nan=False),flush=True)


if __name__=='__main__':main()
