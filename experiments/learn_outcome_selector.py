"""Supervise a TRAIN-only selector comparison under current desktop consent."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import threading
import time
from types import SimpleNamespace

from src.contribution import ResourceGovernor,ResourceRequirements
from src.learning.language_outcome_selector import definition,protocol_id,dataset,verify_report
from src.learning.public_math import ARTIFACTS,load_snapshot
from src.network.artifact_store import ArtifactStore,digest_file
from src.network.client_state import ClientSettings,NodeStateLock
from src.network.compute_process import ComputeSupervisor
from src.network.public_knowledge import managed_public_bytes
from src.network.telemetry import atomic_json
from .prepare_public_math import fingerprints,outside_checkout
from .research_artifact import retained_bytes


def run(args,cancel):
    directory=args.directory.resolve();node=args.node_directory.resolve();p=definition()
    if (directory.exists() or not directory.is_relative_to(Path.home()/'.local/state/lixyswarm-outcome-selector-20261015')
            or not all(outside_checkout(x) for x in [directory,node,*args.outcome_file])
            or len(args.outcome_file)!=2 or not args.engine_python.is_absolute() or not args.engine_python.is_file()):
        raise ValueError('Use a fresh accounted directory, two frozen outcome artifacts and an existing local tensor environment')
    settings=ClientSettings.load(node);policy=asdict(settings.policy);before=fingerprints(node)
    initial=json.loads((node/'diagnostics.json').read_text());pid=initial['pid']
    governor=ResourceGovernor(settings.policy,hardware=SimpleNamespace(has_gpu=True),storage_path=node)
    def allowed():
        current=ClientSettings.load(node);diagnostic=json.loads((node/'diagnostics.json').read_text())
        return (not cancel.is_set() and current.connected and not current.paused and asdict(current.policy)==policy
            and fingerprints(node)==before and diagnostic['pid']==pid and time.time()*1000-diagnostic['timestamp_ms']<60000
            and not any(diagnostic[name].get('active',0) for name in ['compute','language','language_training'])
            and managed_public_bytes(node)+retained_bytes()<=current.policy.max_disk_gb*1024**3)
    if (not allowed() or managed_public_bytes(node)+retained_bytes()+p['disk_reservation_gb']*1024**3>policy['max_disk_gb']*1024**3):
        raise ValueError('Selector learned state does not fit current physical permission')
    source=Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip():
        raise ValueError('Commit the prospective protocol and numerical implementation before learning')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    numerical={name:digest_file(source/name)[0] for name in ['src/learning/language_selector.py',
        'src/learning/language_outcome_selector.py','src/learning/language_outcome_selector_job.py',
        'src/agents/dolphin_agent.py','src/matriarca/matriarca.py']}
    store=ArtifactStore(node/'public/language/artifacts',max_artifact_bytes=4*1024**3)
    settings_language=json.loads((node/'language-engine.json').read_text());base=Path(settings_language['base_directory'])
    training=store.object_path(ARTIFACTS['train'][0])
    # Verify physical evidence before allocating the requested resources.
    outcomes=[json.loads(path.read_text()) for path in args.outcome_file];rows=dataset(outcomes,load_snapshot(training,'train'))
    lease,reason=governor.acquire(ResourceRequirements(kind='training',gpu_required=True,ram_gb=p['ram_gb'],disk_gb=p['disk_reservation_gb']))
    if lease is None:raise ValueError('Outcome selector compute unavailable: '+reason)
    directory.mkdir(mode=0o700,parents=True);deadline=time.time()+p['maximum_seconds']
    request={'action':p['action'],'protocol_id':protocol_id(),'training_file':str(training),
        'outcome_files':[str(x.resolve()) for x in args.outcome_file],'base_directory':str(base),
        'output_directory':str(directory/'engine'),'device':'cuda','cpu_threads':4,
        'gpu_fraction':policy['gpu_percent']/100/max(1,policy['max_gpu_jobs']),'deadline':deadline}
    execution=SimpleNamespace(work=SimpleNamespace(requirements=SimpleNamespace(gpu_required=True)),cancel_event=cancel,deadline=deadline)
    def raise_if_cancelled():
        if cancel.is_set() or time.time()>=deadline:raise RuntimeError('Outcome selector stopped')
    execution.raise_if_cancelled=raise_if_cancelled
    atomic_json(directory/'status.json',{'phase':'learning','success':False})
    try:
        with lease,NodeStateLock(node/'research/public-math'):
            response=ComputeSupervisor().run([str(args.engine_python),'-m','src.learning.language_outcome_selector_job'],
                request,execution,allowed=allowed,ram_gb=p['ram_gb'])
        path=directory/'engine/diagnostic.json';raw=path.read_bytes();report=json.loads(raw)
        if (digest_file(path)!=(response['report_id'],response['bytes']) or len(raw)>1024**2 or fingerprints(node)!=before
                or any(digest_file(source/name)[0]!=value for name,value in numerical.items())):
            raise ValueError('Outcome selector result or physical native progress changed')
        verification=verify_report(report,rows)
        if report['verification']!=verification:raise ValueError('Outcome selector metrics do not reconstruct')
        for name,artifact in report['artifacts'].items():
            if digest_file(directory/'engine'/name)!=(artifact['sha256'],artifact['bytes']):raise ValueError('A learned selector artifact changed')
        result={'success':True,'protocol_id':protocol_id(),'source_revision':revision,'numerical_source_files':numerical,
            'physical_node_id':initial['node_id'],'physical_pid':pid,
            'report_id':response['report_id'],'report_bytes':response['bytes'],'verification':verification,
            'native_controls_unchanged':True,'native_learning_cycles_added':0,'model_promoted':False,
            'original_evaluation_opened':False,'seconds':report['seconds'],'peak_gpu_allocated_bytes':report['peak_gpu_allocated_bytes'],
            'learned_state_bytes':sum(a['bytes'] for a in report['artifacts'].values())}
        atomic_json(directory/'report.json',result);atomic_json(directory/'status.json',{'phase':'complete','success':True});return result
    except BaseException as error:
        atomic_json(directory/'status.json',{'phase':'failed','success':False,'error_type':type(error).__name__,'error':str(error)[:1024]});raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['directory','node-directory','engine-python']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--outcome-file',type=Path,action='append',required=True)
    cancel=threading.Event()
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:cancel.set())
    print(json.dumps(run(parser.parse_args(),cancel),allow_nan=False),flush=True)


if __name__=='__main__':main()
