"""Supervise public response-utility learning without changing the native node."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from types import SimpleNamespace

from desktop.backend import NodeController
from src.contribution import ResourceGovernor,ResourceRequirements
from src.learning.language_response_router import definition,protocol_id
from src.learning.language_retention import PARENT_MODEL,PARENT_RELEASE
from src.learning.public_math import ARTIFACTS
from src.network.artifact_store import ArtifactStore,digest_file
from src.network.client_state import ClientSettings,NodeStateLock
from src.network.compute_process import ComputeSupervisor
from src.network.public_knowledge import managed_public_bytes
from src.network.telemetry import atomic_json
from src.release.language_versions import LanguageVersionRegistry
from .collect_repair_calibration import freeze_source
from .prepare_public_math import fingerprints,outside_checkout
from .research_artifact import retained_bytes


def run(args,cancel):
    p=definition();node=args.node_directory.resolve();directory=args.directory.resolve()
    source=Path(__file__).resolve().parents[1]
    if (directory.exists() or not directory.is_relative_to(Path.home()/'.local/state/lixyswarm-response-router-20261015')
            or not all(outside_checkout(x) for x in [node,directory,*args.outcome_file,*args.repair_file])
            or not args.engine_python.is_absolute() or not args.engine_python.is_file()):
        raise ValueError('Utility fit needs fresh accounted state and explicit local public inputs')
    if subprocess.check_output(['git','status','--porcelain'],cwd=source):raise ValueError('Publish the complete response utility program before learning')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    if json.loads((source/'experiments/reports/response_router_protocol_20261015.json').read_text())!={'protocol_id':protocol_id(),'definition':p}:
        raise ValueError('The response utility protocol differs from its publication')
    before=fingerprints(node);settings=ClientSettings.load(node);policy=asdict(settings.policy)
    controller=NodeController(node);initial=controller.snapshot();pid=initial['diagnostics']['pid']
    registry=LanguageVersionRegistry(node/'public/language','qwen3-1.7b')
    engine=json.loads((node/'language-engine.json').read_text())
    def allowed():
        try:
            now=ClientSettings.load(node);live=controller.snapshot()
            return (not cancel.is_set() and now.connected and not now.paused and asdict(now.policy)==policy
                and fingerprints(node)==before and live['fresh'] and live['diagnostics']['pid']==pid
                and not any(live['diagnostics'].get(k,{}).get('active',0) for k in ['compute','language','language_training'])
                and registry.head()['release']['release_id']==PARENT_RELEASE
                and managed_public_bytes(node)+retained_bytes()<=now.policy.max_disk_gb*1024**3)
        except (OSError,ValueError,KeyError,TypeError):return False
    if (not allowed() or engine['adapter_id']!=PARENT_MODEL
            or managed_public_bytes(node)+retained_bytes()+p['disk_reservation_gb']*1024**3>policy['max_disk_gb']*1024**3):
        raise ValueError('Response utility fit is outside current native permissions, idle state, parent or quota')
    governor=ResourceGovernor(settings.policy,hardware=SimpleNamespace(has_gpu=True),storage_path=node)
    lease,reason=governor.acquire(ResourceRequirements(kind='training',gpu_required=True,ram_gb=p['ram_gb'],disk_gb=p['disk_reservation_gb']))
    if lease is None:raise ValueError('Response utility resource reservation denied: '+reason)
    directory.mkdir(mode=0o700,parents=True);atomic_json(directory/'invariants-before.json',before)
    frozen,archive_id,archive_bytes=freeze_source(source,directory,revision,p['disk_reservation_gb'])
    store=ArtifactStore(node/'public/language/artifacts',max_artifact_bytes=4*1024**3)
    deadline=time.time()+p['maximum_seconds']
    request={'action':p['action'],'protocol_id':protocol_id(),'base_directory':engine['base_directory'],
        'adapter_file':engine['adapter_file'],'model_id':PARENT_MODEL,'parent_release_id':PARENT_RELEASE,
        'training_file':str(store.object_path(ARTIFACTS['train'][0])),
        'outcome_files':[str(v.resolve()) for v in args.outcome_file],'repair_files':[str(v.resolve()) for v in args.repair_file],
        'output_directory':str(directory/'engine'),'device':'cuda','cpu_threads':4,
        'gpu_fraction':policy['gpu_percent']/100/max(1,policy['max_gpu_jobs']),'deadline':deadline}
    def check():
        if cancel.is_set() or time.time()>=deadline:raise RuntimeError('Response utility fit cancelled or expired')
    execution=SimpleNamespace(deadline=deadline,raise_if_cancelled=check)
    atomic_json(directory/'status.json',{'phase':'fit-public-response-utility','success':False})
    prior=Path.cwd()
    try:
        with lease,NodeStateLock(node/'research/public-math'):
            os.chdir(frozen)
            try:
                response=ComputeSupervisor().run([str(args.engine_python),'-m','src.learning.language_response_router_job'],
                    request,execution,allowed=allowed,ram_gb=p['ram_gb'])
            finally:os.chdir(prior)
        atomic_json(directory/'engine-receipt.json',response)
        path=directory/'engine/diagnostic.json'
        if digest_file(path)!=(response['report_id'],response['bytes']) or not allowed():raise ValueError('Response utility evidence or native invariant changed')
        report=json.loads(path.read_text())
        if report['protocol_id']!=protocol_id() or report['fresh_evaluation_performed'] is not False or report['model_promoted'] is not False:
            raise ValueError('Response utility fit exceeded its TRAIN-only scope')
        for name,item in report['artifacts'].items():
            if Path(name).name!=name or digest_file(directory/'engine'/name)!=(item['sha256'],item['bytes']):
                raise ValueError('Learned response artifact integrity changed')
        result={'success':True,'protocol_id':protocol_id(),'source_revision':revision,
            'source_archive_id':archive_id,'source_archive_bytes':archive_bytes,
            'physical_node_id':initial['diagnostics']['node_id'],'physical_pid':pid,
            'report_id':response['report_id'],'report_bytes':response['bytes'],
            'learned_state_bytes':sum(v['bytes'] for v in report['artifacts'].values()),
            'fit':report['fit'],'seconds':report['seconds'],'peak_gpu_allocated_bytes':report['peak_gpu_allocated_bytes'],
            'fresh_evaluation_performed':False,'useful_routing_proven':False,
            'native_controls_unchanged':True,'native_nodes_restarted':False,'native_cycles_added':0,'model_promoted':False}
        atomic_json(directory/'report.json',result);atomic_json(directory/'status.json',{'phase':'complete','success':True})
        return result
    except BaseException as error:
        atomic_json(directory/'status.json',{'phase':'failed','success':False,'error_type':type(error).__name__,'error':str(error)[:1024]});raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['directory','node-directory','engine-python']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--outcome-file',type=Path,action='append',required=True)
    parser.add_argument('--repair-file',type=Path,action='append',required=True)
    args=parser.parse_args();new=not args.directory.exists();cancel=threading.Event()
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:cancel.set())
    try:result=run(args,cancel)
    except BaseException as error:
        if new and args.directory.is_dir():atomic_json(args.directory/'status.json',{'phase':'failed','success':False,
            'error_type':type(error).__name__,'error':str(error)[:1024]})
        raise
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=='__main__':main()
