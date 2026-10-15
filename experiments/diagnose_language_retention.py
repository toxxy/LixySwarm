"""Diagnose one saved public update under current physical-node consent."""
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
from src.learning.language_retention import SOURCE_REPORT,PARENT_MODEL,PARENT_RELEASE,definition,protocol_id
from src.learning.language_scaling import verify_timed_run
from src.network.artifact_store import ArtifactStore
from src.network.client_state import ClientSettings,NodeStateLock
from src.network.compute_process import ComputeSupervisor
from src.network.language_install import materialize_version
from src.network.public_knowledge import managed_public_bytes
from src.network.telemetry import atomic_json
from src.release.language_versions import LanguageVersionRegistry
from .prepare_public_math import fingerprints,outside_checkout


def selected_trial(repo):
    path=Path(repo)/'experiments/reports/language_math_native_scaling_20261015.json'
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=SOURCE_REPORT:
        raise ValueError('Retention diagnosis requires the exact saved native evidence')
    data=json.loads(path.read_text())
    matches=[r['report'] for r in data['trials'] if r['declaration']['label']==definition()['source_trial']]
    if len(matches)!=1:raise ValueError('The fixed retrospective trial is missing')
    trial=matches[0];verify_timed_run(trial)
    if trial['head']['release']['release_id']!=PARENT_RELEASE:raise ValueError('Saved retention trial has another parent')
    return trial


def run(args,cancel):
    node=args.node_directory.resolve();directory=args.directory.resolve();repo=Path(__file__).resolve().parents[1]
    if directory.exists() or not all(outside_checkout(p) for p in [node,directory]):
        raise ValueError('Use existing physical controls and fresh research state outside Dropbox')
    if not args.engine_python.is_absolute() or not args.engine_python.is_file():raise ValueError('Use a locally selected existing tensor runtime')
    if subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip():
        raise ValueError('Commit the diagnostic source and protocol before physical compute')
    source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    finite=getattr(args,'finite_step',False)
    if finite:
        from src.learning.language_finite_protocol import definition as finite_definition,protocol_id as finite_id
        protocol=finite_definition(args.precision);identifier=finite_id(args.precision)
    else:protocol=definition(args.precision);identifier=protocol_id(args.precision)
    ram_gb=protocol['ram_gb']
    trial=selected_trial(repo);before=fingerprints(node);controls=ClientSettings.load(node);policy=asdict(controls.policy)
    diagnostic=json.loads((node/'diagnostics.json').read_text());pid=diagnostic['pid']
    registry=LanguageVersionRegistry(node/'public/language','qwen3-1.7b');head=registry.head()
    if head['release']['release_id']!=PARENT_RELEASE or head['release']['model_artifact_id']!=PARENT_MODEL:
        raise ValueError('The retention diagnostic parent is no longer current')
    store=ArtifactStore(node/'public/language/artifacts',max_artifact_bytes=4*1024**3,max_total_bytes=8*1024**3)
    governor=ResourceGovernor(controls.policy,hardware=SimpleNamespace(has_gpu=True),storage_path=node)
    def allowed():
        try:
            now=ClientSettings.load(node);state=json.loads((node/'diagnostics.json').read_text())
            return (not cancel.is_set() and now.connected and not now.paused and asdict(now.policy)==policy
                and now.policy.mode in {'balanced','maximum'} and now.policy.max_ram_gb>=ram_gb
                and now.policy.cpu_percent>0 and now.policy.gpu_percent>0 and now.policy.max_gpu_jobs>0
                and managed_public_bytes(node)+8*1024**2<=now.policy.max_disk_gb*1024**3
                and state['pid']==pid and time.time()*1000-state['timestamp_ms']<30000
                and state['language_training']['ready'] and state['language_training'].get('math_curriculum_enabled')
                and state['language']['active']==0 and state['language_training']['active']==0
                and fingerprints(node)==before and registry.head()['release']['release_id']==PARENT_RELEASE
                and not governor._system_busy())
        except (OSError,ValueError,TypeError,KeyError):return False
    if not allowed():raise ValueError('Retention diagnosis needs idle GPU work within its declared RAM reservation')
    from src.learning.public_math import ARTIFACTS
    training=store.object_path(ARTIFACTS['train'][0])
    base=materialize_version(node,'qwen3-1.7b',PARENT_RELEASE,allowed=allowed)
    rows=[]
    for result in trial['signed_results']:
        output=result['output'];row={key:output[key] for key in
            ['gradient_id','gradient_bytes','base_hash','adapter_id','dataset_id','start','count','tokens','quantization_relative_l2_error']}
        row['path']=str(store.object_path(row['gradient_id']));rows.append(row)
    directory.mkdir(mode=0o700,parents=True)
    atomic_json(directory/'invariants-before.json',before)
    atomic_json(directory/'protocol.json',{'protocol_id':identifier,'definition':protocol})
    deadline=time.time()+900
    class Execution:
        def raise_if_cancelled(self):
            if cancel.is_set():raise RuntimeError('Retention research cancelled')
    execution=Execution();execution.deadline=deadline
    request={'action':protocol['action'],'protocol_id':identifier,'compute_dtype':args.precision,
        'base_directory':str(base),'adapter_file':str(base/'adapter.safetensors'),'model_id':PARENT_MODEL,
        'parent_release_id':PARENT_RELEASE,'training_file':str(training),'contributions':rows,
        'output_directory':str(directory/'engine'),'device':'cuda','cpu_threads':min(4,max(1,int(4*policy['cpu_percent']/100))),
        'gpu_fraction':policy['gpu_percent']/100/max(1,policy['max_gpu_jobs']),'deadline':deadline}
    lease,reason=governor.acquire(ResourceRequirements(kind='training',gpu_required=True,ram_gb=ram_gb,disk_gb=.008))
    if lease is None:raise ValueError('Retention research reservation denied: '+reason)
    try:
        with lease,NodeStateLock(node/'research/public-math'):
            atomic_json(directory/'status.json',{'phase':'diagnosing','success':False})
            module='src.learning.language_finite_step_job' if finite else 'src.learning.language_retention_job'
            response=ComputeSupervisor().run([str(args.engine_python.absolute()),'-m',module],
                request,execution,allowed=allowed,ram_gb=ram_gb)
        path=directory/'engine/diagnostic.json'
        if path.is_symlink() or path.stat().st_size>256*1024:raise ValueError('Retention diagnostic output exceeds its bound')
        raw=path.read_bytes();result=json.loads(raw)
        if (hashlib.sha256(raw).hexdigest()!=response['report_id'] or len(raw)!=response['bytes']
                or result['protocol_id']!=identifier or result['model_id']!=PARENT_MODEL or not result['success']
                or result['held_out_data_used_in_compute'] is not False or result['updated_model_files_written'] is not False
                or result['parent_parameters_restored'] is not True or fingerprints(node)!=before):
            raise ValueError('Retention diagnostic changed its scope or current native controls')
        if (subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()!=source_revision
                or subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()):
            raise ValueError('Diagnostic source changed during physical compute; preserve the saved result for review')
        report={'success':True,'physical_node_id':diagnostic['node_id'],'native_daemon_job':False,'research_path':True,
            'protocol_id':identifier,'definition':protocol,'source_revision':source_revision,
            'engine_report_sha256':response['report_id'],'diagnostic':result,'controls_and_native_progress_unchanged':True,
            'native_learning_cycles_added':0,'new_model_published':False,'source_daemon_pid_unchanged':True,
            'resource_reservation':{'ram_gb':ram_gb,'gpu_fraction':request['gpu_fraction']},'finished_at':time.time()}
        atomic_json(directory/'report.json',report);atomic_json(directory/'status.json',{'phase':'complete','success':True})
        summary={'success':True,'applied_in_temporary_model':result['guard']['applied'],
            'accepted_rate':result['guard']['accepted_rate']} if finite else {
            'success':True,'loss_changes':result['loss_changes'],'reconstruction':result['reconstruction']}
        print(json.dumps(summary),flush=True)
    except BaseException as error:
        atomic_json(directory/'status.json',{'phase':'failed','success':False,'error_type':type(error).__name__,
            'controls_and_native_progress_unchanged':fingerprints(node)==before})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node-directory',type=Path,required=True);parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--engine-python',type=Path,required=True)
    parser.add_argument('--precision',choices=['float32','bfloat16'],default='float32')
    parser.add_argument('--finite-step',action='store_true',help='Run the prospective TRAIN-only backtracking protocol')
    args=parser.parse_args();cancel=threading.Event()
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,lambda *_:cancel.set())
    try:run(args,cancel)
    except BaseException as error:
        if args.directory.is_dir():atomic_json(args.directory/'status.json',
            {'phase':'failed','success':False,'error_type':type(error).__name__})
        raise


if __name__=='__main__':main()
