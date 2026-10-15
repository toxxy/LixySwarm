"""Collect predeclared public TRAIN outcomes on one of two physical machines."""
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

from src.contribution import ResourceGovernor, ResourceRequirements
from src.learning.language_outcomes import definition, protocol_id, verify_outcomes
from src.learning.language_retention import PARENT_MODEL,PARENT_RELEASE
from src.learning.public_math import ARTIFACTS,load_snapshot
from src.network.artifact_store import ArtifactStore,digest_file
from src.network.client_state import ClientSettings,NodeStateLock
from src.network.compute_process import ComputeSupervisor
from src.network.language_install import materialize_version
from src.network.telemetry import atomic_json
from src.network.public_knowledge import managed_public_bytes
from src.release.language_versions import LanguageVersionRegistry
from .prepare_public_math import fingerprints,outside_checkout
from .research_artifact import retained_bytes


def run(args,cancel):
    node = args.node_directory.resolve(); directory = args.directory.resolve()
    repo = Path(__file__).resolve().parents[1]
    if (directory.exists() or not all(outside_checkout(p) for p in [node,directory,args.specialist_file])
            or not directory.is_relative_to(Path.home()/'.local/state/lixyswarm-outcomes-20261015')
            or not args.engine_python.is_absolute() or not args.engine_python.is_file()):
        raise ValueError('Use accounted fresh research state and locally selected engine outside Dropbox')
    if subprocess.check_output(['git','status','--porcelain'],cwd=repo):raise ValueError('Outcome source must be committed before compute')
    source = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    published = json.loads((repo/'experiments/reports/language_outcome_protocol_20261015.json').read_text())
    if published != {'protocol_id':protocol_id(),'definition':definition()}:raise ValueError('Public outcome protocol is not the predeclared definition')
    before = fingerprints(node); settings = ClientSettings.load(node); policy = asdict(settings.policy)
    initial = json.loads((node/'diagnostics.json').read_text()); pid = initial['pid']
    registry = LanguageVersionRegistry(node/'public/language','qwen3-1.7b')
    if registry.head()['release']['model_artifact_id'] != PARENT_MODEL:raise ValueError('Outcome parent is no longer current')
    governor = ResourceGovernor(settings.policy,hardware=SimpleNamespace(has_gpu=True),storage_path=node)
    def allowed():
        try:
            now = ClientSettings.load(node); state = json.loads((node/'diagnostics.json').read_text())
            checks = {'not_cancelled':not cancel.is_set(),'connected':now.connected,'not_paused':not now.paused,
                'same_policy':asdict(now.policy)==policy,'ram':now.policy.max_ram_gb>=10,
                'cpu':now.policy.cpu_percent>0,'gpu':now.policy.gpu_percent>0 and now.policy.max_gpu_jobs>0,
                'disk':managed_public_bytes(node)+retained_bytes()+.01*1024**3<=now.policy.max_disk_gb*1024**3,
                'same_process':state['pid']==pid,'fresh_diagnostics':time.time()*1000-state['timestamp_ms']<30000,
                'idle_inference':state['language']['active']==0,'idle_training':state['language_training']['active']==0,
                'training_ready':state['language_training']['ready'],'same_native_state':fingerprints(node)==before,
                'same_parent':registry.head()['release']['release_id']==PARENT_RELEASE,'system_available':not governor._system_busy()}
            permitted = all(checks.values())
            if not permitted and directory.exists():
                atomic_json(directory/'guard-failure.json',{'checks':checks,'timestamp':time.time()})
            return permitted
        except (OSError,ValueError,KeyError,TypeError):return False
    if not allowed():raise ValueError('Outcome calibration requires idle work within current physical resources')
    base = materialize_version(node,'qwen3-1.7b',PARENT_RELEASE,allowed=allowed)
    store = ArtifactStore(node/'public/language/artifacts',max_artifact_bytes=4*1024**3)
    directory.mkdir(mode=0o700,parents=True)
    atomic_json(directory/'invariants-before.json',before)
    deadline = time.time()+definition()['maximum_seconds']
    class Execution:
        def raise_if_cancelled(self):
            if cancel.is_set():raise RuntimeError('Outcome collection cancelled')
    execution = Execution();execution.deadline = deadline
    request = {'action':definition()['action'],'protocol_id':protocol_id(),'compute_dtype':'float32','streamed_base':True,
        'base_directory':str(base),'adapter_file':str(base/'adapter.safetensors'),'specialist_file':str(args.specialist_file.resolve()),
        'model_id':PARENT_MODEL,'parent_release_id':PARENT_RELEASE,'training_file':str(store.object_path(ARTIFACTS['train'][0])),
        'output_directory':str(directory/'engine'),'device':'cuda','cpu_threads':4,'gpu_fraction':policy['gpu_percent']/100/max(1,policy['max_gpu_jobs']),
        'deadline':deadline,'shard':args.shard}
    lease,reason = governor.acquire(ResourceRequirements(kind='training',gpu_required=True,ram_gb=10,disk_gb=.01))
    if lease is None:raise ValueError('Outcome resource reservation denied: '+reason)
    atomic_json(directory/'status.json',{'phase':'collecting','success':False})
    try:
        with lease,NodeStateLock(node/'research/public-math'):
            response = ComputeSupervisor().run([str(args.engine_python),'-m','src.learning.language_outcome_job'],
                request,execution,allowed=allowed,ram_gb=10)
        raw = (directory/'engine/diagnostic.json').read_bytes(); result = json.loads(raw)
        if (len(raw)>64*1024 or hashlib.sha256(raw).hexdigest()!=response['report_id'] or len(raw)!=response['bytes']
                or result['success'] is not True or result['protocol_id']!=protocol_id() or fingerprints(node)!=before):
            raise ValueError('Outcome evidence or physical controls changed')
        path = directory/'engine/outcomes.json'
        if digest_file(path)!=(result['outcome_artifact_id'],result['outcome_bytes']):raise ValueError('Outcome artifact bytes changed')
        verified = verify_outcomes(json.loads(path.read_text()),load_snapshot(request['training_file'],'train'))
        if any(result[k]!=v for k,v in verified.items()):raise ValueError('Outcome scores do not independently reconstruct')
        report = {'success':True,'source_commit':source,'physical_node_id':initial['node_id'],'physical_pid':pid,
            'diagnostic':result,'native_controls_unchanged':True,'extra_physical_workers':0,'native_learning_cycles_added':0}
        atomic_json(directory/'report.json',report);atomic_json(directory/'status.json',{'phase':'complete','success':True})
        return report
    except BaseException as error:
        atomic_json(directory/'status.json',{'phase':'failed','success':False,'error_type':type(error).__name__,'error':str(error)[:1024]})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node-directory',type=Path,required=True);parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--engine-python',type=Path,required=True);parser.add_argument('--specialist-file',type=Path,required=True)
    parser.add_argument('--shard',type=int,choices=[0,1],required=True)
    cancel = threading.Event()
    for signum in (signal.SIGTERM,signal.SIGINT):signal.signal(signum,lambda *_:cancel.set())
    print(json.dumps(run(parser.parse_args(),cancel),allow_nan=False),flush=True)


if __name__=='__main__':main()
