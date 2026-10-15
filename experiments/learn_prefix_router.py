"""Supervise prefix routing under unchanged native permission and source."""
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
from src.learning.language_prefix_router import definition,protocol_id,verify_report
from src.learning.language_retention import PARENT_MODEL,PARENT_RELEASE
from src.learning.public_math import ARTIFACTS,load_snapshot
from src.network.artifact_store import ArtifactStore,digest_file
from src.network.client_state import ClientSettings,NodeStateLock
from src.network.compute_process import ComputeSupervisor
from src.network.public_knowledge import managed_public_bytes
from src.network.telemetry import atomic_json
from src.release.language_versions import LanguageVersionRegistry
from .prepare_public_math import fingerprints,outside_checkout
from .research_artifact import retained_bytes

NUMERICAL_FILES=['src/learning/language_prefix_router.py','src/learning/language_prefix_router_job.py',
    'src/learning/language_dense_selector.py','src/learning/language_streamed_float.py',
    'src/learning/language_adapter.py','src/learning/language_specialist.py','src/matriarca/matriarca.py',
    'src/learning/language_outcome_selector.py','src/learning/language_outcomes.py',
    'src/learning/language_selector.py','src/learning/public_math.py']


def run(args,cancel):
    p=definition();directory=args.directory.resolve();node=args.node_directory.resolve();source=Path(__file__).resolve().parents[1]
    if (directory.exists() or not directory.is_relative_to(Path.home()/'.local/state/lixyswarm-prefix-router-20261015')
            or not all(outside_checkout(x) for x in [directory,node,args.specialist_file,*args.outcome_file])
            or len(args.outcome_file)!=2 or not args.engine_python.is_absolute() or not args.engine_python.is_file()):
        raise ValueError('Use fresh accounted prefix state and exact locally selected public inputs')
    if subprocess.check_output(['git','status','--porcelain'],cwd=source):raise ValueError('Commit the complete prospective prefix protocol and code before learning')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    published=json.loads((source/'experiments/reports/prefix_router_protocol_20261015.json').read_text())
    if published!={'protocol_id':protocol_id(),'definition':p}:raise ValueError('Prospective prefix protocol is not published exactly')
    numerical={name:digest_file(source/name)[0] for name in NUMERICAL_FILES}
    before=fingerprints(node);settings=ClientSettings.load(node);policy=asdict(settings.policy)
    initial=json.loads((node/'diagnostics.json').read_text());pid=initial['pid']
    registry=LanguageVersionRegistry(node/'public/language','qwen3-1.7b')
    engine=json.loads((node/'language-engine.json').read_text());store=ArtifactStore(node/'public/language/artifacts',max_artifact_bytes=4*1024**3)
    training=store.object_path(ARTIFACTS['train'][0])
    governor=ResourceGovernor(settings.policy,hardware=SimpleNamespace(has_gpu=True),storage_path=node)
    def allowed():
        try:
            now=ClientSettings.load(node);live=json.loads((node/'diagnostics.json').read_text())
            return (not cancel.is_set() and now.connected and not now.paused and asdict(now.policy)==policy
                and fingerprints(node)==before and live['pid']==pid and time.time()*1000-live['timestamp_ms']<60000
                and not any(live.get(k,{}).get('active',0) for k in ['compute','language','language_training'])
                and registry.head()['release']['release_id']==PARENT_RELEASE
                and managed_public_bytes(node)+retained_bytes()<=now.policy.max_disk_gb*1024**3)
        except (OSError,ValueError,KeyError,TypeError):return False
    if not allowed() or managed_public_bytes(node)+retained_bytes()+p['disk_reservation_gb']*1024**3>policy['max_disk_gb']*1024**3:
        raise ValueError('Prefix comparison is outside current native permission, idle state or storage quota')
    lease,reason=governor.acquire(ResourceRequirements(kind='training',gpu_required=True,ram_gb=p['ram_gb'],disk_gb=p['disk_reservation_gb']))
    if lease is None:raise ValueError('Prefix compute reservation denied: '+reason)
    directory.mkdir(parents=True,mode=0o700);deadline=time.time()+p['maximum_seconds']
    request={'action':p['action'],'protocol_id':protocol_id(),'compute_dtype':'float32','streamed_base':True,
        'base_directory':engine['base_directory'],'adapter_file':engine['adapter_file'],'model_id':PARENT_MODEL,
        'parent_release_id':PARENT_RELEASE,'training_file':str(training),'specialist_file':str(args.specialist_file.resolve()),
        'outcome_files':[str(v.resolve()) for v in args.outcome_file],'output_directory':str(directory/'engine'),
        'device':'cuda','cpu_threads':4,'gpu_fraction':policy['gpu_percent']/100/max(1,policy['max_gpu_jobs']),'deadline':deadline}
    def check():
        if cancel.is_set() or time.time()>=deadline:raise RuntimeError('Prefix comparison cancelled or expired')
    execution=SimpleNamespace(deadline=deadline,raise_if_cancelled=check)
    atomic_json(directory/'status.json',{'phase':'learning-prefix-router','success':False})
    try:
        with lease,NodeStateLock(node/'research/public-math'):
            response=ComputeSupervisor().run([str(args.engine_python),'-m','src.learning.language_prefix_router_job'],
                request,execution,allowed=allowed,ram_gb=p['ram_gb'])
        path=directory/'engine/diagnostic.json';raw=path.read_bytes();report=json.loads(raw)
        if (len(raw)>2*1024**2 or hashlib.sha256(raw).hexdigest()!=response['report_id'] or len(raw)!=response['bytes']
                or not allowed() or any(digest_file(source/name)[0]!=value for name,value in numerical.items())):
            raise ValueError('Prefix result source or native invariants changed')
        verification=verify_report(report,load_snapshot(training,'train'))
        if verification!=report['verification']:raise ValueError('Prefix comparison metrics do not reconstruct')
        for name,item in report['artifacts'].items():
            if Path(name).name!=name or digest_file(directory/'engine'/name)!=(item['sha256'],item['bytes']):
                raise ValueError('Prefix learned artifact integrity changed')
        result={'success':True,'protocol_id':protocol_id(),'source_revision':revision,'numerical_source_files':numerical,
            'physical_node_id':initial['node_id'],'physical_pid':pid,'report_id':response['report_id'],'report_bytes':response['bytes'],
            'verification':verification,'fit':report['fit'],'learned_state_bytes':sum(v['bytes'] for v in report['artifacts'].values()),
            'seconds':report['seconds'],'peak_gpu_allocated_bytes':report['peak_gpu_allocated_bytes'],
            'native_controls_unchanged':True,'native_cycles_added':0,'model_promoted':False,
            'original_evaluation_opened':False,'scope':p['scope']}
        atomic_json(directory/'report.json',result);atomic_json(directory/'status.json',{'phase':'complete','success':True});return result
    except BaseException as error:
        atomic_json(directory/'status.json',{'phase':'failed','success':False,'error_type':type(error).__name__,'error':str(error)[:1024]});raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['directory','node-directory','engine-python','specialist-file']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--outcome-file',type=Path,action='append',required=True);cancel=threading.Event()
    for signum in [signal.SIGTERM,signal.SIGINT]:signal.signal(signum,lambda *_:cancel.set())
    print(json.dumps(run(parser.parse_args(),cancel),allow_nan=False),flush=True)


if __name__=='__main__':main()
