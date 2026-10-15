"""Bound a prospective comparison without disrupting the native participant."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import signal
import subprocess
import threading
import time
from types import SimpleNamespace

from desktop.backend import NodeController
from src.contribution import ResourceGovernor, ResourceRequirements
from src.learning.language_repair_comparison import (definition,protocol_id,build_snapshot,
    load_snapshot,verify_report,SNAPSHOT_ID,SNAPSHOT_BYTES)
from src.learning.language_retention import PARENT_MODEL,PARENT_RELEASE
from src.learning.public_math import canonical
from src.network.artifact_store import ArtifactStore,digest_file
from src.network.client_state import ClientSettings,NodeStateLock
from src.network.compute_process import ComputeSupervisor
from src.network.public_knowledge import managed_public_bytes
from src.network.telemetry import atomic_json
from src.release.language_versions import LanguageVersionRegistry
from .prepare_public_math import fingerprints,outside_checkout
from .research_artifact import retained_bytes


def run(args,cancel):
    p=definition();directory=args.directory.resolve();node=args.node_directory.resolve();source=Path(__file__).resolve().parents[1]
    if (directory.exists() or not directory.is_relative_to(Path.home()/'.local/state/lixyswarm-format-recovery-20261015')
            or not all(outside_checkout(path) for path in [directory,node,args.source_directory,args.specialist_file])
            or not args.engine_python.is_absolute() or not args.engine_python.is_file()):
        raise ValueError('Use fresh accounted comparison state and exact local public inputs')
    if subprocess.check_output(['git','status','--porcelain'],cwd=source):
        raise ValueError('Commit the complete prospective comparison before fresh generation')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    if json.loads((source/'experiments/reports/format_recovery_protocol_20261015.json').read_text())!={'protocol_id':protocol_id(),'definition':p}:
        raise ValueError('The exact prospective comparison is not published')
    numerical={str(path.relative_to(source)):digest_file(path)[0] for path in (source/'src/learning').glob('*.py')}
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
    if not allowed() or managed_public_bytes(node)+retained_bytes()+p['disk_reservation_gb']*1024**3>policy['max_disk_gb']*1024**3:
        raise ValueError('Comparison is outside current native permission, idle state or quota')
    governor=ResourceGovernor(settings.policy,hardware=SimpleNamespace(has_gpu=True),storage_path=node)
    lease,reason=governor.acquire(ResourceRequirements(kind='inference',gpu_required=True,ram_gb=p['ram_gb'],disk_gb=p['disk_reservation_gb']))
    if lease is None:raise ValueError('Comparison reservation denied: '+reason)
    with lease,NodeStateLock(node/'research/public-math'):
        directory.mkdir(parents=True,mode=0o700)
        raw=canonical(build_snapshot(args.source_directory))
        import hashlib
        if (hashlib.sha256(raw).hexdigest(),len(raw))!=(SNAPSHOT_ID,SNAPSHOT_BYTES):
            raise ValueError('Derived prospective dataset differs from its declared pin')
        snapshot=directory/'prospective.json';snapshot.write_bytes(raw)
        store=ArtifactStore(node/'public/language/artifacts',max_artifact_bytes=4*1024**3)
        manifest=store.import_file(snapshot,kind='dataset',media_type='application/json')
        assert (manifest.artifact_id,manifest.size)==(SNAPSHOT_ID,SNAPSHOT_BYTES)
        request={'action':p['action'],'protocol_id':protocol_id(),'compute_dtype':'float32','streamed_base':True,
            'base_directory':engine['base_directory'],'adapter_file':engine['adapter_file'],'model_id':PARENT_MODEL,
            'parent_release_id':PARENT_RELEASE,'snapshot_file':str(snapshot),'specialist_file':str(args.specialist_file),
            'output_directory':str(directory/'engine'),'device':'cuda','cpu_threads':4,
            'gpu_fraction':policy['gpu_percent']/100/max(1,policy['max_gpu_jobs']),'deadline':time.time()+p['maximum_seconds']}
        def check():
            if cancel.is_set() or time.time()>=request['deadline']:raise RuntimeError('Comparison cancelled or expired')
        execution=SimpleNamespace(deadline=request['deadline'],raise_if_cancelled=check)
        atomic_json(directory/'status.json',{'phase':'prospective-format-comparison','success':False,'source_revision':revision})
        try:
            response=ComputeSupervisor().run([str(args.engine_python),'-m','src.learning.language_repair_comparison_job'],
                request,execution,allowed=allowed,ram_gb=p['ram_gb'])
            path=directory/'engine/diagnostic.json';report=json.loads(path.read_text())
            if (digest_file(path)!=(response['report_id'],response['bytes']) or not allowed()
                    or any(digest_file(source/name)[0]!=value for name,value in numerical.items())):
                raise ValueError('Comparison evidence or native/source invariants changed')
            verification=verify_report(report,load_snapshot(snapshot))
            if verification!=report['verification']:raise ValueError('Comparison metrics do not reconstruct')
            result={'success':True,'protocol_id':protocol_id(),'source_revision':revision,'numerical_source_files':numerical,
                'physical_node_id':initial['status']['node_id'],'physical_pid':pid,'report_id':response['report_id'],
                'report_bytes':response['bytes'],'snapshot_id':SNAPSHOT_ID,'verification':verification,
                'seconds':report['seconds'],'peak_gpu_allocated_bytes':report['peak_gpu_allocated_bytes'],
                'native_controls_unchanged':True,'native_cycles_added':0,'model_promoted':False,
                'neural_routing_evaluated':False,'scope':p['scope']}
            atomic_json(directory/'report.json',result);atomic_json(directory/'status.json',{'phase':'complete','success':True});return result
        except BaseException as error:
            atomic_json(directory/'status.json',{'phase':'failed','success':False,'error_type':type(error).__name__,'error':str(error)[:1024]});raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['directory','node-directory','engine-python','source-directory','specialist-file']:
        parser.add_argument('--'+name,type=Path,required=True)
    cancel=threading.Event()
    for signum in [signal.SIGTERM,signal.SIGINT]:signal.signal(signum,lambda *_:cancel.set())
    print(json.dumps(run(parser.parse_args(),cancel),allow_nan=False),flush=True)


if __name__=='__main__':main()
