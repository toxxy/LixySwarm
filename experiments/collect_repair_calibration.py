"""Supervise public TRAIN repairs on an existing physical PC with frozen code."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tarfile
import threading
import time
from types import SimpleNamespace

from src.contribution import ResourceGovernor,ResourceRequirements
from src.learning.language_repair_calibration import definition,protocol_id,load_outcomes,verify_calibration
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


def run(args,cancel):
    p=definition();node=args.node_directory.resolve();directory=args.directory.resolve()
    source=Path(__file__).resolve().parents[1]
    if (directory.exists() or not directory.is_relative_to(Path.home()/'.local/state/lixyswarm-repair-calibration-20261015')
            or not all(outside_checkout(x) for x in [node,directory,*args.outcome_file])
            or not args.engine_python.is_absolute() or not args.engine_python.is_file()):
        raise ValueError('Repair calibration needs fresh accounted external state and explicit local inputs')
    if subprocess.check_output(['git','status','--porcelain'],cwd=source):
        raise ValueError('Publish the complete repair calibration protocol before compute')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    published=json.loads((source/'experiments/reports/repair_calibration_protocol_20261015.json').read_text())
    if published!={'protocol_id':protocol_id(),'definition':p}:raise ValueError('Repair calibration protocol differs from its publication')
    before=fingerprints(node);settings=ClientSettings.load(node);policy=asdict(settings.policy)
    initial=json.loads((node/'diagnostics.json').read_text());pid=initial['pid']
    registry=LanguageVersionRegistry(node/'public/language','qwen3-1.7b')
    engine=json.loads((node/'language-engine.json').read_text())
    store=ArtifactStore(node/'public/language/artifacts',max_artifact_bytes=4*1024**3)
    training=store.object_path(ARTIFACTS['train'][0]);snapshot=load_snapshot(training,'train')
    cached=load_outcomes([str(path.resolve()) for path in args.outcome_file],snapshot)
    governor=ResourceGovernor(settings.policy,hardware=SimpleNamespace(has_gpu=True),storage_path=node)
    def allowed():
        try:
            now=ClientSettings.load(node);live=json.loads((node/'diagnostics.json').read_text())
            return (not cancel.is_set() and now.connected and not now.paused and asdict(now.policy)==policy
                and fingerprints(node)==before and live['pid']==pid and time.time()*1000-live['timestamp_ms']<60000
                and live.get('language_training',{}).get('ready') is True
                and not any(live.get(k,{}).get('active',0) for k in ['compute','language','language_training'])
                and registry.head()['release']['release_id']==PARENT_RELEASE
                and managed_public_bytes(node)+retained_bytes()<=now.policy.max_disk_gb*1024**3
                and not governor._system_busy())
        except (OSError,ValueError,KeyError,TypeError):return False
    if (engine['adapter_id']!=PARENT_MODEL or not allowed()
            or managed_public_bytes(node)+retained_bytes()+p['disk_reservation_gb']*1024**3>policy['max_disk_gb']*1024**3):
        raise ValueError('Repair calibration is outside current native model, consent, idle state or storage quota')
    lease,reason=governor.acquire(ResourceRequirements(kind='training',gpu_required=True,
        ram_gb=p['ram_gb'],disk_gb=p['disk_reservation_gb']))
    if lease is None:raise ValueError('Repair calibration resource reservation denied: '+reason)
    directory.mkdir(mode=0o700,parents=True)
    atomic_json(directory/'invariants-before.json',before)
    archive=directory/'source.tar'
    subprocess.run(['git','archive','--format=tar','--output',str(archive),revision],cwd=source,check=True)
    frozen=directory/'source';frozen.mkdir(mode=0o700)
    with tarfile.open(archive) as content:
        if (archive.stat().st_size+sum(v.size for v in content.getmembers() if v.isfile())+2*1024**2
                >p['disk_reservation_gb']*1024**3):
            raise ValueError('Frozen source and evidence exceed their declared scratch reservation')
        content.extractall(frozen,filter='data')
    archive_id,archive_bytes=digest_file(archive)
    atomic_json(directory/'source.json',{'revision':revision,'archive_sha256':archive_id,'archive_bytes':archive_bytes})
    deadline=time.time()+p['maximum_seconds']
    request={'action':p['action'],'protocol_id':protocol_id(),'compute_dtype':'float32','streamed_base':True,
        'base_directory':engine['base_directory'],'adapter_file':engine['adapter_file'],'model_id':PARENT_MODEL,
        'parent_release_id':PARENT_RELEASE,'training_file':str(training),
        'outcome_files':[str(v.resolve()) for v in args.outcome_file],'shard':args.shard,
        'output_directory':str(directory/'engine'),'device':'cuda','cpu_threads':4,
        'gpu_fraction':policy['gpu_percent']/100/max(1,policy['max_gpu_jobs']),'deadline':deadline}
    def check():
        if cancel.is_set() or time.time()>=deadline:raise RuntimeError('Repair calibration cancelled or expired')
    execution=SimpleNamespace(deadline=deadline,raise_if_cancelled=check)
    atomic_json(directory/'status.json',{'phase':'collecting-train-repairs','success':False})
    prior=Path.cwd()
    try:
        with lease,NodeStateLock(node/'research/public-math'):
            os.chdir(frozen)
            try:
                response=ComputeSupervisor().run([str(args.engine_python),'-m','src.learning.language_repair_calibration_job'],
                    request,execution,allowed=allowed,ram_gb=p['ram_gb'])
            finally:os.chdir(prior)
        atomic_json(directory/'engine-receipt.json',response)
        path=directory/'engine/diagnostic.json';raw=path.read_bytes();report=json.loads(raw)
        if (len(raw)>64*1024 or hashlib.sha256(raw).hexdigest()!=response['report_id'] or len(raw)!=response['bytes']
                or report['protocol_id']!=protocol_id() or not allowed()):
            raise ValueError('Repair calibration receipt or native invariants changed')
        artifact=directory/'engine/calibration.json'
        if digest_file(artifact)!=(report['artifact_id'],report['artifact_bytes']):
            raise ValueError('Repair calibration artifact integrity changed')
        verified=verify_calibration(json.loads(artifact.read_text()),snapshot,cached)
        if verified!=report['verification']:raise ValueError('Repair calibration labels do not independently reconstruct')
        result={'success':True,'protocol_id':protocol_id(),'source_revision':revision,
            'source_archive_id':archive_id,'source_archive_bytes':archive_bytes,
            'physical_node_id':initial['node_id'],'physical_pid':pid,'shard':args.shard,
            'report_id':response['report_id'],'report_bytes':response['bytes'],'diagnostic':report,
            'native_controls_unchanged':True,'native_cycles_added':0,'model_promoted':False,
            'original_evaluation_opened':False,'scope':p['scope']}
        atomic_json(directory/'report.json',result);atomic_json(directory/'status.json',{'phase':'complete','success':True})
        return result
    except BaseException as error:
        atomic_json(directory/'status.json',{'phase':'failed','success':False,'error_type':type(error).__name__,'error':str(error)[:1024]})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['directory','node-directory','engine-python']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--outcome-file',type=Path,action='append',required=True)
    parser.add_argument('--shard',type=int,choices=[0,1],required=True)
    cancel=threading.Event()
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:cancel.set())
    print(json.dumps(run(parser.parse_args(),cancel),allow_nan=False),flush=True)


if __name__=='__main__':main()
