"""Qualify a verified frozen CPU engine inside the optional Linux boundary.

Runs only the engine's synthetic self-test. No network identity, existing
weights, production model, learned memory or signing key is mounted in it.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import signal
import subprocess
import threading
import time
from types import SimpleNamespace

from src.contribution import ResourceGovernor, ResourceRequirements
from src.network.client_state import ClientSettings, NodeStateLock
from src.network.compute_process import ComputeSupervisor
from src.network.compute_sandbox import CPUComputeSandbox
from src.network.engine_package import verify_package
from src.network.public_knowledge import managed_public_bytes
from src.network.telemetry import atomic_json
from .prepare_public_math import fingerprints, outside_checkout
from .research_artifact import retained_bytes


def run(args):
    directory=args.directory.resolve();node=args.node_directory.resolve();package=args.package.resolve()
    if (directory.exists() or not all(outside_checkout(p) for p in (directory,node,package))
            or directory.is_relative_to(node) or node.is_relative_to(directory)
            or directory.is_relative_to(package) or package.is_relative_to(directory)):
        raise ValueError('Use a fresh separate local fixture directory and existing verified package')
    source=Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip():
        raise ValueError('Commit the prospective sandbox implementation before qualification')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    manifest=verify_package(package,args.manifest_sha256,profile='reference-cpu')
    executable=package/manifest['entrypoint']
    settings=ClientSettings.load(node);policy=asdict(settings.policy);before=fingerprints(node)
    initial=json.loads((node/'diagnostics.json').read_text());pid=initial['pid']
    cancelled=threading.Event()
    for signum in (signal.SIGINT,signal.SIGTERM):signal.signal(signum,lambda *_:cancelled.set())
    deadline=time.time()+180
    def allowed():
        now=ClientSettings.load(node);live=json.loads((node/'diagnostics.json').read_text())
        return (not cancelled.is_set() and now.connected and not now.paused and asdict(now.policy)==policy
            and fingerprints(node)==before and live['pid']==pid and time.time()*1000-live['timestamp_ms']<60000
            and not any(live.get(k,{}).get('active',0) for k in ('compute','language','language_training'))
            and managed_public_bytes(node)+retained_bytes()+64*1024**2<=now.policy.max_disk_gb*1024**3)
    if not allowed():raise ValueError('Native state or current contribution limits do not permit qualification')
    governor=ResourceGovernor(settings.policy,storage_path=node)
    lease,reason=governor.acquire(ResourceRequirements(kind='inference',gpu_required=False,ram_gb=2,disk_gb=.0625))
    if lease is None:raise ValueError('CPU fixture resources unavailable: '+reason)
    directory.mkdir(parents=True,mode=0o700);output=directory/'job';output.mkdir(mode=0o700)
    report={'success':False,'scope':'Verified frozen CPU engine synthetic self-test inside explicit Linux sandbox',
        'source_revision':revision,'physical_node_id':initial['node_id'],'physical_pid':pid,
        'engine_manifest_sha256':args.manifest_sha256,'production_weights_mounted':False,
        'operator_state_mounted':False,'network_namespace_shared':False,'cuda_devices_mounted':False,
        'native_service_updated':False,'native_learning_cycles_added':0,'model_promoted':False,
        'default_worker_isolation_enabled':False,'unknown_peer_execution_authorized':False,
        'started_at':time.time()}
    execution=SimpleNamespace(deadline=deadline,raise_if_cancelled=lambda:None)
    try:
        with lease,NodeStateLock(directory):
            value=ComputeSupervisor().run([str(executable),'--self-test'],{'device':'cpu'},execution,
                allowed=allowed,ram_gb=2,sandbox=CPUComputeSandbox((package,),(),output))
        report['engine_self_test']=value
        if (value.get('success') is not True or value.get('frozen') is not True or value.get('cuda_runtime') is not False
                or value.get('candidate_evaluated') is not True or value.get('candidate_signed_or_promoted') is not False
                or value.get('personal_state_loaded') is not False or value.get('job_process_signing_keys_loaded') is not False):
            raise ValueError('Frozen sandbox result does not match its restricted fixture scope')
        report.update(success=True,engine_self_test=value,finished_at=time.time())
    except BaseException as error:
        report.update(error_type=type(error).__name__,error=str(error)[:1024]);raise
    finally:
        report['native_controls_unchanged']=fingerprints(node)==before
        atomic_json(directory/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('directory','node-directory','package'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True)
    print(json.dumps(run(parser.parse_args()),allow_nan=False),flush=True)


if __name__=='__main__':main()
