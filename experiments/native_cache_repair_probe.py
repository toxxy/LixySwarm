"""Verify native cache repair over public LSP with a locally preseeded test cache.

This is one missing-adapter repair, not a clean install or another physical
replica. The physical participant remains running and its files are not edited.
"""
import argparse
from dataclasses import asdict
import ipaddress
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from desktop.backend import NodeController
from src.contribution import ContributionPolicy
from src.network.artifact_store import ArtifactStore,digest_file
from src.network.client_state import ClientSettings
from src.network.language_compute import LanguageSettings
from src.network.language_training import LanguageTrainingSettings
from src.network.public_knowledge import KnowledgeSettings
from src.network.public_memory import PublicMemorySettings
from src.network.telemetry import atomic_json
from src.release.language_versions import LanguageVersionRegistry
from .cold_recovery import accounted_bytes
from .prepare_public_math import fingerprints,outside_checkout
from .research_artifact import RESEARCH_NAMES


def run(args):
    physical=args.node_directory.resolve();directory=args.directory.resolve();app=args.app.resolve()
    if (directory.exists() or not outside_checkout(directory) or not outside_checkout(physical)
            or directory.is_relative_to(physical) or physical.is_relative_to(directory) or not app.is_file()):
        raise ValueError('Use a fresh external test cache and an existing native app')
    original=NodeController(physical);before=original.snapshot();settings=original.settings();policy=asdict(settings.policy)
    if not before['fresh'] or not settings.connected or settings.paused:raise ValueError('Physical donor is not available')
    before_files=fingerprints(physical);pid=before['diagnostics']['pid']
    directory.mkdir(mode=0o700,parents=True);node=directory/'node';node.mkdir(mode=0o700)
    extra=[Path.home()/'.local/state'/name for name in RESEARCH_NAMES];peak=0;cancelled=False;process=None
    def stop(*_):
        nonlocal cancelled
        cancelled=True
    for name in [signal.SIGINT,signal.SIGTERM]:signal.signal(name,stop)
    def allowed():
        nonlocal peak
        current=original.settings();snapshot=original.snapshot()
        total=accounted_bytes([physical,node],extra);peak=max(peak,total)
        return (not cancelled and current.connected and not current.paused and asdict(current.policy)==policy
            and snapshot['running'] and snapshot['diagnostics'].get('pid')==pid
            and total<=settings.policy.max_disk_gb*1024**3)
    linked=0;metadata=0
    for source in sorted((physical/'public').rglob('*')):
        relative=source.relative_to(physical/'public')
        if source.is_symlink():raise ValueError('Public seed contains a linked path')
        if not source.is_file() or any(part in {'incoming','transaction'} for part in relative.parts) or source.name.endswith(('.tmp','.part','.lock')):continue
        destination=node/'public'/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        if 'objects' in relative.parts:
            identifier=source.parent.name+source.name
            if digest_file(source)[0]!=identifier:raise ValueError('Public seed object is corrupt')
            os.link(source,destination);linked+=source.stat().st_size
        else:
            if source.stat().st_size>256*1024 or metadata+source.stat().st_size>16*1024**2:
                raise ValueError('Public metadata seed exceeds its scope')
            destination.write_bytes(source.read_bytes());metadata+=source.stat().st_size
    registry=LanguageVersionRegistry(node/'public/language','qwen3-1.7b');head=registry.head()
    identifier=head['release']['model_artifact_id'];store=ArtifactStore(node/'public/language/artifacts',max_artifact_bytes=4*1024**3)
    manifest=store.manifest(identifier);target=store.object_path(identifier)
    target.unlink()  # Only the owned test hardlink; the physical original stays intact.
    if store.has(identifier):raise ValueError('Test cache miss was not established')
    permission=ContributionPolicy.for_mode('relay');permission.max_disk_gb=settings.policy.max_disk_gb
    permission.max_bandwidth_mbps=min(2.,settings.policy.max_bandwidth_mbps/2)
    ClientSettings(policy=permission).save(node);KnowledgeSettings(enabled=True,consented_at=permission.consented_at).save(node)
    PublicMemorySettings(enabled=False).save(node);LanguageSettings().save(node);LanguageTrainingSettings().save(node)
    protected={name:(node/name).read_bytes() for name in ['controls.json','language-engine.json','language-training.json']}
    operator_head=(registry.head_path.read_bytes() if registry.head_path.exists() else None)
    environment=dict(os.environ);environment.pop('LIXYSWARM_BOOTSTRAP_SEEDS',None)
    report={'success':False,'scope':'Native repair of one missing adapter; all other public data locally preseeded',
        'whole_clean_install_proven':False,'extra_physical_participant':False,'new_independent_replica':False,
        'model_trained':False,'model_published':False,'preseeded_object_bytes_hardlinked':linked,'preseeded_metadata_bytes':metadata}
    began=time.monotonic()
    def wait(predicate,timeout):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            if not allowed():raise RuntimeError('Physical resource permission changed')
            if predicate():return
            time.sleep(.25)
        raise TimeoutError('Native cache repair did not reach its expected state')
    try:
        if not allowed():raise ValueError('Seed exceeds the actual shared disk consent')
        log=(directory/'native.log').open('ab')
        try:process=subprocess.Popen([str(app),'--node','--state-dir',str(node),'--host','0.0.0.0','--port','0'],
            env=environment,stdout=log,stderr=log,start_new_session=True)
        finally:log.close()
        controller=NodeController(node,port=0)
        wait(lambda:controller.snapshot()['fresh'] and controller.snapshot()['diagnostics'].get('knowledge',{}).get('sharing'),60)
        snapshot=controller.snapshot();assert snapshot['diagnostics']['runtime_version']=='0.4.0a20'
        assert snapshot['diagnostics']['pid']==process.pid
        controller.repair_language_cache('qwen3-1.7b')
        wait(lambda:controller.cache_repair_status()['state']=='complete',args.timeout)
        state=controller.cache_repair_status();snapshot=controller.snapshot()
        routes=[p for p in snapshot['diagnostics']['peers'] if p['node_id'] in {'c45094e5fd8c64e8f6f5be3aca703edb6527711a504fcdc747a0fba15c668255','ea914d97caafef310e50385034abae9a50ba86496b0e9cbf4679e6070e4c25b0'}]
        assert routes and all(p['encrypted'] and ipaddress.ip_address(p.get('relay_host') or p['host']).is_global for p in routes)
        assert digest_file(target)==(identifier,manifest.size)
        assert target.stat().st_ino!=ArtifactStore(physical/'public/language/artifacts').object_path(identifier).stat().st_ino
        assert state['release_id']==head['release']['release_id'] and not state['active_expert_changed']
        assert {name:(node/name).read_bytes() for name in protected}==protected
        assert (registry.head_path.read_bytes() if registry.head_path.exists() else None)==operator_head
        assert fingerprints(physical)==before_files and original.snapshot()['diagnostics']['pid']==pid
        report.update(success=True,release_id=state['release_id'],model_id=identifier,recovered_artifact_bytes=manifest.size,
            selected_expert_unchanged=True,operator_controls_unchanged=True,physical_node_unchanged=True,
            native_process=True,public_encrypted_lsp=True,peak_aggregate_managed_bytes=peak,
            physical_disk_quota_bytes=int(settings.policy.max_disk_gb*1024**3),seconds=time.monotonic()-began)
    except Exception as error:
        report.update(error_type=type(error).__name__,error=str(error),peak_aggregate_managed_bytes=peak)
        raise
    finally:
        if process is not None:
            process.terminate()
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
        report['test_process_stopped']=process is None or process.poll() is not None
        atomic_json(directory/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node-directory',type=Path,required=True);parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--app',type=Path,required=True);parser.add_argument('--timeout',type=int,default=240)
    args=parser.parse_args()
    if not 30<=args.timeout<=600:parser.error('Use a bounded cache repair deadline')
    print(json.dumps(run(args),allow_nan=False),flush=True)


if __name__=='__main__':main()
