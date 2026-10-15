"""Bounded native cold preparation of one existing physical pilot participant.

Administration only. This reprovisions its public cache, not a third computer.
Use install before run: a persisted recovery unit and an original-service guard
survive a killed trial or reboot. Model/data/gradient acquisition stays in the
frozen application's encrypted public LSP transport, never in this supervisor.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

from desktop.backend import NodeController
from src.network.artifact_store import ArtifactStore,digest_file
from src.network.client_state import ClientSettings,NodeStateLock
from src.network.engine_catalog import language_engine
from src.network.engine_package import verify_package
from src.network.language_compute import LanguageSettings
from src.network.language_training import LanguageTrainingSettings
from src.network.public_knowledge import KnowledgeSettings
from src.network.public_memory import PublicMemorySettings
from src.network.telemetry import atomic_json
from src.network.work_protocol import WorkResult,ResultReceipt
from src.release.language_versions import LanguageVersionRegistry,dependencies,verify_contributions
from .cold_checkpoint import ColdCheckpoint,atomic_bytes,fingerprint
from .cold_inventory import inventory,validate_replica,evict_weights,restore_cache_links,FAMILIES
from .cold_recovery import (accounted_bytes,checked_file,merge_artifacts,merge_engine_downloads,
    newest_status,read_counter,repair_view)
from .prepare_public_math import outside_checkout
from .research_artifact import RESEARCH_NAMES

ORIGINAL='lixyswarm-node.service'
TRIAL='lixyswarm-cold-trial-20261015.service'
RECOVERY='lixyswarm-cold-recovery-20261015.service'
LAPTOP='ea914d97caafef310e50385034abae9a50ba86496b0e9cbf4679e6070e4c25b0'
PROVIDER='c45094e5fd8c64e8f6f5be3aca703edb6527711a504fcdc747a0fba15c668255'
FRESH_LIMIT=8*1024**3
MARGIN=16*1024**2


def systemctl(*args):
    return subprocess.run(['systemctl','--user',*args],check=True,capture_output=True,text=True,timeout=40).stdout.strip()


def read(path,maximum=256*1024):
    path=Path(path)
    if path.is_symlink() or path.stat().st_size>maximum:raise ValueError('Administrative record exceeds its scope')
    return json.loads(path.read_text())


def load(session):
    value=read(session)
    if value.get('schema')!='lixyswarm-native-cold-session-v1':raise ValueError('Unknown cold session')
    original=Path(value['original']);trial=Path(value['trial']);folder=Path(session).parent
    if (not original.is_absolute() or not trial.is_absolute() or not outside_checkout(original)
            or not outside_checkout(folder) or trial!=folder/'node'
            or original.resolve().is_relative_to(folder.resolve()) or folder.resolve().is_relative_to(original.resolve())):
        raise ValueError('The cold session is not confined to separate external directories')
    if value['physical_id']!=LAPTOP or value['provider_id']!=PROVIDER:raise ValueError('This test is limited to the two known physical PCs')
    if not 1800<=value['timeout']<=6*3600:raise ValueError('Cold test requires a bounded deadline')
    return value,original,trial,folder


def extras():return [Path.home()/'.local/state'/name for name in RESEARCH_NAMES]


def stop_process(process):
    if process is None or process.poll() is not None:return
    process.terminate()
    try:process.wait(timeout=25)
    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)


def start_native(value,directory,folder,name):
    if digest_file(value['app'])[0]!=value['app_sha256']:raise ValueError('Pinned native application changed')
    environment=dict(os.environ);environment.pop('LIXYSWARM_BOOTSTRAP_SEEDS',None)
    with (folder/name).open('ab') as log:
        return subprocess.Popen([value['app'],'--node','--state-dir',str(directory),'--host','0.0.0.0','--port','0'],
            env=environment,stdout=log,stderr=log)


def fresh(controller,process):
    snapshot=controller.snapshot()
    return (process.poll() is None and snapshot['fresh'] and snapshot['diagnostics'].get('pid')==process.pid
        and snapshot['diagnostics'].get('runtime_version')=='0.4.0a20')


def install(args):
    """Arm recovery before the trial is allowed to stop or evict anything."""
    directory=args.directory.resolve();original=args.node_directory.resolve();app=args.app.resolve()
    if directory.exists() or not outside_checkout(directory) or not app.is_file():raise ValueError('Use a fresh external test directory and native app')
    if directory.is_relative_to(original) or original.is_relative_to(directory):raise ValueError('Test and operator state overlap')
    controller=NodeController(original);snapshot=controller.snapshot();settings=controller.settings()
    if not snapshot['fresh'] or snapshot['diagnostics'].get('node_id')!=LAPTOP:raise ValueError('The existing physical laptop must be available')
    if systemctl('show',ORIGINAL,'-p','MainPID','--value')!=str(snapshot['diagnostics']['pid']):raise ValueError('Original service does not own this participant')
    if not settings.connected or settings.paused or settings.policy.max_disk_gb*1024**3<FRESH_LIMIT:raise ValueError('Existing permissions do not allow this trial')
    if snapshot['diagnostics']['language_learning'].get('enabled'):raise ValueError('Do not reprovision an active language coordinator')
    provider=read(args.provider_inventory,4*1024**2);local=inventory(original);validate_replica(local,provider)
    if provider['node_id']!=PROVIDER or time.time()-provider.get('checked_at',0)>600:raise ValueError('Recheck the independent provider immediately before arming')
    if any(len(g['paths'])!=g['link_count'] for g in local['inode_groups']):raise ValueError('Original cache has external weight links')
    retained=accounted_bytes([original],extras())-local['reclaimable_weight_bytes']
    if retained+FRESH_LIMIT>settings.policy.max_disk_gb*1024**3:raise ValueError('Empty preparation would exceed the combined physical quota')
    directory.mkdir(mode=0o700,parents=True)
    value={'schema':'lixyswarm-native-cold-session-v1','original':str(original),'trial':str(directory/'node'),
        'physical_id':LAPTOP,'provider_id':PROVIDER,'app':str(app),'app_sha256':digest_file(app)[0],
        'policy':asdict(settings.policy),'timeout':args.timeout,'phase':'armed','created_at':time.time(),
        'original_pid':snapshot['diagnostics']['pid'],'provider_inventory':provider,'planned_inventory':local,
        'source_revision':args.source_revision,'initial_public_cache_empty':True,'extra_physical_participant':False}
    session=directory/'session.json';atomic_json(session,value)
    # systemd quoting is separate from shell quoting; reject substitution syntax.
    def word(s):
        s=str(s)
        if any(c in s for c in ['%','\n','\r','"','\\']):raise ValueError('Unsupported service command path')
        return '"'+s+'"'
    import sys
    command=' '.join(map(word,[sys.executable,'-m','experiments.cold_native_trial']))
    cwd=Path(__file__).resolve().parents[1]
    word(cwd)  # Validate the path; this scalar setting does not unquote shell words.
    common='WorkingDirectory='+str(cwd)+'\nUMask=0077\nKillMode=control-group\nTimeoutStopSec=45\n'
    recovery=('[Unit]\nDescription=Restore physical LixySwarm participant after cold trial\n'
        f'Before={ORIGINAL}\nConflicts={TRIAL}\nAfter=network-online.target {TRIAL}\nWants=network-online.target\n'
        '[Service]\nType=oneshot\nRemainAfterExit=yes\nRestart=on-failure\nRestartSec=30\nTimeoutStartSec=infinity\n'+common+
        f'ExecStart={command} recover --session {word(session)}\n'
        f'ExecStartPost=/usr/bin/systemctl --user --no-block start {ORIGINAL}\n[Install]\nWantedBy=default.target\n')
    trial=('[Unit]\nDescription=Native public-cache reprovisioning of physical laptop\nAfter=network-online.target\n'
        '[Service]\nType=oneshot\nTimeoutStartSec='+str(args.timeout+180)+'\nMemoryMax=12G\n'+common+
        f'ExecStart={command} run --session {word(session)}\n'
        f'ExecStopPost=/usr/bin/systemctl --user --no-block start {RECOVERY}\n')
    units=Path.home()/'.config/systemd/user';guard=units/(ORIGINAL+'.d/90-cold-recovery.conf')
    for p in [units/RECOVERY,units/TRIAL,guard]:
        if p.exists():raise ValueError('A previous cold recovery installation already owns these units')
    # Recovery is installed and enabled before the original unit receives its guard.
    atomic_bytes(units/RECOVERY,recovery.encode());atomic_bytes(units/TRIAL,trial.encode())
    systemctl('daemon-reload');systemctl('enable',RECOVERY)
    atomic_bytes(guard,f'[Unit]\nRequires={RECOVERY}\nAfter={RECOVERY}\n'.encode());systemctl('daemon-reload')
    value['recovery_armed']=True
    value['unit_hashes']={str(p):fingerprint(p) for p in [units/RECOVERY,units/TRIAL,guard]}
    atomic_json(session,value)
    return {'session':str(session),'recovery_armed':True,'original_stopped':False,'public_weights_evicted':False}


def verify_donation(value,donation,prepared_at):
    from src.learning.language_policy import dataset_id,policy_id
    result=WorkResult.from_dict(donation['result'])
    if (result.status!='ok' or result.finished_at<prepared_at
            or not ResultReceipt.from_dict(result.receipt).verify(result,expected_worker_id=LAPTOP,expected_requester_id=PROVIDER)
            or donation.get('artifact_initially_local') is not False or donation.get('encoded_data_bytes',0)<=0):
        raise ValueError('Cold donation is not a new signed remote contribution')
    head=value['planned_inventory']['heads']['qwen3-1.7b']
    payload=donation['payload']
    if (payload['release_id']!=head or payload['adapter_id']!=value['model_id']
            or payload['dataset_id']!=dataset_id(4) or payload['training_policy_id']!=policy_id(4)
            or payload['start']!=2048 or payload['count']!=2
            or result.output['start']!=2048 or result.output['count']!=2):raise ValueError('Cold donation uses another public fragment')
    verify_contributions([result.to_dict()],base_name='qwen3-1.7b',adapter_id=payload['adapter_id'],
        dataset_id=payload['dataset_id'],require_distributed=False,policy_version=4)
    import ipaddress
    route=donation['route']
    if not route['encrypted'] or not ipaddress.ip_address(route.get('relay_host') or route['host']).is_global:
        raise ValueError('Cold donation did not use encrypted public Internet')
    return result


def run(session):
    value,original,trial,folder=load(session)
    if not value.get('recovery_armed') or value['phase']!='armed' or (folder/'recovered.json').exists():
        raise ValueError('Only a fresh armed session may run')
    if any(fingerprint(Path(p))!=digest for p,digest in value['unit_hashes'].items()) or systemctl('is-enabled',RECOVERY)!='enabled':
        raise ValueError('The independent recovery unit or original-service guard changed')
    if time.time()-value['provider_inventory']['checked_at']>600:raise ValueError('Provider evidence expired before the trial began')
    cancelled=False;process=None;peak=0;began=time.time();controller=NodeController(trial,port=0)
    def cancel(*_):
        nonlocal cancelled
        cancelled=True
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,cancel)
    def allowed():
        nonlocal peak
        total=accounted_bytes([original,trial],extras());peak=max(peak,total)
        settings=ClientSettings.load(original)
        if (cancelled or asdict(settings.policy)!=value['policy'] or not settings.connected or settings.paused
                or total>settings.policy.max_disk_gb*1024**3 or time.time()-began>value['timeout']):
            raise RuntimeError('The cold trial reached its deadline, cancellation or physical resource boundary')
    def wait(predicate,seconds):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            allowed()
            if process.poll() is not None:raise RuntimeError('Native trial process stopped')
            if predicate():return
            time.sleep(1)
        raise TimeoutError('Native trial did not reach the expected state before its deadline')
    try:
        allowed()
        if not KnowledgeSettings.load(original).enabled:raise ValueError('Public storage permission was withdrawn')
        live=NodeController(original).snapshot()
        if not live['fresh'] or live['diagnostics'].get('pid')!=value['original_pid']:
            raise ValueError('The physical participant changed after arming recovery')
        systemctl('stop',ORIGINAL)
        with NodeStateLock(original):
            allowed()
            with NodeStateLock(trial):
                checkpoint=ColdCheckpoint.capture(original,trial,folder/'checkpoint.json')
                evict_weights(original,value['planned_inventory'],value['provider_inventory'])
                checkpoint.record('weights_evicted')
                atomic_bytes(trial/'identity.key',(original/'identity.key').read_bytes())
                atomic_bytes(trial/'status.json',(original/'status.json').read_bytes())
                settings=ClientSettings.load(original);settings.policy.max_disk_gb=FRESH_LIMIT/1024**3;settings.save(trial)
                KnowledgeSettings.load(original).save(trial);PublicMemorySettings(enabled=False).save(trial)
                # No model, engine, training settings, ledger or counters are seeded.
                assert not any((trial/name).exists() for name in ['public','engines','language-models','language-training.json'])
            value.update(phase='preparing',started_at=began);atomic_json(session,value)
            process=start_native(value,trial,folder,'trial-native.log')
            wait(lambda:fresh(controller,process) and controller.snapshot()['diagnostics'].get('knowledge',{}).get('sharing'),90)
            if controller.automatic_preparation_target()!='language':raise ValueError('Native automatic preparation did not choose the GPU expert')
            controller.prepare_language()
            wait(lambda:controller.snapshot()['diagnostics'].get('language_preparation',{}).get('state')=='complete',value['timeout'])
            wait(lambda:controller.snapshot()['diagnostics']['language_training']['ready'],45)
            head=LanguageVersionRegistry(trial/'public/language','qwen3-1.7b').head()
            if head['release']['release_id']!=value['planned_inventory']['heads']['qwen3-1.7b']:raise ValueError('The provider changed the accepted model during the trial')
            prepared=time.time();value.update(phase='awaiting_donation',prepared_at=prepared,trial_pid=process.pid,
                prepared_release_id=head['release']['release_id'],model_id=head['release']['model_artifact_id'])
            atomic_json(session,value);atomic_json(folder/'ready.json',{'physical_id':LAPTOP,'prepared_at':prepared,
                'release_id':value['prepared_release_id'],'model_id':value['model_id']})
            wait(lambda:(folder/'donation.json').exists(),1800)
            donation=read(folder/'donation.json');verify_donation(value,donation,prepared)
            wait(lambda:read_counter(trial,'language-training-status.json')>=1,30)
            value.update(phase='trial_complete',success=True,donation_verified=True,peak_aggregate_managed_bytes=peak,
                seconds=time.time()-began,physical_service_updated=False,model_promoted=False,new_independent_replica=False)
            atomic_json(session,value)
    except Exception as error:
        value.update(success=False,error_type=type(error).__name__,error=str(error),peak_aggregate_managed_bytes=peak)
        atomic_json(session,value);raise
    finally:
        stop_process(process)


def prune_owned_code(trial,entry):
    """Remove only trial extraction scratch and verified redundant code pieces."""
    root=trial/'engine-downloads';staging=root/(entry.sha256+'.unpacked')
    if staging.exists():
        if staging.is_symlink() or not staging.is_dir():raise ValueError('Unexpected extraction scratch')
        shutil.rmtree(staging)
    for staging in (trial/'engines').glob('.install-*'):
        if (staging.is_symlink() or not staging.is_dir() or not re.fullmatch(r'\.install-[a-z0-9_]{8}',staging.name)
                or {p.name for p in staging.iterdir()}-{'package'}):raise ValueError('Unexpected engine installation scratch')
        package=staging/'package'
        if package.exists():verify_package(package,entry.manifest_sha256,profile=entry.profile)
        shutil.rmtree(staging)
    archive=root/(entry.sha256+'.zip')
    assembling=root/(entry.sha256+'.assembling')
    if archive.exists():
        checked_file(archive,entry.sha256,entry.bytes)
        for part in entry.parts:
            path=root/(part.sha256+'.piece')
            if path.exists():checked_file(path,part.sha256,part.bytes);path.unlink()
    elif assembling.exists():
        for part in entry.parts:checked_file(root/(part.sha256+'.piece'),part.sha256,part.bytes)
    if assembling.exists():
        if assembling.is_symlink() or not assembling.is_file() or assembling.stat().st_size>entry.bytes:raise ValueError('Invalid code assembly scratch')
        assembling.unlink()


def recover(session):
    value,original,trial,folder=load(session)
    if (folder/'recovered.json').exists():return read(folder/'recovered.json')
    if not (folder/'checkpoint.json').exists():
        # A kill before capture cannot have evicted a single public model file.
        result={'recovered':True,'weights_were_evicted':False};atomic_json(folder/'recovered.json',result);return result
    checkpoint=ColdCheckpoint(folder/'checkpoint.json');process=None;controller=NodeController(original,port=0)
    entry=language_engine('cuda')
    if entry is None:raise ValueError('The locally pinned code catalog is unavailable')
    with NodeStateLock(original),NodeStateLock(trial):
        if checkpoint.original!=original or checkpoint.trial!=trial:raise ValueError('Checkpoint points to another participant')
        restore_cache_links(original,value['planned_inventory'])
        original_store=ArtifactStore(original/'public/language/artifacts',max_artifact_bytes=4*1024**3)
        trial_store=ArtifactStore(trial/'public/language/artifacts',max_artifact_bytes=4*1024**3)
        required={i:size for name in FAMILIES for bundle in LanguageVersionRegistry(original/'public/language',name).history(
            value['planned_inventory']['heads'][name]) for i,size in dependencies(bundle)}
        merge_artifacts(trial_store,original_store,allowed_partials=required)
        prune_owned_code(trial,entry)
        # Keep the new installation when all original knowledge also fits. Its
        # verified code is disposable if otherwise it would crowd out knowledge.
        missing=sum(size for i,size in required.items() if not original_store.has(i))
        quota=ClientSettings.load(original).policy.max_disk_gb*1024**3
        engine=trial/'engines'/entry.manifest_sha256
        if accounted_bytes([original,trial],extras())+missing+MARGIN>quota and engine.exists():
            verify_package(engine,entry.manifest_sha256,profile=entry.profile);shutil.rmtree(engine)
            checkpoint.record('recovering',trial_code_released_for_knowledge=True)
        merge_engine_downloads(trial,original,[entry])
        newest_status(original,trial,LAPTOP)
        checkpoint.suspend_compute()
        checkpoint.set_owned_control('language-engine.json',json.dumps(asdict(LanguageSettings())).encode())
        checkpoint.set_owned_control('language-training.json',json.dumps(asdict(LanguageTrainingSettings())).encode())
        checkpoint.record('repairing_original')
    # The native process owns original's lock while receiving the missing bytes.
    # Trial is stopped by the recovery unit's Conflicts/After ordering.
    try:
        process=start_native(value,original,folder,'recovery-native.log')
        end=time.monotonic()+24*3600
        while not fresh(controller,process):
            if process.poll() is not None or time.monotonic()>end:raise RuntimeError('Native recovery process unavailable')
            time.sleep(1)
        for name in FAMILIES:
            controller.repair_language_cache(name)
            while controller.cache_repair_status()['state']!='complete':
                if process.poll() is not None or time.monotonic()>end:raise RuntimeError('Recovery is retained but could not finish')
                settings=ClientSettings.load(original)
                if (not settings.connected or settings.paused or fingerprint(original/'controls.json') not in checkpoint.value['owned_hashes'].get('controls.json',[])):
                    raise RuntimeError('Operator changed recovery permissions; preserving its progress')
                if accounted_bytes([original,trial],extras())>settings.policy.max_disk_gb*1024**3:raise ValueError('Recovery exceeded the aggregate physical quota')
                time.sleep(2)
    finally:stop_process(process)
    with NodeStateLock(original),NodeStateLock(trial):
        for name in FAMILIES:
            registry=LanguageVersionRegistry(original/'public/language',name)
            chain=registry.history(value['planned_inventory']['heads'][name]);registry.verify_available(chain,original_store)
            for bundle in chain:
                descriptor=bundle['descriptor'];release=bundle['release']
                declared={row['name']:(row['artifact_id'],row['bytes']) for row in descriptor['files']}
                declared['adapter.safetensors']=(release['model_artifact_id'],descriptor['adapter_bytes'])
                repair_view(original,original_store,release['release_id'],declared)
        # Verified trial views now only duplicate the recovered original CAS.
        # Release their known links, retaining any unexpected paths for review.
        for name in FAMILIES:
            registry=LanguageVersionRegistry(original/'public/language',name)
            for bundle in registry.history(value['planned_inventory']['heads'][name]):
                descriptor=bundle['descriptor'];release=bundle['release']
                declared={row['name']:(row['artifact_id'],row['bytes']) for row in descriptor['files']}
                declared['adapter.safetensors']=(release['model_artifact_id'],descriptor['adapter_bytes'])
                view=trial/'language-models'/release['release_id']
                for filename,(identifier,size) in declared.items():
                    path=view/filename
                    if path.exists():
                        checked_file(path,identifier,size);checked_file(original_store.object_path(identifier),identifier,size);path.unlink()
        metadata=checkpoint.recover_metadata();restored=inventory(original)
        if set(restored['verified_transferable'])!=set(value['planned_inventory']['weights']):raise ValueError('Recovered public knowledge is incomplete')
        result={'recovered':True,'physical_id':LAPTOP,'original_knowledge_verified':True,'metadata':metadata,
            'aggregate_managed_bytes':accounted_bytes([original,trial],extras()),'finished_at':time.time()}
        atomic_json(folder/'recovered.json',result);checkpoint.record('recovered')
        return result


def peek(session):
    value,original,trial,folder=load(session)
    snapshot=NodeController(original).snapshot();diagnostic=snapshot['diagnostics']
    states={}
    for unit in [ORIGINAL,TRIAL,RECOVERY]:
        raw=systemctl('show',unit,'-p','ActiveState','-p','SubState','-p','MainPID','-p','Result')
        states[unit]=dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
    trial_diagnostic=NodeController(trial,port=0).snapshot()['diagnostics']
    return {'phase':value['phase'],'units':states,
        'ready':read(folder/'ready.json') if (folder/'ready.json').exists() else None,
        'recovered':read(folder/'recovered.json') if (folder/'recovered.json').exists() else None,
        'preparation':trial_diagnostic.get('language_preparation'),
        'original':{'fresh':snapshot['fresh'],'node_id':diagnostic.get('node_id'),'pid':diagnostic.get('pid'),
            'runtime_version':diagnostic.get('runtime_version'),'language':diagnostic.get('language'),
            'language_training':diagnostic.get('language_training')}}


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    setup=sub.add_parser('install');setup.add_argument('--node-directory',type=Path,required=True)
    setup.add_argument('--directory',type=Path,required=True);setup.add_argument('--app',type=Path,required=True)
    setup.add_argument('--provider-inventory',type=Path,required=True);setup.add_argument('--source-revision',required=True)
    setup.add_argument('--timeout',type=int,default=5*3600)
    for name in ['run','recover','peek']:
        command=sub.add_parser(name);command.add_argument('--session',type=Path,required=True)
    args=parser.parse_args()
    result=install(args) if args.action=='install' else run(args.session) if args.action=='run' else recover(args.session) if args.action=='recover' else peek(args.session)
    if result is not None:print(json.dumps(result,allow_nan=False),flush=True)


if __name__=='__main__':main()
