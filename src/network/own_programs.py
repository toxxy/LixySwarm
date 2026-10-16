"""Locally opted-in public food queue and authenticated, code-free admission."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import ssl
import threading
import time
import urllib.request

from src.learning.own_food import build, recipe
from src.learning.own_food_catalog import PROGRAMS, descriptor, policy_identifier
from src.learning.public_math import canonical, digest, SOURCES, SOURCE_REVISION
from src.learning.swarm_policy import definition, state_id
from .artifact_store import digest_file
from .client_state import ClientSettings, NodeStateLock
from .engine_download import download_pinned, HTTPSRedirect
from .own_network_settings import OwnNetworkSettings
from .public_knowledge import KnowledgeSettings
from .own_storage import usage as managed_public_bytes
from .report_privacy import public_route
from .telemetry import atomic_json

ADMIT_OPERATION='swarm.program-admit.v1'
TERMINAL={'complete','awaiting_new_curriculum','no_improvement'}


def read_small(path,maximum=128*1024):
    if path.is_symlink() or path.stat().st_size>maximum:raise ValueError('Own program metadata exceeds its local bound')
    return json.loads(path.read_text())


def queue_controls(directory):
    path=Path(directory)/'own-curriculum-controls.json'
    if not path.exists():return {'version':1,'enabled':False,'maximum_programs':len(PROGRAMS)}
    value=read_small(path,4096)
    if (set(value)!={'version','enabled','maximum_programs'} or value['version']!=1
            or type(value['enabled']) is not bool or type(value['maximum_programs']) is not int
            or not 1<=value['maximum_programs']<=len(PROGRAMS)):
        raise ValueError('Invalid own curriculum controls')
    return value


def queue_enabled(directory):
    try:return queue_controls(directory)['enabled']
    except (OSError,ValueError,TypeError):return False


def supports_programs(service):
    try:
        manifest=read_small(Path(service.engine.command[0]).parent/'engine-package.json',2*1024**2)
        return {'swarm.gradient.v3','swarm.candidate.v3','swarm.audit.v1'}<=set(manifest['operations'])
    except (OSError,ValueError,TypeError,IndexError,KeyError):return False


def recover_selection(directory):
    """Finish only an unchanged local two-file intent; never rewind training."""
    from .own_swarm import OwnSwarmSettings
    directory=Path(directory);path=directory/'own-program-pending.json'
    if not path.exists():return
    plan=read_small(path)
    if plan.get('committed') is True:return
    descriptor(plan['program_id'])
    with NodeStateLock(directory/'own-state-selection'):
        own=asdict(OwnSwarmSettings.load(directory));controls=asdict(OwnNetworkSettings.load(directory))
        if own not in [plan['previous_own'],plan['own']] or controls not in [plan['previous_controls'],plan['controls']]:
            raise ValueError('Own public-food intent conflicts with changed local state; preserve it')
        client=ClientSettings.load(directory)
        if not client.connected or client.paused or not KnowledgeSettings.load(directory).enabled:
            raise ValueError('Own public-food selection is paused')
        selected=OwnSwarmSettings(**plan['own']);settings=OwnNetworkSettings(**plan['controls'])
        selected.validate();settings.validate()
        if selected.program_id!=plan['program_id'] or settings.program_id!=plan['program_id']:
            raise ValueError('Own public-food intent names different programs')
        settings.save(directory);selected.save(directory)
        atomic_json(path,{**plan,'committed':True,'committed_at':time.time()})
        atomic_json(directory/'own-network/admissions'/plan['program_id']/'selection.json',
            {**plan,'committed':True,'committed_at':time.time()})


def select_program(service,program_id,owner,*,coordinator,check):
    """Caller has already acquired pinned data; no peer can name code or paths."""
    from .own_swarm import OwnSwarmSettings
    row=descriptor(program_id);check()
    if not supports_programs(service):raise ValueError('Installed own engine does not support public-food programs')
    p=definition(policy_identifier(program_id))
    for key in ['training','evaluation','retention']:
        if digest_file(service.store.object_path(p[key+'_dataset_id']))!=(p[key+'_dataset_id'],p[key+'_bytes']):
            raise ValueError('Own public-food dependency is missing or changed')
    directory=service.directory
    with NodeStateLock(directory/'own-state-selection'):
        selected=OwnSwarmSettings.load(directory);controls=OwnNetworkSettings.load(directory)
        if not controls.follow_enabled or controls.owner_id!=owner or owner not in selected.requesters:
            raise ValueError('Own program coordinator has no local admission')
        if selected.version==3 and descriptor(selected.program_id)['index']>=row['index']:
            if selected.program_id==program_id:return
            raise ValueError('Own public food cannot rewind a consumed program')
        if service.active:raise ValueError('Own program cannot change while tensor work is active')
        check()
        current=state_id(selected.model_id,selected.optimizer_id,selected.step)
        following=replace(selected,version=3,program_id=program_id)
        next_controls=OwnNetworkSettings(version=3,enabled=coordinator,follow_enabled=True,owner_id=owner,
            initial_state_id=current,documents_per_worker=controls.documents_per_worker,
            workers_per_window=controls.workers_per_window,stop_after_rejections=controls.stop_after_rejections,
            parent_program_id=controls.contract(),program_id=program_id)
        plan={'version':1,'program_id':program_id,'selected_state_id':current,'committed':False,
            'previous_own':asdict(selected),'own':asdict(following),
            'previous_controls':asdict(controls),'controls':asdict(next_controls),'prepared_at':time.time()}
        archive=directory/'own-network/admissions'/program_id
        if (archive/'selection.json').exists():raise ValueError('Own program already has a different local admission record')
        atomic_json(archive/'previous-controls.json',asdict(controls))
        atomic_json(directory/'own-program-pending.json',plan)
    recover_selection(directory)


def acquire_source(service,check,cancel):
    root=service.root/'food-source';root.mkdir(parents=True,exist_ok=True)
    opener=urllib.request.build_opener(HTTPSRedirect(),urllib.request.HTTPSHandler(context=ssl.create_default_context())).open
    for name,(identifier,size) in SOURCES.items():
        path=root/name
        if path.is_file() and digest_file(path)==(identifier,size):continue
        if managed_public_bytes(service.directory)+2*size+20*1024**2>service.settings.policy.max_disk_gb*1024**3:
            raise ValueError('New public food exceeds authorized storage')
        suffix='LICENSE' if name=='LICENSE' else 'grade_school_math/data/'+name
        url='https://raw.githubusercontent.com/openai/grade-school-math/'+SOURCE_REVISION+'/'+suffix
        partial=root/(name+'.part');offset=[partial.stat().st_size if partial.exists() else 0]
        def progress(done):
            delta=max(0,done-offset[0]);offset[0]=done
            service.knowledge.budget.wait(delta,cancel_event=cancel,deadline=time.time()+15)
        download_pinned(url,size,partial,check=check,opener=opener,
            rate_mbps=lambda:service.settings.policy.max_bandwidth_mbps/2,event=cancel,progress=progress)
        if digest_file(partial)!=(identifier,size):raise ValueError('Public food source failed pinned verification')
        partial.replace(path)
    return root


def publish_food(service,program_id,check,cancel):
    row=descriptor(program_id);p=definition(policy_identifier(program_id))
    if all(service.store.has(p[key+'_dataset_id']) for key in ['training','evaluation','retention']):return
    source=acquire_source(service,check,cancel);check();compiled=build(source)
    if compiled['programs']!=PROGRAMS or digest(compiled['recipe'])!=row['recipe_id']:
        raise ValueError('Public food compilation differs from app-pinned recipes')
    for key in ['training','evaluation']:
        check();identifier=p[key+'_dataset_id'];raw=compiled['objects'][identifier]
        if service.store.has(identifier):continue
        if managed_public_bytes(service.directory)+2*len(raw)+20*1024**2>service.settings.policy.max_disk_gb*1024**3:
            raise ValueError('Compiled public food exceeds authorized storage')
        path=service.root/(identifier+'.food');path.write_bytes(raw)
        try:service.store.import_file(path,kind='dataset')
        finally:path.unlink(missing_ok=True)
    if not service.store.has(p['retention_dataset_id']):raise ValueError('Historical retention corpus is unavailable')


def advance(coordinator,contract):
    """Prospective fixed order; each terminated ledger remains in its own path."""
    c=coordinator;service=c.service;path=service.directory/'own-network/curriculum.json'
    try:
        c._check(contract);controls=queue_controls(service.directory)
        if not controls['enabled'] or c.ledger.get('pending') is not None or c.ledger['status'] not in TERMINAL:return
        if not supports_programs(service):raise ValueError('Public-food queue needs a compatible installed engine')
        queue=read_small(path) if path.exists() else {'version':1,'recipe_id':digest(recipe()),'programs':[],'status':'ready'}
        if queue['version']!=1 or queue['recipe_id']!=digest(recipe()) or len(queue['programs'])>len(PROGRAMS):
            raise ValueError('Own public-food queue changed recipe or bound')
        index=descriptor(c.settings.program_id)['index']+1 if c.settings.version==3 else 0
        if c.settings.version==3:
            current=next((v for v in queue['programs'] if v['program_id']==c.settings.program_id),None)
            if current is None:raise ValueError('Own active program is absent from its durable queue')
            current.update(status='terminated',termination=c.ledger['status'],cursor=c.ledger['cursor'],
                unconsumed_units=definition(c.settings.policy_id())['training_units']-c.ledger['cursor'],
                accepted=c.ledger['accepted'],rounds=c.ledger['round'],state_id=c.ledger['state_id'],
                ledger_sha256=hashlib.sha256(c.path.read_bytes()).hexdigest())
        if index>=controls['maximum_programs']:
            queue['status']='awaiting_new_public_food' if index>=len(PROGRAMS) else 'program_limit'
            atomic_json(path,queue);c._maintain(contract);c.maintenance_at=time.time()+30;return
        row=PROGRAMS[index];queue['status']='preparing';atomic_json(path,queue)
        publish_food(service,row['program_id'],lambda:c._check(contract),c.cancel)
        c._check(contract)
        if not any(v['program_id']==row['program_id'] for v in queue['programs']):
            queue['programs'].append({'program_id':row['program_id'],'index':index,'status':'selected',
                'parent_contract_id':contract,'initial_state_id':service.snapshot()['state_id']})
        atomic_json(path,queue)
        select_program(service,row['program_id'],c.settings.owner_id,coordinator=True,check=lambda:c._check(contract))
        queue['status']='learning';atomic_json(path,queue)
    except InterruptedError:pass
    except Exception as error:
        c.error='Public food: '+type(error).__name__+': '+str(error)[:160]
        c.maintenance_at=time.time()+30
        atomic_json(service.directory/'own-network/curriculum-status.json',{'status':'waiting','error':c.error,'at':time.time()})


def admit(service,payload,execution):
    execution.raise_if_cancelled();origin=execution.origin_node_id
    if not isinstance(payload,dict) or set(payload)!={'program_id'}:raise ValueError('Own admission needs a known public program')
    row=descriptor(payload['program_id']);settings=OwnNetworkSettings.load(service.directory)
    if (not service.ready or not service.knowledge.allowed or not settings.follow_enabled
            or settings.owner_id!=origin or origin not in service.engine.requesters or not supports_programs(service)):
        raise ValueError('Own public program has no local permission or compatible engine')
    selected=asdict(service.engine);permission=asdict(service.settings.policy)
    def check():
        execution.raise_if_cancelled()
        if (not service.ready or not service.knowledge.allowed or asdict(service.engine)!=selected
                or asdict(service.settings.policy)!=permission):raise ValueError('Own admission permission or state changed')
    if service.engine.version==3 and service.engine.program_id==row['program_id']:
        return {'admitted':True,'program_id':row['program_id'],'state_id':service.snapshot()['state_id']}
    p=definition(policy_identifier(row['program_id']))
    peer=next(r for r in service.knowledge.node.peers() if r['node_id']==origin)
    route=public_route({k:peer[k] for k in ['host','relay_host','transport','encrypted'] if k in peer})
    if not route['public_internet'] or not route['encrypted']:raise ValueError('Public food requires encrypted public Internet')
    transfers=[]
    for key in ['training','evaluation','retention']:
        check();identifier,size=p[key+'_dataset_id'],p[key+'_bytes'];local=service.store.has(identifier)
        if not local:
            if managed_public_bytes(service.directory)+2*size+20*1024**2>service.settings.policy.max_disk_gb*1024**3:
                raise ValueError('Own public food exceeds authorized storage')
            service.transfer.fetch(identifier,peer_id=origin,expected_size=size,
                timeout_s=min(60,max(1,execution.deadline-time.time())),cancel_event=execution.cancel_event)
        if digest_file(service.store.object_path(identifier))!=(identifier,size):raise ValueError('Own public food changed in transit')
        transfers.append({'artifact_id':identifier,'bytes':size,'initially_local':local,'route':route})
    select_program(service,row['program_id'],origin,coordinator=False,check=check)
    service.update(service.settings)
    return {'admitted':True,'program_id':row['program_id'],'state_id':service.snapshot()['state_id'],'transfers':transfers}
