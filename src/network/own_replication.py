"""Native public-state transfer, independent replay and locally permitted adoption."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time
import uuid

from src.learning.swarm_distributed import WORKERS,policy_id,state_id,definition
from src.learning.swarm_state_audit import replay_request
from .artifact_store import digest_file
from .own_learning_settings import OwnLearningSettings
from .own_state import verify_transition,activate
from .public_knowledge import managed_public_bytes
from .work_protocol import ResultReceipt,WorkResult
from .report_privacy import public_route
from .telemetry import atomic_json

HEAD_OPERATION='swarm.state-head.v1'
ADOPT_OPERATION='swarm.state-adopt.v1'
ADOPT_DISK_GB=.12


class OwnStateReplication:
    def __init__(self,service):
        self.service=service;self.knowledge=service.knowledge
        self.knowledge.work.register_handler(HEAD_OPERATION,'artifact',self.head,permit_cached=lambda *_:False)
        self.knowledge.work.register_handler(ADOPT_OPERATION,'training',self.adopt,permit_cached=self.permit_cached)

    def head(self,payload,execution):
        execution.raise_if_cancelled()
        if payload!={} or not self.knowledge.allowed:raise ValueError('Own state metadata is unavailable')
        return self.service.snapshot()

    def allowed(self,origin):
        try:
            settings=OwnLearningSettings.load(self.service.directory)
            return (self.service.ready and self.knowledge.allowed and settings.follow_enabled
                and origin==settings.owner_id and origin in self.service.engine.requesters)
        except (OSError,ValueError,TypeError):return False

    def permit_cached(self,execution,result):
        try:
            return (self.allowed(execution.origin_node_id) and result['success'] is True
                and result['state_id']==self.service.snapshot()['state_id']
                and result['training_policy_id']==policy_id())
        except (ValueError,TypeError,KeyError):return False

    def adopt(self,payload,execution):
        service=self.service;own=self.knowledge.node.identity.node_id_hex;origin=execution.origin_node_id
        execution.raise_if_cancelled()
        if (not isinstance(payload,dict) or set(payload)!={'candidate_result'}
                or len(json.dumps(payload,allow_nan=False))>16*1024 or not self.allowed(origin)
                or execution.resource_requirements().gpu_required
                or execution.resource_requirements().ram_gb<definition()['ram_gb']
                or execution.resource_requirements().disk_gb<ADOPT_DISK_GB):
            raise ValueError('Own state adoption has no local permission or resource lease')
        candidate=payload['candidate_result'];work=WorkResult.from_dict(candidate)
        if (work.status!='ok' or not work.receipt or not ResultReceipt.from_dict(work.receipt).verify(work,
                expected_worker_id=origin,expected_requester_id=origin)):
            raise ValueError('Own state must be signed by the locally selected coordinator')
        output=work.output
        following=state_id(output['candidate_model_id'],output['candidate_optimizer_id'],output['candidate_optimizer_step'])
        current=service.snapshot()['state_id']
        if output['training_policy_id']!=policy_id() or output['accepted'] is not True:
            raise ValueError('Own state did not pass the fixed training policy')
        if current not in {output['state_id'],following}:raise ValueError('Own state does not extend the local lineage')
        if current==following:
            history=service.root/'history'/(following+'.json')
            if not history.is_file() or json.loads(history.read_text())['candidate_result']!=candidate:
                raise ValueError('Own state retry has different transition provenance')
            return {'success':True,'already_selected':True,'state_id':following,
                'optimizer_step':service.engine.step,'training_policy_id':policy_id()}
        selected=asdict(service.engine);permission=asdict(service.settings.policy)
        def allowed():
            return (self.allowed(origin) and asdict(service.engine)==selected
                and asdict(service.settings.policy)==permission)
        def check():
            execution.raise_if_cancelled()
            if not allowed():raise ValueError('Own adoption state or local permission changed')
        transfers=[]
        def fetch(identifier,size):
            check()
            if type(size) is not int or not 0<size<=definition()['maximum_artifact_bytes']:
                raise ValueError('Own state dependency exceeds its bound')
            initially_local=service.store.has(identifier)
            if not initially_local:
                if origin==own:raise ValueError('Coordinator lost a required own state dependency')
                if origin not in self.knowledge.node.connected_node_ids:self.knowledge.node.connect_node(origin,timeout=12)
                peer=next(row for row in self.knowledge.node.peers() if row['node_id']==origin)
                route=public_route({k:peer[k] for k in ['host','relay_host','transport','encrypted'] if k in peer})
                if not route['public_internet'] or not route['encrypted']:
                    raise ValueError('Learned state must arrive over encrypted public Internet')
                if managed_public_bytes(service.directory)+2*size+ADOPT_DISK_GB*1024**3>service.settings.policy.max_disk_gb*1024**3:
                    raise ValueError('Own replica exceeds public storage consent')
                began=time.monotonic();chunks=[]
                service.transfer.fetch(identifier,peer_id=origin,expected_size=size,
                    timeout_s=min(600,max(1,execution.deadline-time.time())),cancel_event=execution.cancel_event,
                    observer=chunks.append)
                transfers.append({'artifact_id':identifier,'bytes':size,'route':route,
                    'seconds':time.monotonic()-began,'encoded_data_bytes':sum(v['encoded_data_bytes'] for v in chunks)})
            if digest_file(service.store.object_path(identifier))!=(identifier,size):
                raise ValueError('Own replica dependency failed content verification')
            check()
        fetch(output['evaluation_id'],output['evaluation_bytes'])
        transition=verify_transition(candidate,service.store.object_path(output['evaluation_id']).read_bytes())
        for name in ['candidate_model','candidate_optimizer']:fetch(output[name+'_id'],output[name+'_bytes'])
        evaluation=json.loads(service.store.object_path(output['evaluation_id']).read_text())
        for row in evaluation['contributions']:
            gradient=row['result']['output'];fetch(gradient['gradient_id'],gradient['gradient_bytes'])
        check()
        if managed_public_bytes(service.directory)+ADOPT_DISK_GB*1024**3>service.settings.policy.max_disk_gb*1024**3:
            raise ValueError('Own replay exceeds current public storage permission')
        job=service.root/'jobs'/uuid.uuid4().hex;job.mkdir(parents=True,mode=0o700)
        with service.lock:service.active+=1
        service.changed.set()
        try:
            request=replay_request(service.directory,candidate,job/'engine',
                deadline=min(execution.deadline,time.time()+definition()['maximum_seconds']),
                cpu_threads=max(1,min(8,int(service.resources['cpu_cores']*service.settings.policy.cpu_percent/100))))
            audit=service.supervisor.run(service.engine.command,request,execution,allowed=allowed,ram_gb=definition()['ram_gb'])
            check()
            # Evidence is durable before the atomic selection, so an interrupted
            # caller can retry without losing or repeating the accepted step.
            atomic_json(service.root/'audits'/(following+'.json'),audit)
            adopted=activate(service.directory,candidate,audit)
            service.update(service.settings)
            result={**adopted,'training_policy_id':policy_id(),'audit':audit,'transfers':transfers}
            atomic_json(service.root/'adoptions'/(following+'.json'),result)
            return result
        finally:
            shutil.rmtree(job)
            with service.lock:service.active-=1
            service.changed.set()
