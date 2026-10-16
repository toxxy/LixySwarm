"""Own-weight training on the physical node's existing identity and governor."""
from dataclasses import asdict,dataclass,field
import json
from pathlib import Path
import shutil
import threading
import uuid

from src.learning.swarm_policy import (GRADIENT_OPERATION,CANDIDATE_OPERATION,WORKERS,
    NETWORK_GRADIENT_OPERATION,NETWORK_CANDIDATE_OPERATION,network_policy_id,
    FOOD_GRADIENT_OPERATION,FOOD_CANDIDATE_OPERATION,
    IDENTIFIER,definition,policy_id,state_id,validate_payload,verify_contributions,verdict)
from src.learning.public_math import ARTIFACTS
from .artifact_store import ArtifactStore,ArtifactService,digest_file
from .compute_process import ComputeSupervisor,ComputeProcessError
from .telemetry import atomic_json
from .own_storage import usage as managed_public_bytes


def public_data_path(directory, identifier):
    directory=Path(directory)
    own=directory/'public/own/artifacts/objects'/identifier[:2]/identifier[2:]
    return own if own.is_file() else directory/'public/language/artifacts/objects'/identifier[:2]/identifier[2:]


@dataclass
class OwnSwarmSettings:
    version:int=1
    enabled:bool=False
    command:list=field(default_factory=list)
    model_id:str=''
    optimizer_id:str=''
    step:int=0
    requesters:list=field(default_factory=list)
    program_id:str=''

    def validate(self):
        if (type(self.version) is not int or self.version not in {1,2,3} or type(self.enabled) is not bool
                or not isinstance(self.command,list) or len(self.command)>4
                or any(not isinstance(v,str) or not v or len(v)>4096 for v in self.command)
                or not isinstance(self.requesters,list) or len(self.requesters)>8
                or any(not isinstance(v,str) or not IDENTIFIER.fullmatch(v) for v in self.requesters)
                or len(set(self.requesters))!=len(self.requesters)
                or self.version==1 and (len(self.requesters)>2 or any(v not in WORKERS for v in self.requesters))):
            raise ValueError('Own-swarm settings require locally admitted coordinators and a local engine')
        if self.version<3 and self.program_id:raise ValueError('Historical own state cannot name new food')
        self.policy_id()
        if self.enabled:
            state_id(self.model_id,self.optimizer_id,self.step)
            if not self.command or not all(Path(v).is_absolute() for v in self.command) or not self.requesters:
                raise ValueError('Own-swarm computation requires a locally installed engine and selected state')

    def policy_id(self):
        if self.version==3:
            from src.learning.own_food_catalog import policy_identifier
            return policy_identifier(self.program_id)
        return network_policy_id() if self.version==2 else policy_id()

    def save(self,directory):
        self.validate();atomic_json(Path(directory)/'own-swarm.json',asdict(self))

    @classmethod
    def load(cls,directory):
        path=Path(directory)/'own-swarm.json'
        if not path.exists():return cls()
        if path.stat().st_size>8192:raise ValueError('Own-swarm settings exceed their bound')
        raw=json.loads(path.read_text())
        if not isinstance(raw,dict) or set(raw) not in [set(asdict(cls())),set(asdict(cls()))-{'program_id'}]:raise ValueError('Own-swarm controls have unknown fields')
        result=cls(**raw);result.validate();return result


class OwnSwarmService:
    def __init__(self,knowledge,directory,resources):
        self.knowledge,self.directory,self.resources=knowledge,Path(directory),resources
        self.root=self.directory/'public/own'
        self.store=ArtifactStore(self.root/'artifacts',max_artifact_bytes=16*1024**2,max_total_bytes=1024**3)
        self.transfer=ArtifactService(knowledge.work,self.store,namespace='swarm.artifact',
            register_handlers=False,receive_budget=knowledge.budget)
        self.engine=OwnSwarmSettings();self.settings=None;self.ready=False
        self.active=0;self.completed=0;self.lock=threading.RLock();self.changed=threading.Event()
        self.supervisor=ComputeSupervisor()
        counter=self.directory/'own-swarm-status.json'
        if counter.exists():
            if counter.stat().st_size>1024:raise ValueError('Own-swarm counter exceeds its bound')
            count=json.loads(counter.read_text())['completed']
            if type(count) is not int or not 0<=count<=10**12:raise ValueError('Own-swarm counter is invalid')
            self.completed=count
        for suffix,handler in [('describe.v1',self.transfer._describe),('read-chunk.v1',self.transfer._read_chunk),
                               ('read-chunk.v2',self.transfer._read_encoded_chunk)]:
            knowledge.work.register_handler('swarm.artifact.'+suffix,'artifact',self._guard(handler),
                permit_cached=self._artifact_cached)
        knowledge.work.register_handler(GRADIENT_OPERATION,'training',self.gradient,permit_cached=self.permit_cached)
        knowledge.work.register_handler(CANDIDATE_OPERATION,'training',self.candidate,
            permit_cached=lambda execution,result:self.permit_cached(execution,result,candidate=True))
        knowledge.work.register_handler(NETWORK_GRADIENT_OPERATION,'training',self.gradient,permit_cached=self.permit_cached)
        knowledge.work.register_handler(NETWORK_CANDIDATE_OPERATION,'training',self.candidate,
            permit_cached=lambda execution,result:self.permit_cached(execution,result,candidate=True))
        knowledge.work.register_handler(FOOD_GRADIENT_OPERATION,'training',self.gradient,permit_cached=self.permit_cached)
        knowledge.work.register_handler(FOOD_CANDIDATE_OPERATION,'training',self.candidate,
            permit_cached=lambda execution,result:self.permit_cached(execution,result,candidate=True))
        from .own_programs import ADMIT_OPERATION,admit
        knowledge.work.register_handler(ADMIT_OPERATION,'artifact',lambda payload,execution:admit(self,payload,execution),
            permit_cached=lambda *_:False)

    def _artifact_cached(self,execution,result):
        if not self.knowledge.allowed:return False
        if 'data' in result:
            self.knowledge.budget.wait(len(result['data'])+2048,
                cancel_event=execution.cancel_event,deadline=execution.deadline)
        return self.knowledge.allowed

    def _guard(self,handler):
        def guarded(payload,execution):
            if not self.knowledge.allowed:raise ValueError('Own public storage is paused')
            result=handler(payload,execution)
            if 'data' in result:
                self.knowledge.budget.wait(len(result['data'])+2048,
                    cancel_event=execution.cancel_event,deadline=execution.deadline)
            if not self.knowledge.allowed:raise ValueError('Own public storage permission was withdrawn')
            return result
        return guarded

    def update(self,settings):
        self.settings=settings
        try:
            from .own_programs import recover_selection
            recover_selection(self.directory)
            self.engine=OwnSwarmSettings.load(self.directory)
        except (OSError,ValueError,TypeError):self.engine=OwnSwarmSettings()
        if self.engine.enabled and self.knowledge.allowed and settings.connected and not settings.paused:
            # Older installations hold the pinned public data in the language
            # CAS. Publish verified copies through the own namespace as well,
            # so a new own-only participant has no dependency on that expert.
            for identifier,size in ARTIFACTS.values():
                source=self.directory/'public/language/artifacts/objects'/identifier[:2]/identifier[2:]
                if (not self.store.has(identifier) and source.is_file()
                        and managed_public_bytes(self.directory)+2*size<=settings.policy.max_disk_gb*1024**3):
                    try:
                        if digest_file(source)==(identifier,size):self.store.import_file(source,kind='dataset')
                    except (OSError,ValueError,RuntimeError):pass
        self.ready=bool(self.engine.enabled and self.knowledge.allowed and settings.connected
            and not settings.paused and settings.policy.mode in {'balanced','maximum'}
            and settings.policy.max_ram_gb>=definition()['ram_gb'] and settings.policy.cpu_percent>0
            and all(Path(v).is_file() for v in self.engine.command)
            and self.store.has(self.engine.model_id) and self.store.has(self.engine.optimizer_id)
            and self.training_path().is_file() and self.evaluation_path().is_file())
        if self.engine.version==3:
            from .own_programs import supports_programs
            self.ready=self.ready and supports_programs(self) and public_data_path(
                self.directory,definition(self.engine.policy_id())['retention_dataset_id']).is_file()

    def training_path(self):
        identifier=definition(self.engine.policy_id())['training_dataset_id']
        return public_data_path(self.directory,identifier)

    def evaluation_path(self):
        identifier=definition(self.engine.policy_id())['evaluation_dataset_id']
        return public_data_path(self.directory,identifier)

    def snapshot(self):
        return {'ready':self.ready,'active':self.active,'completed':self.completed,
            'model_id':self.engine.model_id if self.engine.enabled else '',
            'state_id':state_id(self.engine.model_id,self.engine.optimizer_id,self.engine.step) if self.engine.enabled else '',
            'optimizer_step':self.engine.step if self.engine.enabled else 0,'training_policy_id':self.engine.policy_id(),
            'family':definition()['family'],'device':'cpu','membership_version':self.engine.version,
            'program_id':self.engine.program_id,
            'cpu_threads':max(1,min(8,int(self.resources['cpu_cores']*self.settings.policy.cpu_percent/100))) if self.settings else 1,
            'requesters':list(self.engine.requesters)}

    def permit_cached(self,execution,result,*,candidate=False):
        try:
            own=self.knowledge.node.identity.node_id_hex
            return (self.ready and self.knowledge.allowed and execution.origin_node_id in self.engine.requesters
                and execution.payload['state_id']==self.snapshot()['state_id']
                and execution.payload['model_id']==self.engine.model_id
                and result['training_policy_id']==self.engine.policy_id()
                and (not candidate or execution.origin_node_id==own))
        except (ValueError,TypeError,KeyError):return False

    def gradient(self,payload,execution):return self._run(payload,execution,candidate=False)
    def candidate(self,payload,execution):return self._run(payload,execution,candidate=True)

    def _run(self,payload,execution,*,candidate):
        validate_payload(payload,candidate=candidate)
        execution.raise_if_cancelled();p=definition(payload['training_policy_id']);own=self.knowledge.node.identity.node_id_hex
        if (not self.ready or not self.knowledge.allowed or execution.origin_node_id not in self.engine.requesters
                or payload['training_policy_id']!=self.engine.policy_id()
                or candidate and execution.origin_node_id!=own
                or payload['model_id']!=self.engine.model_id or payload['state_id']!=self.snapshot()['state_id']
                or execution.resource_requirements().gpu_required
                or execution.resource_requirements().ram_gb<p['ram_gb']
                or execution.resource_requirements().disk_gb<(p['candidate_disk_gb'] if candidate else p['gradient_disk_gb'])):
            raise ValueError('Own-swarm state, permission or resource lease is unavailable')
        reservation=p['candidate_disk_gb'] if candidate else p['gradient_disk_gb']
        if managed_public_bytes(self.directory)+reservation*1024**3>self.settings.policy.max_disk_gb*1024**3:
            raise ValueError('Own-swarm computation exceeds current public storage permission')
        outputs=(verify_contributions(payload['contributions'],model_id=self.engine.model_id,
            training_state_id=payload['state_id'],requester_id=own,training_policy_id=payload['training_policy_id']) if candidate else [])
        selected=asdict(self.engine);permission=asdict(self.settings.policy)
        job=self.root/'jobs'/uuid.uuid4().hex;job.mkdir(parents=True,mode=0o700)
        request={**payload,'action':'own-candidate' if candidate else 'own-gradient',
            'optimizer_id':self.engine.optimizer_id,'step':self.engine.step,
            'model_file':str(self.store.object_path(self.engine.model_id)),
            'training_file':str(self.training_path()),'output_directory':str(job/'engine'),
            'device':'cpu','cpu_threads':max(1,min(8,int(self.resources['cpu_cores']*self.settings.policy.cpu_percent/100))),
            'deadline':min(execution.deadline,__import__('time').time()+p['maximum_seconds'])}
        if candidate:
            request.update(optimizer_file=str(self.store.object_path(self.engine.optimizer_id)),
                evaluation_file=str(self.evaluation_path()),requester_id=own,
                gradient_files={v['gradient_id']:str(self.store.object_path(v['gradient_id'])) for v in outputs})
            if p['version']==3:request['retention_file']=str(public_data_path(self.directory,p['retention_dataset_id']))
        def allowed():
            return (self.ready and self.knowledge.allowed and asdict(self.engine)==selected
                    and asdict(self.settings.policy)==permission)
        with self.lock:self.active+=1
        self.changed.set()
        try:
            result=self.supervisor.run(self.engine.command,request,execution,allowed=allowed,ram_gb=p['ram_gb'])
            if not allowed():raise ComputeProcessError('Own training state or permissions changed during computation')
            if any(result.get(k)!=payload[k] for k in ['state_id','model_id','training_policy_id']):
                raise ValueError('Own engine returned work for another selected state')
            items=[('gradient','gradient')] if not candidate else [('evaluation','evaluation')]
            if candidate:
                if result['accepted']!=verdict(result['before'],result['after'],payload['training_policy_id']):
                    raise ValueError('Own candidate verdict does not match its declared gate')
                if result['previous_optimizer_step']!=self.engine.step or result['candidate_optimizer_step']!=self.engine.step+1:
                    raise ValueError('Own candidate lost optimizer continuity')
                if result['accepted']:items += [('candidate_model','model'),('candidate_optimizer','other')]
            for prefix,kind in items:
                path=Path(result.pop(prefix+'_file'))
                if not path.resolve().is_relative_to((job/'engine').resolve()):raise ValueError('Own child output escaped its job')
                item=self.store.import_file(path,kind=kind)
                if (item.artifact_id,item.size)!=(result[prefix+'_id'],result[prefix+'_bytes']):
                    raise ValueError('Own child artifact content differs from its report')
            with self.lock:
                self.completed+=1;atomic_json(self.directory/'own-swarm-status.json',{'completed':self.completed})
            return result
        finally:
            error=job/'engine/failure.json'
            if error.is_file() and error.stat().st_size<=16*1024:
                failure=self.root/'failures'/job.name;failure.parent.mkdir(exist_ok=True)
                shutil.copy2(error,failure)
            shutil.rmtree(job)
            with self.lock:self.active-=1
            self.changed.set()
