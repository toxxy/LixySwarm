"""Known-requester public LoRA gradients under the physical node's permissions."""
import json
import math
import threading
from dataclasses import dataclass,asdict,field
from pathlib import Path

from src.learning.public_conversation import public_corpus,corpus_digest
from src.learning.public_bases import base_hash
from src.learning.public_bases import BASES
from src.learning.language_policy import policy_id,policy_from_id,descriptor_policy,HALF_POLICY,limits,gradient_operation,candidate_operation,enabled_policy
from src.release.language_family import ADAPTERS
from .compute_process import ComputeSupervisor,ComputeProcessError
from .artifact_store import digest_file
from .telemetry import atomic_json

OPERATION='language.gradient.v1'
VERSIONED_OPERATION='language.gradient.v2'
BATCH_OPERATION='language.gradient.v3'
HALF_OPERATION='language.gradient.v4'


@dataclass
class LanguageTrainingSettings:
    version:int=2
    enabled:bool=False
    command:list=field(default_factory=list)
    base_directory:str=''
    adapter_file:str=''
    requesters:list=field(default_factory=list)
    base_name:str='qwen3-0.6b'
    candidates_enabled:bool=False
    batching_enabled:bool=False
    half_precision_enabled:bool=False

    def validate(self):
        if type(self.version) is not int or self.version!=2 or type(self.enabled) is not bool or type(self.candidates_enabled) is not bool or type(self.batching_enabled) is not bool or type(self.half_precision_enabled) is not bool or self.base_name not in BASES:raise ValueError('Invalid language training controls')
        if not isinstance(self.command,list) or len(self.command)>4 or any(not isinstance(x,str) or not x or len(x)>4096 for x in self.command):raise ValueError('Invalid local gradient command')
        if not isinstance(self.requesters,list) or len(self.requesters)>8 or any(not isinstance(x,str) or len(x)!=64 or any(c not in '0123456789abcdef' for c in x) for x in self.requesters):raise ValueError('Invalid known language requesters')
        if self.enabled and (not self.command or not self.requesters or any(not Path(x).is_absolute() for x in [self.command[0],self.base_directory,self.adapter_file])):raise ValueError('Public language training requires locally approved inputs/requesters')

    def save(self,directory):self.validate();atomic_json(Path(directory)/'language-training.json',asdict(self))

    @classmethod
    def load(cls,directory):
        path=Path(directory)/'language-training.json'
        if not path.exists():return cls()
        if path.stat().st_size>16*1024:raise ValueError('Language training controls exceed their bound')
        value=json.loads(path.read_text())
        if isinstance(value,dict) and 'half_precision_enabled' not in value:value={**value,'half_precision_enabled':False}
        if isinstance(value,dict) and set(value)==set(asdict(cls()))-{'base_name','candidates_enabled','batching_enabled'} and value.get('version')==1:
            value={**value,'version':2,'base_name':'qwen3-0.6b'}
        if isinstance(value,dict) and set(value)==set(asdict(cls()))-{'candidates_enabled','batching_enabled'} and value.get('version')==2:
            value={**value,'candidates_enabled':False}
        if isinstance(value,dict) and set(value)==set(asdict(cls()))-{'batching_enabled'} and value.get('version')==2:
            value={**value,'batching_enabled':False}
        if not isinstance(value,dict) or set(value)!=set(asdict(cls())):raise ValueError('Invalid language training settings schema')
        result=cls(**value);result.validate();return result


class LanguageTraining:
    def __init__(self,knowledge,families,directory,resources):
        self.knowledge,self.families,self.directory,self.resources=knowledge,families,Path(directory),resources
        self.engine=LanguageTrainingSettings();self.settings=None;self.ready=False;self.active=0;self.completed=0
        self.changed=threading.Event();self.lock=threading.RLock();self.supervisor=ComputeSupervisor()
        try:
            path=self.directory/'language-training-status.json'
            if path.stat().st_size>1024:raise ValueError('Language training counter exceeds its bound')
            count=json.loads(path.read_text())['completed']
            if type(count) is not int or not 0<=count<=10**12:raise ValueError('Invalid language training counter')
            self.completed=count
        except (OSError,ValueError,TypeError,KeyError):pass
        knowledge.work.register_handler(OPERATION,'training',self.gradient,permit_cached=self._permit_cached)
        knowledge.work.register_handler(VERSIONED_OPERATION,'training',self.versioned_gradient,permit_cached=self._permit_versioned_cached)
        knowledge.work.register_handler(BATCH_OPERATION,'training',self.batched_gradient,permit_cached=self._permit_versioned_cached)
        knowledge.work.register_handler(HALF_OPERATION,'training',self.half_gradient,permit_cached=self._permit_versioned_cached)
        from .language_candidate import OPERATION as candidate_operation,run_candidate
        knowledge.work.register_handler(candidate_operation,'training',lambda payload,execution:run_candidate(self,payload,execution),
            permit_cached=lambda execution,result:self._permit_candidate_cached(execution,result))
        knowledge.work.register_handler('language.candidate.v2','training',lambda payload,execution:run_candidate(self,payload,execution,policy_version=2),
            permit_cached=lambda execution,result:self._permit_candidate_cached(execution,result))
        knowledge.work.register_handler('language.candidate.v3','training',lambda payload,execution:run_candidate(self,payload,execution,policy_version=3),
            permit_cached=lambda execution,result:self._permit_candidate_cached(execution,result))

    def policy_enabled(self,version):
        return enabled_policy(self.engine,version)

    def _permit_candidate_cached(self,execution,result):
        if not self.ready or not self.engine.candidates_enabled or not self.knowledge.allowed or execution.work.origin_node_id not in self.engine.requesters:return False
        try:
            version=policy_from_id(execution.work.payload.get('training_policy_id'))
            operation=getattr(execution,'operation',getattr(execution.work,'operation',None))
            return (operation in {None,candidate_operation(version)} and self.policy_enabled(version)
            and result['summary']['parent_release_id']==self._head()['release']['release_id']
            and result['summary'].get('training_policy_id')==execution.work.payload.get('training_policy_id')
            and result['summary'].get('training_policy_id')==policy_id(version))
        except (OSError,ValueError,TypeError,KeyError):return False

    def _head(self):
        from src.release.language_versions import LanguageVersionRegistry
        return LanguageVersionRegistry(self.families.root,self.engine.base_name).head()

    def _permit_versioned_cached(self,execution,result):
        if not self.ready or not self.knowledge.allowed or execution.work.origin_node_id not in self.engine.requesters:return False
        try:
            head=self._head()
            version=policy_from_id(execution.work.payload.get('training_policy_id'))
            operation=getattr(execution,'operation',getattr(execution.work,'operation',None))
            return (operation in {None,gradient_operation(version)} and self.policy_enabled(version)
                and execution.work.payload.get('release_id')==head['release']['release_id']
                and result.get('training_policy_id')==execution.work.payload.get('training_policy_id')
                and result.get('training_policy_id')==policy_id(version)
                and execution.work.payload.get('base_name')==self.engine.base_name
                and result.get('adapter_id')==head['release']['model_artifact_id'] and result.get('dataset_id')==corpus_digest(public_corpus()))
        except (OSError,ValueError,TypeError):return False

    def _permit_cached(self,execution,result):
        return bool(self.ready and self.knowledge.allowed and self.engine.base_name=='qwen3-0.6b'
            and execution.work.origin_node_id in self.engine.requesters
            and result.get('adapter_id')==ADAPTERS['qwen3-0.6b']
            and result.get('dataset_id')==corpus_digest(public_corpus()))

    def update(self,settings):
        self.settings=settings
        try:self.engine=LanguageTrainingSettings.load(self.directory)
        except (OSError,ValueError,TypeError):self.engine=LanguageTrainingSettings()
        policy=settings.policy
        self.ready=bool(self.engine.enabled and settings.connected and not settings.paused and self.knowledge.allowed
            and policy.mode in {'balanced','maximum'} and policy.cpu_percent>0 and policy.max_cpu_jobs>0
            and policy.max_ram_gb>=self.required_ram() and policy.gpu_percent>0 and policy.max_gpu_jobs>0 and self.resources['has_gpu']
            and self.resources['gpu_vram_gb']*policy.gpu_percent/100/max(1,policy.max_gpu_jobs)>=(6 if self.engine.base_name=='qwen3-1.7b' else 2)
            and all(Path(x).is_file() for x in self.engine.command) and Path(self.engine.base_directory).is_dir() and Path(self.engine.adapter_file).is_file())
        node=getattr(self.knowledge,'node',None)
        if node is not None:
            node.capabilities['language_gradient_batches_v1']=bool(self.ready and self.engine.batching_enabled)
            node.capabilities['language_gradient_half_v1']=bool(self.ready and self.policy_enabled(3))

    def required_ram(self):return 6 if self.engine.base_name=='qwen3-1.7b' else 4

    def snapshot(self):
        try:model=self._head()['release']['model_artifact_id'] if self.ready else ''
        except (OSError,ValueError,TypeError):model=''
        return {'ready':self.ready,'active':self.active,'completed':self.completed,'model_id':model,'base_name':self.engine.base_name,'batching_enabled':self.engine.batching_enabled,'half_precision_enabled':self.engine.half_precision_enabled}

    def versioned_gradient(self,payload,execution):
        return self._versioned_gradient(payload,execution,policy_version=1)

    def batched_gradient(self,payload,execution):
        if not self.engine.batching_enabled:raise ComputeProcessError('Public batch training is not locally enabled')
        return self._versioned_gradient(payload,execution,policy_version=2)

    def half_gradient(self,payload,execution):
        if not self.policy_enabled(3):raise ComputeProcessError('Public half-precision training is not locally enabled')
        return self._versioned_gradient(payload,execution,policy_version=3)

    def _versioned_gradient(self,payload,execution,*,policy_version):
        fields={'base_name','release_id','adapter_id','dataset_id','start','count'}
        if policy_version>=2:fields.add('training_policy_id')
        if not isinstance(payload,dict) or set(payload)!=fields or policy_version>=2 and payload.get('training_policy_id')!=policy_id(policy_version):
            raise ValueError('Invalid versioned language gradient payload')
        if (not self.ready or execution.work.origin_node_id not in self.engine.requesters
                or payload['base_name']!=self.engine.base_name):
            raise ComputeProcessError('Language family or requester is not locally approved')
        corpus=public_corpus();start,count=payload['start'],payload['count']
        if (payload['dataset_id']!=corpus_digest(corpus) or type(start) is not int or type(count) is not int
                or not 1<=count<=limits(policy_version)[0] or not 0<=start<=len(corpus['train'])-count):
            raise ValueError('Invalid public language data or shard')
        requirements=execution.resource_requirements()
        if not requirements.gpu_required or requirements.ram_gb<self.required_ram():
            raise ComputeProcessError('Language gradient requires its full GPU/RAM lease')
        head=self._head()
        if head.get('descriptor') and descriptor_policy(head['descriptor'])>policy_version:
            raise ComputeProcessError('This accepted parent requires the batch training policy')
        if payload['release_id']!=head['release']['release_id'] or payload['adapter_id']!=head['release']['model_artifact_id']:
            raise ValueError('Language gradient is from a stale or unverified version')
        from .language_install import materialize_version
        fingerprint=asdict(self.engine);policy=asdict(self.settings.policy)
        allowed=lambda:self.ready and asdict(self.engine)==fingerprint and asdict(self.settings.policy)==policy and self._head()['release']['release_id']==payload['release_id']
        base=materialize_version(self.directory,self.engine.base_name,payload['release_id'],allowed=allowed)
        compact={k:payload[k] for k in ('adapter_id','dataset_id','start','count')}
        return self._gradient(compact,execution,base_name=self.engine.base_name,base_directory=base,
            adapter_file=base/'adapter.safetensors',expected_adapter=payload['adapter_id'],still_current=allowed,policy_version=policy_version)

    def gradient(self,payload,execution):
        if self.engine.base_name!='qwen3-0.6b':raise ValueError('Legacy gradients are restricted to the original small family')
        return self._gradient(payload,execution,base_name='qwen3-0.6b',base_directory=self.engine.base_directory,
            adapter_file=self.engine.adapter_file,expected_adapter=ADAPTERS['qwen3-0.6b'])

    def _gradient(self,payload,execution,*,base_name,base_directory,adapter_file,expected_adapter,still_current=lambda:True,policy_version=1):
        if not isinstance(payload,dict) or set(payload)!={'adapter_id','dataset_id','start','count'}:raise ValueError('Invalid language gradient payload')
        corpus=public_corpus()
        if payload['adapter_id']!=expected_adapter or payload['dataset_id']!=corpus_digest(corpus):raise ValueError('Only the pinned public model/corpus may train')
        start,count=payload['start'],payload['count']
        if type(start) is not int or type(count) is not int or not 1<=count<=limits(policy_version)[0] or not 0<=start<len(corpus['train']) or start+count>len(corpus['train']):raise ValueError('Invalid disjoint language shard')
        requirements=execution.resource_requirements()
        if not self.ready or execution.work.origin_node_id not in self.engine.requesters or not requirements.gpu_required or requirements.ram_gb<self.required_ram():
            raise ComputeProcessError('Public language gradient requires known requester, GPU and RAM permission')
        policy=asdict(self.settings.policy);engine=asdict(self.engine)
        path=self.families.store.incoming/f'{execution.work.job_id}.gradient'
        request={'action':'gradient','base_name':base_name,'base_directory':str(base_directory),'adapter_file':str(adapter_file),
            **payload,'output_file':str(path),'cpu_threads':max(1,min(4,int(self.resources['cpu_cores']*self.settings.policy.cpu_percent/100))),
            'gpu_fraction':self.settings.policy.gpu_percent/100/max(1,self.settings.policy.max_gpu_jobs),'device':'cuda','deadline':execution.deadline}
        if policy_version>=2:request.update(action='gradient-half' if policy_version==3 else 'gradient-batch',training_policy_id=policy_id(policy_version))
        with self.lock:self.active+=1
        self.changed.set()
        try:
            result=self.supervisor.run(self.engine.command,request,execution,ram_gb=requirements.ram_gb,
                allowed=lambda:self.ready and asdict(self.settings.policy)==policy and asdict(self.engine)==engine and still_current())
            expected={'base_hash','adapter_id','dataset_id','start','count','tokens','loss','retention_loss','gradient_id','gradient_bytes'}
            if policy_version>=2:expected.add('training_policy_id')
            if policy_version==3:expected.add('quantization_relative_l2_error')
            if set(result)!=expected or policy_version>=2 and result.get('training_policy_id')!=policy_id(policy_version) or result['base_hash']!=base_hash(base_name) or any(result[k]!=payload[k] for k in payload) or type(result['tokens']) is not int or not 1<=result['tokens']<=limits(policy_version)[1] or any(type(result[k]) not in (int,float) or not math.isfinite(result[k]) for k in ['loss','retention_loss']):raise ValueError('Invalid language gradient result')
            if policy_version>=2 and result['tokens']>192*count:raise ValueError('Batch token count exceeds its distinct examples')
            if policy_version==3:
                error=result['quantization_relative_l2_error']
                if type(error) not in (int,float) or not math.isfinite(error) or not 0<=error<=HALF_POLICY['maximum_relative_l2_error']:raise ValueError('Half-precision gradient exceeds its reconstruction bound')
            if not still_current():raise ComputeProcessError('Language version changed before gradient commit')
            identifier,size=digest_file(path)
            if size>32*1024**2 or identifier!=result['gradient_id'] or type(result['gradient_bytes']) is not int or size!=result['gradient_bytes']:
                raise ValueError('Language gradient artifact identity mismatch')
            manifest=self.families.store.commit_generated(path,kind='gradient',media_type='application/octet-stream')
            with self.lock:
                self.completed+=1;atomic_json(self.directory/'language-training-status.json',{'completed':self.completed})
            return {**result,'gradient':manifest.to_dict()}
        finally:
            path.unlink(missing_ok=True)
            with self.lock:self.active-=1
            self.changed.set()
