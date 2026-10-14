"""Known-requester public LoRA gradients under the physical node's permissions."""
import json
import math
import threading
from dataclasses import dataclass,asdict,field
from pathlib import Path

from src.learning.public_conversation import public_corpus,corpus_digest
from src.learning.public_bases import base_hash
from src.release.language_family import ADAPTERS
from .compute_process import ComputeSupervisor,ComputeProcessError
from .artifact_store import digest_file
from .telemetry import atomic_json

OPERATION='language.gradient.v1'


@dataclass
class LanguageTrainingSettings:
    version:int=1
    enabled:bool=False
    command:list=field(default_factory=list)
    base_directory:str=''
    adapter_file:str=''
    requesters:list=field(default_factory=list)

    def validate(self):
        if type(self.version) is not int or self.version!=1 or type(self.enabled) is not bool:raise ValueError('Invalid language training controls')
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

    def _permit_cached(self,execution,result):
        return bool(self.ready and self.knowledge.allowed
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
            and policy.max_ram_gb>=4 and policy.gpu_percent>0 and policy.max_gpu_jobs>0 and self.resources['has_gpu']
            and self.resources['gpu_vram_gb']*policy.gpu_percent/100>=2
            and all(Path(x).is_file() for x in self.engine.command) and Path(self.engine.base_directory).is_dir() and Path(self.engine.adapter_file).is_file())

    def snapshot(self):return {'ready':self.ready,'active':self.active,'completed':self.completed,'model_id':ADAPTERS['qwen3-0.6b'] if self.ready else ''}

    def gradient(self,payload,execution):
        if not isinstance(payload,dict) or set(payload)!={'adapter_id','dataset_id','start','count'}:raise ValueError('Invalid language gradient payload')
        corpus=public_corpus()
        if payload['adapter_id']!=ADAPTERS['qwen3-0.6b'] or payload['dataset_id']!=corpus_digest(corpus):raise ValueError('Only the pinned public model/corpus may train')
        start,count=payload['start'],payload['count']
        if type(start) is not int or type(count) is not int or not 1<=count<=8 or not 0<=start<len(corpus['train']) or start+count>len(corpus['train']):raise ValueError('Invalid disjoint language shard')
        requirements=execution.resource_requirements()
        if not self.ready or execution.work.origin_node_id not in self.engine.requesters or not requirements.gpu_required or requirements.ram_gb<4:
            raise ComputeProcessError('Public language gradient requires known requester, GPU and RAM permission')
        policy=asdict(self.settings.policy);engine=asdict(self.engine)
        path=self.families.store.incoming/f'{execution.work.job_id}.gradient'
        request={'base_name':'qwen3-0.6b','base_directory':self.engine.base_directory,'adapter_file':self.engine.adapter_file,
            **payload,'output_file':str(path),'cpu_threads':max(1,min(4,int(self.resources['cpu_cores']*self.settings.policy.cpu_percent/100))),
            'gpu_fraction':self.settings.policy.gpu_percent/100/max(1,self.settings.policy.max_gpu_jobs),'device':'cuda','deadline':execution.deadline}
        with self.lock:self.active+=1
        self.changed.set()
        try:
            result=self.supervisor.run(self.engine.command,request,execution,ram_gb=requirements.ram_gb,
                allowed=lambda:self.ready and asdict(self.settings.policy)==policy and asdict(self.engine)==engine)
            expected={'base_hash','adapter_id','dataset_id','start','count','tokens','loss','retention_loss','gradient_id','gradient_bytes'}
            if set(result)!=expected or result['base_hash']!=base_hash('qwen3-0.6b') or any(result[k]!=payload[k] for k in payload) or type(result['tokens']) is not int or not 1<=result['tokens']<=1536 or any(type(result[k]) not in (int,float) or not math.isfinite(result[k]) for k in ['loss','retention_loss']):raise ValueError('Invalid language gradient result')
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
