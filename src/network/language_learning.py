"""Bounded language rounds on an existing physical node, with durable progress.

Only explicit local controls select the known workers and publisher. Compute
children receive public artifacts and never the signing key or this ledger.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict,dataclass,field
from pathlib import Path

from src.contribution import ResourceRequirements
from src.learning.public_bases import BASES
from src.learning.public_conversation import public_corpus,corpus_digest
from src.learning.language_capacity import CapacityHistory,allocate_shards,observation
from src.learning.language_policy import BATCH_POLICY_ID,limits
from src.release.language_family import SIGNER
from src.release.language_publish import publish_update
from src.release.language_versions import LanguageVersionRegistry,WORKERS,verify_contributions,verify_evaluation
from .language_version_service import fetch_history
from .lsp import LSPIdentity
from .telemetry import atomic_json
from .work_protocol import ResultReceipt,WorkResult


@dataclass
class LanguageLearningSettings:
    version:int=3
    enabled:bool=False
    cycles:int=1
    workers:list[str]=field(default_factory=list)
    base_name:str='qwen3-1.7b'
    examples_per_worker:int=8
    learning_rate:float=.001
    signing_key_file:str=''
    stop_after_rejections:int=3
    generation:int=0
    shards_per_cycle:int=4
    allocation:str='observed'
    training_policy_version:int=1

    def validate(self):
        maximum,_=limits(self.training_policy_version)
        if (type(self.version) is not int or self.version!=3 or type(self.enabled) is not bool
                or type(self.cycles) is not int or not 1<=self.cycles<=10
                or self.base_name not in BASES or type(self.examples_per_worker) is not int
                or not 1<=self.examples_per_worker<=maximum or type(self.learning_rate) not in (int,float)
                or not 1e-6<=self.learning_rate<=1e-3 or type(self.stop_after_rejections) is not int
                or not 1<=self.stop_after_rejections<=10 or type(self.generation) is not int or not 0<=self.generation<=10**9
                or type(self.shards_per_cycle) is not int or not 2<=self.shards_per_cycle<=16
                or self.allocation not in {'balanced','observed'}):
            raise ValueError('Invalid bounded language learning settings')
        if (not isinstance(self.workers,list) or len(self.workers)>2 or len(set(self.workers))!=len(self.workers)
                or any(worker not in WORKERS for worker in self.workers)):
            raise ValueError('Language coordination is limited to the known physical pilot')
        if self.enabled and (set(self.workers)!=WORKERS or not Path(self.signing_key_file).is_absolute()):
            raise ValueError('Learning needs both physical workers and a local publisher')

    def save(self,directory):
        self.validate();atomic_json(Path(directory)/'language-learning-controls.json',asdict(self))

    @classmethod
    def load(cls,directory):
        path=Path(directory)/'language-learning-controls.json'
        if not path.exists():return cls()
        if path.stat().st_size>8192:raise ValueError('Language learning controls exceed their bound')
        value=json.loads(path.read_text())
        if (isinstance(value,dict) and type(value.get('version')) is int and value['version']==1
                and set(value)==set(asdict(cls()))-{'shards_per_cycle','allocation','training_policy_version'}):
            value={**value,'version':2,'shards_per_cycle':2,'allocation':'balanced'}
        if (isinstance(value,dict) and type(value.get('version')) is int and value['version']==2
                and set(value)==set(asdict(cls()))-{'training_policy_version'}):
            value={**value,'version':3,'training_policy_version':1}
        if not isinstance(value,dict) or set(value)!=set(asdict(cls())):raise ValueError('Invalid language learning schema')
        result=cls(**value);result.validate();return result


class LanguageLearningCoordinator:
    def __init__(self,knowledge,training,directory):
        self.knowledge,self.training,self.directory=knowledge,training,Path(directory)
        self.root=self.directory/'language-learning';self.path=self.root/'ledger.json'
        self.lock=threading.RLock();self.changed=threading.Event();self.cancel=threading.Event()
        self.settings=LanguageLearningSettings();self.thread=None;self.closed=False;self.next_attempt=0
        self.ledger=None
        self.capacity=CapacityHistory(self.root/'capacity.json');self.calculation_slots={}
        if self.path.exists():
            if self.path.stat().st_size>512*1024:raise ValueError('Language learning ledger exceeds its bound')
            self.ledger=json.loads(self.path.read_text())
            if (self.ledger.get('version')!=1 or type(self.ledger.get('round')) is not int
                    or not 0<=self.ledger['round']<=10 or type(self.ledger.get('cursor')) is not int
                    or not 0<=self.ledger['cursor']<len(public_corpus()['train'])
                    or self.ledger.get('dataset_id')!=corpus_digest(public_corpus())):
                raise ValueError('Invalid durable language progress; preserve it for inspection')

    def _registry(self):
        return LanguageVersionRegistry(self.training.families.root,self.settings.base_name)

    def _save(self,**values):
        with self.lock:
            self.ledger.update(values);atomic_json(self.path,self.ledger);self.changed.set()

    def snapshot(self):
        with self.lock:
            try:required=2 if self._registry().head()['descriptor']['version']==3 else 1
            except (OSError,ValueError):required=1
            if self.ledger is None:return {'status':'idle','round':0,'cursor':0,'epoch':0,'enabled':self.settings.enabled,
                'required_training_policy_version':required,'training_policy_version':self.settings.training_policy_version}
            return {**{key:self.ledger.get(key) for key in ['status','round','cursor','epoch','accepted_sequence','error']},
                'enabled':self.settings.enabled,'paused':self.cancel.is_set() and self.ledger.get('pending') is not None,
                'required_training_policy_version':required,'training_policy_version':self.settings.training_policy_version}

    def update(self):
        try:self.settings=LanguageLearningSettings.load(self.directory)
        except (OSError,ValueError,TypeError):self.settings=LanguageLearningSettings()
        allowed=(not self.closed and self.settings.enabled and self.training.ready
            and self.training.engine.candidates_enabled and self.knowledge.allowed
            and self.training.engine.base_name==self.settings.base_name
            and self.knowledge.node.identity.node_id_hex in self.settings.workers)
        if not allowed:self.cancel.set();return
        if self.thread is not None and self.thread.is_alive():return
        if self.ledger is not None:
            if self.ledger.get('generation')!=self.settings.generation:
                pending=deepcopy(self.ledger['pending'])
                if pending is not None:pending['attempts']=0
                self._save(generation=self.settings.generation,rejections=0,pending=pending,error=None)
            if self.ledger['round']>=self.settings.cycles:
                if self.ledger['status']!='complete':self._save(status='complete')
                return
            if self.ledger.get('rejections',0)>=self.settings.stop_after_rejections:
                if self.ledger['status']!='no_improvement':self._save(status='no_improvement')
                return
            if self.ledger.get('pending') and self.ledger['pending'].get('attempts',0)>=3:
                if self.ledger['status']!='needs_attention':self._save(status='needs_attention')
                return
        if time.monotonic()<self.next_attempt:return
        self.cancel=threading.Event()
        self.thread=threading.Thread(target=self._run,args=(self.cancel,),name='lixy-language-learning',daemon=True)
        self.thread.start()

    def _initialize(self):
        if self.ledger is not None:
            if self.ledger['base_name']!=self.settings.base_name:raise ValueError('Durable cursor belongs to another language family')
            return
        registry=self._registry();head=registry.head();cursor=0
        if head['release']['sequence']:
            chain=registry.history(head['release']['release_id'])
            path=self.training.families.store.object_path(head['descriptor']['evaluation_id'])
            report=json.loads(path.read_text());verify_evaluation(report,chain[-2],head)
            cursor=max(row['output']['start']+row['output']['count'] for row in report['contributions'])
        length=len(public_corpus()['train'])
        self.ledger={'version':1,'base_name':self.settings.base_name,'dataset_id':corpus_digest(public_corpus()),
            'round':0,'cursor':cursor%length,'epoch':cursor//length,'pending':None,'status':'idle',
            'accepted_sequence':head['release']['sequence'],'rejections':0,'generation':self.settings.generation,'history':[],'error':None}
        self._save()

    def _prepare_round(self):
        if self.ledger['pending'] is not None:return
        head=self._registry().head();start=self.ledger['cursor']
        if ((self.settings.training_policy_version==2 and not self.training.engine.batching_enabled)
                or head['descriptor']['version']==3 and self.settings.training_policy_version!=2):
            self._save(status='waiting_policy',error='The selected batch policy needs local batch-worker permission');return
        measured=None
        slots=self._slots()
        if any(count==0 for count in slots.values()):
            self._save(status='waiting_capacity',error=None);return
        if self.settings.allocation=='observed':
            try:measured=self.capacity.estimate(head,self.settings.workers,**({'policy_version':2} if self.settings.training_policy_version==2 else {}))
            except ValueError:pass
        plan=allocate_shards(self.settings.workers,self.settings.examples_per_worker,self.settings.shards_per_cycle,
            start=start,capacity=measured,slots=slots,policy_version=self.settings.training_policy_version)
        pending={'id':uuid.uuid4().hex,'head':head,'attempts':0,'contributions':{},'candidate':None,'bundle':None,
            'plan':plan,'end':plan[-1]['start']+plan[-1]['count'],'learning_rate':self.settings.learning_rate,
            'measurements':{},'capacity_used':measured,'training_policy_version':self.settings.training_policy_version}
        self._save(status='collecting',pending=pending,error=None)

    def _slots(self):
        own=self.knowledge.node.identity.node_id_hex
        policy=getattr(getattr(self.training,'settings',None),'policy',None)
        peers={peer['node_id']:peer.get('resources',{}) for peer in getattr(self.knowledge.node,'peers',lambda:[])()}
        batch_workers={peer['node_id'] for peer in getattr(self.knowledge.node,'peers',lambda:[])()
            if peer.get('capabilities',{}).get('language_gradient_batches_v1') is True}
        result={}
        for worker in self.settings.workers:
            count=getattr(policy,'max_gpu_jobs',1) if worker==own else peers.get(worker,{}).get('max_gpu_jobs',1)
            result[worker]=max(0,min(count,4)) if type(count) is int else 1
            available=peers.get(worker,{}).get('work',{})
            if worker!=own and isinstance(available,dict) and available.get('training') is False:result[worker]=0
            if worker!=own and self.settings.training_policy_version==2 and worker not in batch_workers:result[worker]=0
            if worker==own and self.settings.training_policy_version==2 and not self.training.engine.batching_enabled:result[worker]=0
        return result

    def _check(self,cancel,configuration):
        if (cancel.is_set() or self.closed or not self.training.ready or not self.training.engine.candidates_enabled or not self.knowledge.allowed
                or asdict(LanguageLearningSettings.load(self.directory))!=configuration):
            raise RuntimeError('Language learning paused or permission changed')

    def _request(self,operation,payload,worker,requirements,cancel,timeout):
        if worker==self.knowledge.node.identity.node_id_hex:
            result=self.knowledge.work.execute_local(operation,payload,requirements,timeout_s=timeout,cancel_event=cancel)
        else:
            if worker not in self.knowledge.node.connected_node_ids:self.knowledge.node.connect_node(worker,timeout=12)
            result=self.knowledge.work.submit(operation,payload,requirements,peer_id=worker,timeout_s=timeout,cancel_event=cancel)
        if result.status!='ok':raise RuntimeError('Language work failed: '+result.error)
        if not ResultReceipt.from_dict(result.receipt).verify(result,expected_worker_id=worker,
                expected_requester_id=self.knowledge.node.identity.node_id_hex):
            raise ValueError('Language work receipt does not bind the physical assignment')
        return result

    def _verify_saved(self,saved,assignment):
        head=self.ledger['pending']['head'];worker=assignment['worker_id']
        result=WorkResult.from_dict(saved);output=result.output
        verify_contributions([saved],base_name=self.settings.base_name,adapter_id=head['release']['model_artifact_id'],
            dataset_id=self.ledger['dataset_id'],require_distributed=False,policy_version=self.ledger['pending'].get('training_policy_version',1))
        if (not ResultReceipt.from_dict(result.receipt).verify(result,expected_worker_id=worker,
                expected_requester_id=self.knowledge.node.identity.node_id_hex)
                or output['start']!=assignment['start'] or output['count']!=assignment['count']):
            raise ValueError('Saved contribution does not match the durable shard assignment')
        return output

    def _collect(self,index,assignment,cancel,configuration):
        self._check(cancel,configuration)
        pending=self.ledger['pending'];head=pending['head'];worker=assignment['worker_id']
        with self.lock:saved=pending['contributions'].get(str(index))
        payload={'base_name':self.settings.base_name,'release_id':head['release']['release_id'],
            'adapter_id':head['release']['model_artifact_id'],'dataset_id':self.ledger['dataset_id'],
            'start':assignment['start'],'count':assignment['count']}
        policy_version=self.ledger['pending'].get('training_policy_version',1)
        if policy_version==2:payload['training_policy_id']=BATCH_POLICY_ID
        if saved is None:
            with self.calculation_slots.get(worker,threading.BoundedSemaphore(1)):
                self._check(cancel,configuration);began=time.monotonic()
                result=self._request('language.gradient.v3' if policy_version==2 else 'language.gradient.v2',payload,worker,
                    ResourceRequirements(kind='training',gpu_required=True,ram_gb=self.training.required_ram(),disk_gb=.04),cancel,240)
                ready_seconds=time.monotonic()-began
            saved=result.to_dict()
        else:ready_seconds=None
        output=self._verify_saved(saved,assignment)
        # Receipt survives even if fetching the artifact or the process fails.
        with self.lock:
            changed=deepcopy(self.ledger['pending']);changed['contributions'][str(index)]=saved
            if ready_seconds is not None:changed.setdefault('measurements',{})[str(index)]={'request_seconds':ready_seconds}
            self._save(pending=changed)
        store=self.training.families.store
        transfer_started=time.monotonic()
        if not store.has(output['gradient_id']):
            self.training.families.artifacts.fetch(output['gradient_id'],peer_id=worker,expected_size=output['gradient_bytes'],
                timeout_s=120,cancel_event=cancel)
        if store.manifest(output['gradient_id']).size!=output['gradient_bytes']:raise ValueError('Recovered gradient size changed')
        self._check(cancel,configuration)
        if ready_seconds is not None:
            self.capacity.record(observation(saved,base_name=self.settings.base_name,release_id=payload['release_id'],
                request_seconds=ready_seconds,transfer_seconds=time.monotonic()-transfer_started))
        return saved

    def _finish(self,accepted):
        pending=self.ledger['pending'];head=self._registry().head()
        expected=pending['bundle'] if accepted else pending['head']
        if head!=expected:raise ValueError('Language head changed before saving completed progress')
        outcome={'round':self.ledger['round'],'epoch':self.ledger['epoch'],'accepted':accepted,
            'cursor_before':self.ledger['cursor'],'cursor_after':pending['end'],'pending':pending,
            'accepted_release_id':head['release']['release_id'],'finished_at':time.time()}
        atomic_json(self.root/'cycles'/pending['id']/'outcome.json',outcome)
        length=len(public_corpus()['train']);summary={key:outcome[key] for key in outcome if key!='pending'}
        self._save(status='idle',round=self.ledger['round']+1,cursor=pending['end']%length,
            epoch=self.ledger['epoch']+pending['end']//length,pending=None,error=None,
            accepted_sequence=head['release']['sequence'],rejections=0 if accepted else self.ledger['rejections']+1,
            history=self.ledger['history']+[summary])

    def _recover_publication(self):
        pending=self.ledger['pending']
        if pending is None:return
        bundle=pending['bundle']
        registry=self._registry();head=registry.head()
        chain=registry.history(pending['head']['release']['release_id'])
        if head not in (pending['head'],bundle):raise ValueError('Pending publication conflicts with the accepted head')
        registry.install(chain+[bundle],self.training.families.store)
        self._finish(True)

    def _run(self,cancel):
        configuration=asdict(self.settings)
        try:
            self._check(cancel,configuration);self._initialize()
            if self.ledger['round']>=self.settings.cycles:
                self._save(status='complete');return
            self._prepare_round()
            if self.ledger['pending'] is None:return
            pending=deepcopy(self.ledger['pending'])
            signer=LSPIdentity.load(self.settings.signing_key_file)
            if signer is None or signer.node_id_hex!=SIGNER:raise ValueError('Language publisher is not locally authorized')
            if pending['bundle'] is not None:self._recover_publication();return
            if pending.get('training_policy_version',1)!=self.settings.training_policy_version:
                atomic_json(self.root/'cycles'/pending['id']/'superseded-policy.json',{'pending':pending,
                    'reason':'Locally selected training policy changed; replan without advancing corpus position'})
                self._save(pending=None,status='waiting_policy',error=None);return
            head=self._registry().head()
            if head!=pending['head']:
                atomic_json(self.root/'cycles'/pending['id']/'superseded.json',{'pending':pending,
                    'reason':'Accepted parent changed; replan the same corpus position without reusing old gradients','observed_head':head})
                self._save(pending=None,status='waiting_replicas',error=None);return
            for index,assignment in enumerate(pending['plan']):
                if str(index) in pending['contributions']:self._verify_saved(pending['contributions'][str(index)],assignment)
            for assignment in pending['plan']:
                worker=assignment['worker_id']
                if worker!=self.knowledge.node.identity.node_id_hex:
                    try:
                        if worker not in self.knowledge.node.connected_node_ids:self.knowledge.node.connect_node(worker,timeout=12)
                        remote=fetch_history(self.knowledge.work,self.settings.base_name,worker,cancel_event=cancel)
                    except (OSError,ValueError,RuntimeError):
                        self._save(status='waiting_replicas',error='Waiting for the physical worker language history');return
                    if remote[-1]!=head:
                        self._save(status='waiting_replicas',error='Waiting for both workers to acquire the same accepted version');return
            slots=self._slots()
            if any(slots[row['worker_id']]==0 and str(index) not in pending['contributions']
                    for index,row in enumerate(pending['plan'])):
                self._save(status='waiting_capacity',error=None);return
            pending['attempts']+=1;self._save(pending=pending,error=None)
            # A worker may receive several distinct shards; never exceed its
            # declared job cap, and retain the worker-side lease as authority.
            self._save(status='collecting')
            self.calculation_slots={worker:threading.BoundedSemaphore(max(1,count)) for worker,count in slots.items()}
            with ThreadPoolExecutor(max_workers=max(1,min(4,len(pending['plan']),sum(slots.values())))) as pool:
                futures=[pool.submit(self._collect,index,row,cancel,configuration) for index,row in enumerate(pending['plan'])]
                try:results=[future.result() for future in futures]
                except BaseException:cancel.set();raise
            self._check(cancel,configuration)
            candidate=self.ledger['pending']['candidate']
            if candidate is None:
                self._save(status='evaluating')
                payload={'base_name':self.settings.base_name,'release_id':head['release']['release_id'],
                    'adapter_id':head['release']['model_artifact_id'],'dataset_id':self.ledger['dataset_id'],
                    'contributions':results,'learning_rate':pending['learning_rate']}
                policy_version=pending.get('training_policy_version',1)
                if policy_version==2:payload['training_policy_id']=BATCH_POLICY_ID
                result=self._request('language.candidate.v2' if policy_version==2 else 'language.candidate.v1',payload,self.knowledge.node.identity.node_id_hex,
                    ResourceRequirements(kind='training',gpu_required=True,ram_gb=self.training.required_ram(),disk_gb=.1),cancel,900)
                candidate=result.to_dict();changed=deepcopy(self.ledger['pending']);changed['candidate']=candidate;self._save(pending=changed)
            self._check(cancel,configuration)
            result=WorkResult.from_dict(candidate)
            if not ResultReceipt.from_dict(result.receipt).verify(result,expected_worker_id=self.knowledge.node.identity.node_id_hex,
                    expected_requester_id=self.knowledge.node.identity.node_id_hex):raise ValueError('Saved candidate receipt is invalid')
            if not result.output['summary']['eligible_for_publication']:self._finish(False);return
            candidate_directory=self.directory/'language-candidates'/result.job_id
            def journal(bundle):
                self._check(cancel,configuration)
                changed=deepcopy(self.ledger['pending']);changed['bundle']=bundle;self._save(status='promoting',pending=changed)
            publish_update(self.training.families.root,self.settings.base_name,candidate_directory/'adapter.safetensors',
                candidate_directory/'evaluation.json',signer,before_install=journal)
            self._recover_publication()
        except (OSError,ValueError,RuntimeError,KeyError,TypeError) as error:
            if self.ledger is not None:self._save(status='interrupted',error=str(error)[:200])
        finally:self.next_attempt=time.monotonic()+30

    def close(self):
        self.closed=True;self.cancel.set()
        if self.thread is not None:self.thread.join(timeout=10)
