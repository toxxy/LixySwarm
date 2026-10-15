"""Bounded owner-only work through the existing physical node, never a new identity."""
from copy import deepcopy
import json
import re
import threading
import time

from src.contribution import ResourceRequirements
from src.learning.language_policy import policy_id,work_policy,enabled_policy,dataset_id,work_timeout
from src.learning.public_conversation import public_corpus,corpus_digest
from src.release.language_versions import WORKERS,verify_contributions
from .artifact_store import ArtifactService,digest_file
from .work_protocol import ResultReceipt

OPERATIONS={'language.gradient.v2','language.gradient.v3','language.gradient.v4','language.gradient.v5','language.candidate.v1','language.candidate.v2','language.candidate.v3','language.candidate.v4'}


class LocalLanguageWork:
    def __init__(self,knowledge,training,coordinator):
        self.knowledge,self.training,self.coordinator=knowledge,training,coordinator
        self.lock=threading.RLock();self.jobs={};self.closed=threading.Event()
        self.transfer=ArtifactService(knowledge.work,training.families.store,namespace='language.artifact',
            register_handlers=False,receive_budget=knowledge.budget)
        self.monitor=threading.Thread(target=self._watch,name='lixy-owner-work-lease',daemon=True);self.monitor.start()

    def start(self,value):
        fields={'request_id','worker_id','operation','payload'}
        if not isinstance(value,dict) or set(value)!=fields or len(json.dumps(value).encode())>16*1024:
            raise ValueError('Invalid bounded local language work')
        request,worker,operation,payload=(value[k] for k in ['request_id','worker_id','operation','payload'])
        own=self.knowledge.node.identity.node_id_hex
        if (not isinstance(request,str) or not re.fullmatch('[0-9a-f]{32}',request)
                or not isinstance(worker,str) or worker not in WORKERS or not isinstance(operation,str) or operation not in OPERATIONS
                or not isinstance(payload,dict) or operation.startswith('language.candidate.') and worker!=own):
            raise ValueError('Only locally selected known-worker language operations are permitted')
        if self.closed.is_set() or not self.training.ready or not self.knowledge.allowed:
            raise RuntimeError('Local language computation is unavailable')
        if self.coordinator.ledger is not None and self.coordinator.ledger.get('pending') is not None:
            raise RuntimeError('The autonomous coordinator has pending work')
        if self.coordinator.thread is not None and self.coordinator.thread.is_alive():
            raise RuntimeError('The autonomous coordinator is working')
        version=work_policy(operation);batch=version>=2
        expected={'base_name','release_id','adapter_id','dataset_id'}
        expected|={'start','count'} if operation.startswith('language.gradient.') else {'contributions','learning_rate'}
        if batch:expected.add('training_policy_id')
        if set(payload)!=expected or payload['dataset_id']!=dataset_id(version):
            raise ValueError('Local work cannot select private data, paths or programs')
        if batch and (not enabled_policy(self.training.engine,version) or payload.get('training_policy_id')!=policy_id(version)):
            raise ValueError('Batch policy has not been locally enabled')
        head=self.training._head()
        if (payload.get('base_name')!=self.training.engine.base_name
                or payload.get('release_id')!=head['release']['release_id']
                or payload.get('adapter_id')!=head['release']['model_artifact_id']):
            raise ValueError('Local work requires the currently accepted public parent')
        with self.lock:
            if request in self.jobs:raise ValueError('Local work request already exists')
            if sum(row['state']=='running' for row in self.jobs.values())>=4:
                raise RuntimeError('Local work request slots are occupied')
            for old in list(self.jobs):
                if len(self.jobs)<32:break
                if self.jobs[old]['state']!='running':self.jobs.pop(old)
            job={'state':'running','operation':operation,'batch':batch,'half':version>=3,'policy_version':version,'cancel':threading.Event(),'lease':time.monotonic()+30,'started_at':time.time()}
            self.jobs[request]=job
            job['thread']=threading.Thread(target=self._run,args=(job,worker,operation,deepcopy(payload)),daemon=True)
            job['thread'].start()
        return {'request_id':request,'state':'running'}

    def status(self,request):
        with self.lock:
            if request not in self.jobs:raise ValueError('Unknown local work request')
            job=self.jobs[request];job['lease']=time.monotonic()+30
            return deepcopy({key:value for key,value in job.items() if key not in {'cancel','lease','thread'}})

    def cancel(self,request):
        with self.lock:
            if request not in self.jobs:return False
            self.jobs[request]['cancel'].set();return True

    def _watch(self):
        while not self.closed.wait(.5):
            with self.lock:
                for job in self.jobs.values():
                    if job['state']=='running' and (time.monotonic()>job['lease'] or not self.knowledge.allowed
                            or job['batch'] and not self.training.engine.batching_enabled
                            or job['half'] and not enabled_policy(self.training.engine,3)
                            or not enabled_policy(self.training.engine,job.get('policy_version',1))
                            or job['operation'].startswith('language.candidate.') and not self.training.engine.candidates_enabled
                            or not self.training.ready or self.coordinator.thread is not None and self.coordinator.thread.is_alive()):
                        job['cancel'].set()

    def _run(self,job,worker,operation,payload):
        cancel=job['cancel'];began=time.monotonic();node=self.knowledge.node
        try:
            gradient=operation.startswith('language.gradient.')
            timeout=work_timeout(work_policy(operation),candidate=not gradient)
            requirements=ResourceRequirements(kind='training',gpu_required=True,ram_gb=self.training.required_ram(),disk_gb=.04 if gradient else .1)
            if worker==node.identity.node_id_hex:
                result=self.knowledge.work.execute_local(operation,payload,requirements,timeout_s=timeout,cancel_event=cancel)
                route={'transport':'local','encrypted':False,'public_internet':False}
            else:
                if worker not in node.connected_node_ids:node.connect_node(worker,timeout=12)
                result=self.knowledge.work.submit(operation,payload,requirements,peer_id=worker,timeout_s=timeout,cancel_event=cancel)
                peer=next(row for row in node.peers() if row['node_id']==worker)
                route={key:peer.get(key) for key in ['transport','encrypted','host','relay_host']}
            ready=time.monotonic()-began
            if result.status!='ok' or not ResultReceipt.from_dict(result.receipt).verify(result,
                    expected_worker_id=worker,expected_requester_id=node.identity.node_id_hex):
                raise ValueError('Native work result is unavailable or unsigned')
            stats=[];initially_local=False
            if gradient:
                output=result.output
                verify_contributions([result.to_dict()],base_name=payload['base_name'],adapter_id=payload['adapter_id'],
                    dataset_id=payload['dataset_id'],require_distributed=False,policy_version=work_policy(operation))
                if any(output[key]!=payload[key] for key in ['start','count']):raise ValueError('Native result belongs to another shard')
                store=self.training.families.store;initially_local=store.has(output['gradient_id'])
                path=self.transfer.fetch(output['gradient_id'],peer_id=worker,expected_size=output['gradient_bytes'],timeout_s=120,
                    cancel_event=cancel,observer=stats.append)
                if digest_file(path)!=(output['gradient_id'],output['gradient_bytes']):raise ValueError('Native gradient failed verification')
            if cancel.is_set():raise RuntimeError('Native owner work cancelled')
            with self.lock:job.update(state='complete',result=result.to_dict(),compute_seconds=ready,seconds=time.monotonic()-began,
                route=route,artifact_initially_local=initially_local,encoded_data_bytes=sum(row['encoded_data_bytes'] for row in stats),
                transfer_seconds=time.monotonic()-began-ready,finished_at=time.time())
        except Exception as error:
            with self.lock:job.update(state='cancelled' if cancel.is_set() else 'failed',error_type=type(error).__name__,finished_at=time.time())

    def close(self):
        self.closed.set()
        with self.lock:
            jobs=list(self.jobs.values())
            for job in jobs:job['cancel'].set()
        self.monitor.join(timeout=1)
        for job in jobs:job['thread'].join(timeout=1)
