"""Durable bounded native coordination of the organism's own parameters.

This known-operator pilot has one locally selected coordinator. It does not
claim open consensus, leader election, learned metabolism or unlimited growth.
"""
from copy import deepcopy
import json
from pathlib import Path
import threading
import time
import uuid

from src.contribution import ResourceRequirements
from src.learning.swarm_distributed import (WORKERS,GRADIENT_OPERATION,CANDIDATE_OPERATION,
    policy_id,definition,verify_contributions,verdict)
from .own_learning_settings import OwnLearningSettings
from .own_replication import HEAD_OPERATION,ADOPT_OPERATION,ADOPT_DISK_GB
from .own_state import verify_transition
from .work_protocol import ResultReceipt
from .report_privacy import public_route
from .telemetry import atomic_json


class OwnLearningCoordinator:
    def __init__(self,service,local_work):
        self.service=service;self.local_work=local_work;self.knowledge=service.knowledge
        self.root=service.directory/'own-learning';self.path=self.root/'ledger.json'
        self.settings=OwnLearningSettings();self.ledger=None;self.error=None
        self.lock=threading.RLock();self.changed=threading.Event();self.cancel=threading.Event()
        self.thread=None;self.closed=False
        if self.path.exists():
            if self.path.stat().st_size>512*1024:raise ValueError('Own learning ledger exceeds its bound')
            ledger=json.loads(self.path.read_text())
            if (ledger.get('version')!=1 or ledger.get('training_policy_id')!=policy_id()
                    or type(ledger.get('round')) is not int or not 0<=ledger['round']<=10
                    or type(ledger.get('cursor')) is not int or not 0<=ledger['cursor']<=4096
                    or type(ledger.get('rejections')) is not int or not 0<=ledger['rejections']<=10
                    or not isinstance(ledger.get('history'),list) or len(ledger['history'])!=ledger['round']):
                raise ValueError('Invalid own progress; preserve the ledger for inspection')
            self.ledger=ledger

    def _save(self,**values):
        with self.lock:
            self.ledger.update(values);atomic_json(self.path,self.ledger);self.changed.set()

    def snapshot(self):
        with self.lock:
            value={k:self.ledger.get(k) for k in ['status','round','cursor','accepted','rejections','state_id','error','retry_at']} if self.ledger else {
                'status':'idle','round':0,'cursor':self.settings.initial_cursor,'accepted':0,'rejections':0}
            return {**value,'enabled':self.settings.enabled,'follow_enabled':self.settings.follow_enabled,
                'training_policy_id':policy_id(),'program_error':self.error,
                'pending':self.ledger.get('pending') is not None if self.ledger else False}

    def _allowed(self,contract):
        return (not self.closed and self.settings.enabled and self.settings.contract()==contract
            and self.service.ready and self.knowledge.allowed
            and getattr(getattr(self.service,'engine',None),'version',1)==1
            and self.knowledge.node.identity.node_id_hex==self.settings.owner_id)

    def _check(self,contract):
        if self.cancel.is_set() or not self._allowed(contract):raise InterruptedError('Own learning paused')

    def update(self):
        try:
            self.settings=OwnLearningSettings.load(self.service.directory)
            if self.ledger and self.ledger['contract_id']!=self.settings.contract():
                raise ValueError('Own learning program changed; durable progress was preserved')
            self.error=None
        except (OSError,ValueError,TypeError) as error:
            self.error=str(error);self.cancel.set();return
        contract=self.settings.contract()
        if not self._allowed(contract):self.cancel.set();return
        if self.thread is not None and self.thread.is_alive():return
        if self.ledger is None:
            if self.service.snapshot()['state_id']!=self.settings.initial_state_id:
                self.error='Own learning initial state is not the selected verified state';return
            self.ledger={'version':1,'contract_id':contract,'training_policy_id':policy_id(),
                'initial_state_id':self.settings.initial_state_id,'state_id':self.settings.initial_state_id,
                'round':0,'cursor':self.settings.initial_cursor,'epoch':0,'accepted':0,'rejections':0,
                'pending':None,'history':[],'status':'ready','error':None,'retry_at':0}
            self._save()
        if self.ledger['pending'] is None:
            if self.ledger['round']>=self.settings.maximum_rounds:
                if self.ledger['status']!='complete':self._save(status='complete')
                return
            if self.ledger['rejections']>=self.settings.stop_after_rejections:
                if self.ledger['status']!='no_improvement':self._save(status='no_improvement')
                return
            if self.ledger['cursor']+2*self.settings.documents_per_worker>4096:
                if self.ledger['status']!='curriculum_complete':self._save(status='curriculum_complete')
                return
        if time.time()<self.ledger.get('retry_at',0):return
        self.cancel=threading.Event()
        self.thread=threading.Thread(target=self._run,args=(contract,),name='lixy-own-learning',daemon=True)
        self.thread.start()

    def _rpc(self,worker,operation,payload,contract):
        self._check(contract);node=self.knowledge.node
        adopt=operation==ADOPT_OPERATION
        requirements=ResourceRequirements(kind='training' if adopt else 'artifact',
            ram_gb=definition()['ram_gb'] if adopt else .01,disk_gb=ADOPT_DISK_GB if adopt else 0)
        options={'timeout_s':1200 if adopt else 30,'cancel_event':self.cancel}
        if worker==node.identity.node_id_hex:
            result=self.knowledge.work.execute_local(operation,payload,requirements,**options)
        else:
            if worker not in node.connected_node_ids:node.connect_node(worker,timeout=12)
            peer=next(row for row in node.peers() if row['node_id']==worker)
            route=public_route({k:peer[k] for k in ['host','relay_host','transport','encrypted'] if k in peer})
            if not route['encrypted'] or not route['public_internet']:
                raise ValueError('Own coordination requires encrypted public Internet')
            result=self.knowledge.work.submit(operation,payload,requirements,peer_id=worker,**options)
        self._check(contract)
        if (result.status!='ok' or not result.receipt or not ResultReceipt.from_dict(result.receipt).verify(result,
                expected_worker_id=worker,expected_requester_id=node.identity.node_id_hex)):
            raise RuntimeError('Own native state operation did not return a verified physical receipt')
        return result.to_dict()

    def _sync_peer(self,worker,contract):
        head=self._rpc(worker,HEAD_OPERATION,{},contract)['output']
        if head.get('training_policy_id')!=policy_id() or not head.get('ready'):
            raise RuntimeError('Own physical worker is not ready for the pinned policy')
        following=self.service.snapshot()['state_id'];chain=[]
        while following!=head['state_id']:
            if len(chain)>=32:raise ValueError('Own worker lineage cannot be recovered within the history bound')
            path=self.service.root/'history'/(following+'.json')
            if not path.is_file():raise ValueError('Own worker follows another or unavailable lineage')
            transition=json.loads(path.read_text());chain.append(transition['candidate_result'])
            following=transition['parent_state_id']
        for result in reversed(chain):self._rpc(worker,ADOPT_OPERATION,{'candidate_result':result},contract)
        return head

    def _wait_jobs(self,assignments,contract,pending):
        active={}
        try:
            for row in assignments:
                worker=row['worker_id']
                if worker in pending['contributions']:
                    identifier=pending['contributions'][worker]['result']['output']['gradient_id']
                    if self.service.store.has(identifier):continue
                    pending['contributions'].pop(worker)
                self._check(contract);request=uuid.uuid4().hex
                payload={k:pending[k] for k in ['state_id','model_id']}
                payload.update(dataset_id=definition()['training_dataset_id'],training_policy_id=policy_id(),
                    start=row['start'],count=row['count'])
                self.local_work.start({'request_id':request,'worker_id':worker,'operation':GRADIENT_OPERATION,'payload':payload})
                active[worker]=request
            while active:
                self._check(contract)
                for worker,request in list(active.items()):
                    row=self.local_work.status(request)
                    if row['state']=='complete':
                        pending['contributions'][worker]={'worker_id':worker,'result':row['result']}
                        pending['work'][worker]=row
                        self._save(pending=deepcopy(pending));active.pop(worker)
                    elif row['state']!='running':raise RuntimeError('Own gradient job did not complete: '+row['state'])
                if active:self.cancel.wait(.2)
        finally:
            for request in active.values():self.local_work.cancel(request)

    def _candidate(self,pending,contract):
        self._check(contract);request=uuid.uuid4().hex
        payload={k:pending[k] for k in ['state_id','model_id']}
        payload.update(dataset_id=definition()['training_dataset_id'],training_policy_id=policy_id(),
            contributions=list(pending['contributions'].values()))
        self.local_work.start({'request_id':request,'worker_id':self.settings.owner_id,
            'operation':CANDIDATE_OPERATION,'payload':payload})
        try:
            while True:
                self._check(contract);row=self.local_work.status(request)
                if row['state']=='complete':return row
                if row['state']!='running':raise RuntimeError('Own candidate did not complete: '+row['state'])
                self.cancel.wait(.2)
        finally:self.local_work.cancel(request)

    def _run(self,contract):
        pending=None
        try:
            self._check(contract)
            pending=deepcopy(self.ledger['pending'])
            if pending is None:
                selected=self.service.snapshot()
                if selected['state_id']!=self.ledger['state_id']:
                    raise ValueError('Own selected state moved outside durable learning history')
                pending={'round':self.ledger['round'],'cursor':self.ledger['cursor'],
                    'state_id':selected['state_id'],'model_id':selected['model_id'],
                    'assignments':[{'worker_id':worker,'start':self.ledger['cursor']+i*self.settings.documents_per_worker,
                        'count':self.settings.documents_per_worker} for i,worker in enumerate(WORKERS)],
                    'contributions':{},'work':{},'candidate':None,'adoptions':{},'attempts':0,'started_at':time.time()}
                self._save(pending=deepcopy(pending))
            pending['attempts']+=1;self._save(pending=deepcopy(pending),status='synchronizing',error=None)
            expected=[{'worker_id':worker,'start':self.ledger['cursor']+i*self.settings.documents_per_worker,
                'count':self.settings.documents_per_worker} for i,worker in enumerate(WORKERS)]
            if (pending['round']!=self.ledger['round'] or pending['cursor']!=self.ledger['cursor']
                    or pending['state_id']!=self.ledger['state_id'] or pending['assignments']!=expected):
                raise ValueError('Pending own curriculum does not match durable progress')
            if pending['candidate'] is None:
                if self.service.snapshot()['state_id']!=pending['state_id']:
                    raise ValueError('Pending own work belongs to another selected state')
                for worker in WORKERS:
                    if worker!=self.settings.owner_id:self._sync_peer(worker,contract)
                self._save(status='gradients')
                self._wait_jobs(pending['assignments'],contract,pending)
                verify_contributions(list(pending['contributions'].values()),model_id=pending['model_id'],
                    training_state_id=pending['state_id'],requester_id=self.settings.owner_id)
                for row in expected:
                    received=pending['contributions'][row['worker_id']]['result']['output']
                    if any(received[k]!=row[k] for k in ['start','count']):
                        raise ValueError('Own contribution does not match its curriculum assignment')
                self._save(status='evaluating')
                pending['candidate']=self._candidate(pending,contract)
                self._save(pending=deepcopy(pending))
            result=pending['candidate']['result'];output=result['output']
            if output['state_id']!=pending['state_id'] or output['model_id']!=pending['model_id']:
                raise ValueError('Own candidate does not belong to pending curriculum')
            if output['accepted']!=verdict(output['before'],output['after']):
                raise ValueError('Own candidate result contradicts the fixed quality policy')
            selected=pending['state_id']
            if not output['accepted'] and self.service.snapshot()['state_id']!=selected:
                raise ValueError('Rejected own attempt cannot change the selected state')
            if output['accepted']:
                transition=verify_transition(result,self.service.store.object_path(output['evaluation_id']).read_bytes())
                selected=transition['state_id'];self._save(status='replicating')
                for worker in [self.settings.owner_id]+[v for v in WORKERS if v!=self.settings.owner_id]:
                    # Always reconcile physically after a restart; a durable
                    # receipt alone is not evidence that the worker is live.
                    adopted=self._rpc(worker,ADOPT_OPERATION,{'candidate_result':result},contract)
                    if adopted['output']['state_id']!=selected:raise ValueError('Own replica selected another state')
                    pending['adoptions'][worker]=adopted;self._save(pending=deepcopy(pending))
            self._check(contract)
            finished={**pending,'accepted':output['accepted'],'resulting_state_id':selected,
                'finished_at':time.time(),'autonomous':True}
            atomic_json(self.root/'rounds'/f"{pending['round']:04d}.json",finished)
            self._save(round=self.ledger['round']+1,cursor=pending['cursor']+2*self.settings.documents_per_worker,
                accepted=self.ledger['accepted']+int(output['accepted']),
                rejections=0 if output['accepted'] else self.ledger['rejections']+1,
                state_id=selected,pending=None,status='ready',error=None,retry_at=0,
                history=self.ledger['history']+[{'round':pending['round'],'accepted':output['accepted'],
                    'state_id':selected,'cursor':pending['cursor'],'finished_at':finished['finished_at']}])
        except InterruptedError:
            if self.ledger:self._save(status='paused')
        except Exception as error:
            # Network failures preserve contributions/candidate and do not count
            # as quality rejections, advance the cursor or reset the lineage.
            if self.ledger:self._save(status='waiting',error=type(error).__name__+': '+str(error)[:160],
                retry_at=time.time()+min(300,15*max(1,(pending or {}).get('attempts',1))))

    def close(self):
        self.closed=True;self.cancel.set()
        if self.thread is not None:self.thread.join(timeout=5)
