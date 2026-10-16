"""Owner-only leased own-swarm jobs through the existing authenticated IPC."""
from copy import deepcopy
import json
import re
import threading
import time

from src.contribution import ResourceRequirements
from src.learning.swarm_distributed import (WORKERS,GRADIENT_OPERATION,CANDIDATE_OPERATION,
    validate_payload,verify_contributions,definition)
from .artifact_store import digest_file
from .work_protocol import ResultReceipt
from .report_privacy import public_route
from .public_knowledge import managed_public_bytes


class LocalOwnWork:
    def __init__(self,service):
        self.service=service;self.knowledge=service.knowledge
        self.jobs={};self.lock=threading.RLock();self.closed=threading.Event()
        self.monitor=threading.Thread(target=self._watch,daemon=True,name='lixy-own-work-lease');self.monitor.start()

    def start(self,value):
        if not isinstance(value,dict) or set(value)!={'request_id','worker_id','operation','payload'} or len(json.dumps(value).encode())>16*1024:
            raise ValueError('Invalid local own-swarm request')
        request,worker,operation,payload=(value[k] for k in ['request_id','worker_id','operation','payload'])
        own=self.knowledge.node.identity.node_id_hex
        candidate=operation==CANDIDATE_OPERATION
        if (not isinstance(request,str) or not re.fullmatch('[0-9a-f]{32}',request)
                or worker not in WORKERS or operation not in {GRADIENT_OPERATION,CANDIDATE_OPERATION}
                or candidate and worker!=own):raise ValueError('Only known physical own-swarm operations are allowed')
        validate_payload(payload,candidate=candidate)
        if not self.service.ready or self.closed.is_set():raise RuntimeError('Own-swarm computation is unavailable')
        current=self.service.snapshot()
        if payload['state_id']!=current['state_id'] or payload['model_id']!=current['model_id']:
            raise ValueError('Own work must use the locally selected training state')
        if candidate:verify_contributions(payload['contributions'],model_id=current['model_id'],
            training_state_id=current['state_id'],requester_id=own)
        with self.lock:
            if request in self.jobs:raise ValueError('Own request already exists')
            if sum(v['state']=='running' for v in self.jobs.values())>=2:raise RuntimeError('Own request slots are occupied')
            for key in list(self.jobs):
                if len(self.jobs)<32:break
                if self.jobs[key]['state']!='running':self.jobs.pop(key)
            job={'state':'running','lease':time.monotonic()+30,'cancel':threading.Event(),
                'operation':operation,'started_at':time.time(),'selected_state_id':current['state_id']}
            self.jobs[request]=job
            job['thread']=threading.Thread(target=self._run,args=(job,worker,operation,deepcopy(payload)),daemon=True)
            job['thread'].start()
        return {'request_id':request,'state':'running'}

    def status(self,request):
        with self.lock:
            if request not in self.jobs:raise ValueError('Unknown own request')
            job=self.jobs[request];job['lease']=time.monotonic()+30
            return deepcopy({k:v for k,v in job.items() if k not in {'lease','cancel','thread'}})

    def cancel(self,request):
        with self.lock:
            if request not in self.jobs:return False
            self.jobs[request]['cancel'].set();return True

    def _watch(self):
        while not self.closed.wait(.5):
            with self.lock:
                for job in self.jobs.values():
                    if job['state']=='running' and (not self.service.ready or time.monotonic()>job['lease']
                            or job['selected_state_id']!=self.service.snapshot()['state_id']):job['cancel'].set()

    def _run(self,job,worker,operation,payload):
        began=time.monotonic();node=self.knowledge.node;p=definition();cancel=job['cancel']
        try:
            requirements=ResourceRequirements(kind='training',gpu_required=False,ram_gb=p['ram_gb'],
                disk_gb=p['candidate_disk_gb'] if operation==CANDIDATE_OPERATION else p['gradient_disk_gb'])
            if worker==node.identity.node_id_hex:
                result=self.knowledge.work.execute_local(operation,payload,requirements,
                    timeout_s=p['maximum_seconds'],cancel_event=cancel)
                route={'transport':'local','public_internet':False,'encrypted':False}
            else:
                if worker not in node.connected_node_ids:node.connect_node(worker,timeout=12)
                peer=next(row for row in node.peers() if row['node_id']==worker)
                route=public_route({k:peer[k] for k in ['host','relay_host','transport','encrypted'] if k in peer})
                if not route['public_internet'] or not route['encrypted']:raise ValueError('Own remote work requires encrypted public Internet')
                result=self.knowledge.work.submit(operation,payload,requirements,peer_id=worker,
                    timeout_s=p['maximum_seconds'],cancel_event=cancel)
            if result.status!='ok' or not result.receipt or not ResultReceipt.from_dict(result.receipt).verify(result,
                    expected_worker_id=worker,expected_requester_id=node.identity.node_id_hex):
                raise ValueError('Own physical work result is unavailable or unsigned')
            initially_local=False;chunks=[]
            if operation==GRADIENT_OPERATION:
                output=result.output
                if any(output.get(k)!=payload[k] for k in payload):raise ValueError('Own physical work result differs from its assignment')
                initially_local=self.service.store.has(output['gradient_id'])
                if (not initially_local and managed_public_bytes(self.service.directory)+2*output['gradient_bytes']
                        >self.service.settings.policy.max_disk_gb*1024**3):
                    raise ValueError('Own gradient acquisition exceeds public storage consent')
                path=self.service.transfer.fetch(output['gradient_id'],peer_id=worker,
                    expected_size=output['gradient_bytes'],timeout_s=120,cancel_event=cancel,observer=chunks.append)
                if digest_file(path)!=(output['gradient_id'],output['gradient_bytes']):
                    raise ValueError('Own acquired gradient differs from its signed result')
            if cancel.is_set():raise RuntimeError('Own work was cancelled')
            with self.lock:job.update(state='complete',result=result.to_dict(),route=route,
                artifact_initially_local=initially_local,seconds=time.monotonic()-began,
                encoded_data_bytes=sum(v['encoded_data_bytes'] for v in chunks),finished_at=time.time())
        except Exception as error:
            with self.lock:job.update(state='cancelled' if cancel.is_set() else 'failed',error_type=type(error).__name__,finished_at=time.time())

    def close(self):
        self.closed.set()
        with self.lock:jobs=list(self.jobs.values())
        for job in jobs:job['cancel'].set()
        for job in jobs:job['thread'].join(timeout=3)
        self.monitor.join(timeout=2)
