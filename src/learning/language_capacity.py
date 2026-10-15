"""Requester-observed capacity from verified public work, never hardware claims."""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
import statistics
import threading

from src.network.telemetry import atomic_json
from src.network.work_protocol import ResultReceipt,WorkResult
from src.release.language_versions import verify_contributions,validate_bundle
from .language_scaling import verify_timed_run
from .public_conversation import public_corpus,corpus_digest
from .language_policy import limits,policy_from_id


def observation(result,*,base_name,release_id,request_seconds,transfer_seconds=0):
    value=WorkResult.from_dict(result);receipt=ResultReceipt.from_dict(value.receipt)
    policy_version=policy_from_id(value.output.get('training_policy_id'))
    verify_contributions([result],base_name=base_name,adapter_id=value.output['adapter_id'],
        dataset_id=corpus_digest(public_corpus()),require_distributed=False,policy_version=policy_version)
    if (not re.fullmatch('[0-9a-f]{64}',str(release_id))
            or any(type(number) not in (int,float) or not math.isfinite(number)
                for number in [request_seconds,transfer_seconds])
            or not 0<request_seconds<=3600 or not 0<=transfer_seconds<=7200):
        raise ValueError('Invalid requester-observed language timing')
    return {'base_name':base_name,'release_id':release_id,'worker_id':receipt.worker_node_id,
        'result':result,'request_seconds':request_seconds,'transfer_seconds':transfer_seconds}


def profile(rows,*,base_name,release_id,adapter_id,workers,exact_version=True,policy_version=1):
    grouped=defaultdict(list);jobs=set()
    for row in rows:
        checked=observation(row['result'],base_name=row['base_name'],release_id=row['release_id'],
            request_seconds=row['request_seconds'],transfer_seconds=row['transfer_seconds'])
        result=checked['result'];job=result['job_id']
        if job in jobs:raise ValueError('Capacity evidence repeats a signed calculation')
        jobs.add(job)
        if (row['base_name']==base_name and checked['worker_id'] in workers
                and policy_from_id(result['output'].get('training_policy_id'))==policy_version
                and (not exact_version or row['release_id']==release_id and result['output']['adapter_id']==adapter_id)):
            grouped[checked['worker_id']].append(checked)
    if any(len(grouped[worker])<2 for worker in workers):
        raise ValueError('Capacity allocation needs two fresh observations per worker and version')
    values={}
    for worker in workers:
        samples=grouped[worker][-8:]
        # A worker that suffers a slower shard is not credited merely for
        # returning more small answers. Plan cost uses examples, with token
        # throughput reported separately; all timings are requester-local.
        seconds_per_example=statistics.median(row['request_seconds']/row['result']['output']['count'] for row in samples)
        values[worker]={'samples':len(samples),'examples_per_request_second':1/seconds_per_example,
            'verified_output_tokens_per_request_second':sum(row['result']['output']['tokens'] for row in samples)/sum(row['request_seconds'] for row in samples),
            'mean_transfer_seconds':statistics.mean(row['transfer_seconds'] for row in samples)}
    return {'base_name':base_name,'release_id':release_id,'adapter_id':adapter_id,'workers':values,'training_policy_version':policy_version,
        'sampled_versions':sorted({row['release_id'] for samples in grouped.values() for row in samples}),
        'version_scope':'exact' if exact_version else 'same_pinned_base_family',
        'scope':'Requester-observed calculation readiness; includes network/materialization/compute, excludes artifact acquisition; not a signed worker timer or an end-to-end speedup claim'}


def crossed_calibration(reports):
    if not isinstance(reports,list) or not 2<=len(reports)<=16:raise ValueError('Use two to sixteen crossed calibration trials')
    rows=[];reference=reports[0];coverage=defaultdict(list)
    requester=reference.get('requester_id')
    if not isinstance(requester,str) or not re.fullmatch('[0-9a-f]{64}',requester):raise ValueError('Calibration needs its observed requester identity')
    for report in reports:
        verify_timed_run(report)
        if (report.get('resumed') or report['head']!=reference['head'] or report['base_name']!=reference['base_name']
                or report.get('training_policy_version',1)!=reference.get('training_policy_version',1)
                or report.get('receive_bandwidth_mbps')!=reference.get('receive_bandwidth_mbps')
                or report.get('requester_id')!=requester):
            raise ValueError('Capacity calibration must use the same current model, data and receive policy')
        results={r['job_id']:r for r in report['signed_results']}
        for timing in report['contributions']:
            if timing.get('reused_signed_contribution'):raise ValueError('Recovered work cannot calibrate fresh execution')
            result=results[timing['receipt']['job_id']]
            ready=timing['compute_seconds'];total=timing['seconds']
            if (type(ready) not in (int,float) or not math.isfinite(ready) or ready>total+.01
                    or ResultReceipt.from_dict(result['receipt']).requester_node_id!=requester):
                raise ValueError('Calculation timing or requester differs from its signed work')
            row=observation(result,base_name=report['base_name'],release_id=report['head']['release']['release_id'],
                request_seconds=ready,transfer_seconds=max(0,total-ready))
            rows.append(row);coverage[row['worker_id']].append(tuple(result['output'][key] for key in ['start','count','tokens']))
    workers=sorted(coverage)
    if len(workers)<2 or any(sorted(coverage[w])!=sorted(coverage[workers[0]]) for w in workers):
        raise ValueError('Crossed calibration must give every worker the same shards and tokens')
    result=profile(rows,base_name=reference['base_name'],release_id=reference['head']['release']['release_id'],
        adapter_id=reference['head']['release']['model_artifact_id'],workers=workers,policy_version=reference.get('training_policy_version',1))
    return {**result,'observations':rows,'crossed_inputs_verified':True}


def allocate_shards(workers,count,shards,*,start=0,capacity=None,slots=None,policy_version=1):
    if (not isinstance(workers,list) or not 1<=len(workers)<=8 or len(set(workers))!=len(workers)
            or any(not isinstance(w,str) or not re.fullmatch('[0-9a-f]{64}',w) for w in workers)
            or type(count) is not int or not 1<=count<=limits(policy_version)[0] or type(shards) is not int or not len(workers)<=shards<=16
            or type(start) is not int or not 0<=start<len(public_corpus()['train'])):
        raise ValueError('Invalid bounded language work plan')
    slots=slots or {worker:1 for worker in workers}
    if set(slots)!=set(workers) or any(type(n) is not int or not 1<=n<=4 for n in slots.values()):
        raise ValueError('Use one to four locally bounded calculation slots per worker')
    remaining=len(public_corpus()['train'])-start
    total=min(count*shards,remaining)
    if remaining-total==1:total-=1
    shards=min(shards,total)
    if shards<len(workers):raise ValueError('The corpus tail cannot supply every physical worker')
    lengths=[total//shards+(index<total%shards) for index in range(shards)]
    rates={worker:1.0 for worker in workers}
    if capacity is not None:
        for worker in workers:
            rate=capacity['workers'][worker]['examples_per_request_second']
            if type(rate) not in (int,float) or not math.isfinite(rate) or rate<=0:raise ValueError('Invalid measured capacity rate')
            rates[worker]=rate
    queues={worker:[0.0]*slots[worker] for worker in workers};plan=[];position=start
    for index,length in enumerate(lengths):
        if capacity is None:worker=workers[index%len(workers)]
        elif index<len(workers):worker=workers[index]
        else:worker=min(workers,key=lambda w:(min(queues[w])+length/rates[w],workers.index(w)))
        slot=min(range(slots[worker]),key=lambda n:queues[worker][n]);queues[worker][slot]+=length/rates[worker]
        plan.append({'worker_id':worker,'start':position,'count':length});position+=length
    return plan


class CapacityHistory:
    """Private durable observations; restoring a receipt cannot create credit."""
    def __init__(self,path):
        self.path=Path(path);self.lock=threading.RLock();self.rows=[]
        if self.path.exists():
            if self.path.stat().st_size>512*1024:raise ValueError('Capacity history exceeds its bound')
            data=json.loads(self.path.read_text())
            if (set(data)!={'version','observations'} or type(data['version']) is not int or data['version']!=1
                    or not isinstance(data['observations'],list) or len(data['observations'])>64):
                raise ValueError('Invalid capacity history')
            for row in data['observations']:self._validate(row)
            if len({r['result']['job_id'] for r in data['observations']})!=len(data['observations']):raise ValueError('Repeated capacity receipt')
            self.rows=data['observations']

    @staticmethod
    def _validate(row):
        return observation(row['result'],base_name=row['base_name'],release_id=row['release_id'],
            request_seconds=row['request_seconds'],transfer_seconds=row['transfer_seconds'])

    def record(self,row):
        checked=self._validate(row)
        with self.lock:
            if any(r['result']['job_id']==checked['result']['job_id'] for r in self.rows):return
            self.rows=(self.rows+[checked])[-64:]
            atomic_json(self.path,{'version':1,'observations':self.rows})

    def estimate(self,head,workers,*,policy_version=1):
        with self.lock:
            validate_bundle(head)
            return profile(self.rows,base_name=head['descriptor']['base_name'],release_id=head['release']['release_id'],
                adapter_id=head['release']['model_artifact_id'],workers=workers,exact_version=False,policy_version=policy_version)
