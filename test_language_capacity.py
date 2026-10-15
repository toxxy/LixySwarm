"""Signed synthetic timings test allocation; no physical speedup claim."""
import hashlib
from copy import deepcopy
from dataclasses import replace

import pytest

from src.learning.language_capacity import CapacityHistory,allocate_shards,crossed_calibration
from src.network.work_protocol import WorkResult,ResultReceipt
from test_language_versions import fixture


def calibrations(monkeypatch,tmp_path,*,crossed=True):
    _store,head,update,publisher=fixture(monkeypatch,tmp_path)
    _next,evaluation=update(head);workers=update.workers;reports=[]
    for round_index in range(2):
        results=[];timings=[]
        for index,row in enumerate(evaluation['contributions']):
            worker=workers[index if round_index==0 or not crossed else 1-index]
            result=replace(WorkResult.from_dict(row),job_id=hashlib.sha256(f'{round_index}/{index}'.encode()).hexdigest())
            result=replace(result,receipt=ResultReceipt.create(result,worker,publisher.node_id_hex).to_dict())
            ready=.5 if worker==workers[0] else 2
            results.append(result.to_dict());timings.append({**result.output,'receipt':result.receipt,
                'worker_id':worker.node_id_hex,'seconds':ready+.25,'compute_seconds':ready,'encrypted':True})
        reports.append({'success':True,'operation':'language.gradient.v2','base_name':'qwen3-0.6b','head':head,
            'history':[head],'requester_id':publisher.node_id_hex,'receive_bandwidth_mbps':2.5,'signed_results':results,'contributions':timings,
            'physical_workers':[worker.node_id_hex for worker in workers],'contribution_wall_seconds':3})
    return reports,[worker.node_id_hex for worker in workers]


def test_crossed_evidence_allocates_more_to_observed_faster_worker(monkeypatch,tmp_path):
    reports,workers=calibrations(monkeypatch,tmp_path);capacity=crossed_calibration(reports)
    plan=allocate_shards(workers,8,4,start=16,capacity=capacity)
    assert [row['worker_id'] for row in plan]==[workers[0],workers[1],workers[0],workers[0]]
    assert [(row['start'],row['count']) for row in plan]==[(16,8),(24,8),(32,8),(40,8)]
    assert capacity['crossed_inputs_verified'] and all(row['samples']==2 for row in capacity['workers'].values())
    assert not any('speedup' in key for key in capacity)


@pytest.mark.parametrize('failure',['signature','duplicate_job','unmatched_shards','resumed','timing','bandwidth','requester'])
def test_unverified_or_biased_measurements_cannot_allocate_capacity(monkeypatch,tmp_path,failure):
    reports,workers=calibrations(monkeypatch,tmp_path,crossed=failure!='unmatched_shards');reports=deepcopy(reports)
    if failure=='signature':reports[1]['signed_results'][0]['output']['tokens']+=1
    elif failure=='duplicate_job':reports+=deepcopy(reports)
    elif failure=='unmatched_shards':pass
    elif failure=='resumed':reports[1]['resumed']=True
    elif failure=='timing':reports[1]['contributions'][0]['compute_seconds']=float('inf')
    elif failure=='requester':reports[1]['requester_id']='e'*64
    else:reports[1]['receive_bandwidth_mbps']=5
    with pytest.raises(ValueError):crossed_calibration(reports)


def test_slot_limits_and_corpus_tail_preserve_every_example():
    workers=['a'*64,'b'*64]
    capacity={'workers':{worker:{'examples_per_request_second':1} for worker in workers}}
    plan=allocate_shards(workers,8,4,capacity=capacity,slots={workers[0]:2,workers[1]:1})
    assert sum(row['worker_id']==workers[0] for row in plan)==3
    end=allocate_shards(workers,8,4,start=167)
    assert sum(row['count'] for row in end)==31 and end[-1]['start']+end[-1]['count']==198
    tail=allocate_shards(workers,8,4,start=198)
    assert [(row['start'],row['count']) for row in tail]==[(198,1),(199,1)]
    with pytest.raises(ValueError):allocate_shards(workers,8,4,start=199)
    with pytest.raises(ValueError):allocate_shards(workers,8,4,slots={workers[0]:5,workers[1]:1})


def test_private_history_survives_restart_without_counting_receipts_twice(monkeypatch,tmp_path):
    reports,workers=calibrations(monkeypatch,tmp_path/'source');calibration=crossed_calibration(reports)
    history=CapacityHistory(tmp_path/'capacity.json')
    for row in calibration['observations']:history.record(row);history.record(row)
    restored=CapacityHistory(history.path);assert len(restored.rows)==4
    head=reports[0]['head'];capacity=restored.estimate(head,workers)
    assert all(row['samples']==2 for row in capacity['workers'].values())
    changed=deepcopy(head);changed['release']['model_artifact_id']='f'*64
    with pytest.raises(ValueError):restored.estimate(changed,workers)
