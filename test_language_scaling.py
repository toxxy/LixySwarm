"""Signed synthetic work only; no physical speedup is asserted by these tests."""
from copy import deepcopy
from dataclasses import replace
import hashlib

import pytest

from src.learning.language_scaling import compare_runs
from src.network.work_protocol import ResultReceipt,WorkResult
from test_language_versions import fixture


def reports(monkeypatch,tmp_path):
    _store,first,update,publisher=fixture(monkeypatch,tmp_path);_second,evaluation=update(first)
    results=evaluation['contributions']
    def trial(results,seconds):
        return {'success':True,'operation':'language.gradient.v2','head':first,'base_name':'qwen3-0.6b','receive_bandwidth_mbps':2.5,
            'signed_results':results,'physical_workers':sorted({row['receipt']['worker_node_id'] for row in results}),
            'contributions':[{**row['output'],'receipt':row['receipt'],'worker_id':row['receipt']['worker_node_id'],
                'seconds':seconds/2,'encrypted':True} for row in results],'contribution_wall_seconds':seconds}
    single=[]
    for row in results:
        result=replace(WorkResult.from_dict(row),job_id=hashlib.sha256((row['job_id']+'baseline').encode()).hexdigest())
        single.append(replace(result,receipt=ResultReceipt.create(result,update.workers[0],publisher.node_id_hex).to_dict()).to_dict())
    return trial(single,10),trial(results,6)


def test_identical_signed_work_reports_a_measured_gain_or_slowdown(monkeypatch,tmp_path):
    baseline,distributed=reports(monkeypatch,tmp_path)
    value=compare_runs(baseline,distributed)
    assert value['tokens']==32 and value['observed_speedup']==10/6 and value['training_throughput_improved']
    distributed['contribution_wall_seconds']=12
    assert not compare_runs(baseline,distributed)['training_throughput_improved']
    assert not value['new_model_promoted']


@pytest.mark.parametrize('failure',['signature','model','timing','nan','row','workers','incomplete','tokens','duplicates','warmup','bandwidth','slots','requester'])
def test_incomparable_or_unverified_trials_cannot_claim_speedup(monkeypatch,tmp_path,failure):
    baseline,distributed=reports(monkeypatch,tmp_path);distributed=deepcopy(distributed)
    if failure=='signature':distributed['signed_results'][0]['output']['tokens']+=1
    elif failure=='model':distributed['head']['release']['model_artifact_id']='e'*64
    elif failure=='timing':distributed['contribution_wall_seconds']=.01
    elif failure=='nan':distributed['contribution_wall_seconds']=float('nan')
    elif failure=='row':distributed['contributions'][0]['start']+=1
    elif failure=='workers':distributed['physical_workers']=['f'*64]
    elif failure=='incomplete':distributed['success']=False
    elif failure=='tokens':distributed['contributions'][0]['tokens']+=1
    elif failure=='duplicates':distributed['contributions'].append(distributed['contributions'][0])
    elif failure=='warmup':baseline['concurrent_initial_model_copy']=True
    elif failure=='slots':
        worker=baseline['physical_workers'][0];baseline['worker_slot_limits']={worker:1};distributed['worker_slot_limits']={worker:2}
    elif failure=='requester':distributed['requester_id']='e'*64
    else:distributed['receive_bandwidth_mbps']=5
    with pytest.raises((ValueError,RuntimeError)):compare_runs(baseline,distributed)


@pytest.mark.parametrize('failure',[None,'remote_cache','local_bytes','false_local','different_mode','reused_signed_job'])
def test_native_baseline_reuses_own_artifacts_but_requires_fresh_remote_transfers(monkeypatch,tmp_path,failure):
    from dataclasses import replace
    from src.network.work_protocol import WorkResult,ResultReceipt
    from test_language_batches import batch_fixture
    _store,chain,report,_publisher,workers=batch_fixture(monkeypatch,tmp_path)
    own=workers[0].node_id_hex
    def make(distributed):
        results=[];rows=[]
        for index,value in enumerate(report['contributions']):
            worker=workers[index if distributed else 0]
            result=replace(WorkResult.from_dict(value),job_id=hashlib.sha256((value['job_id']+str(distributed)).encode()).hexdigest())
            result=replace(result,receipt=ResultReceipt.create(result,worker,own).to_dict())
            local=worker.node_id_hex==own
            results.append(result.to_dict());rows.append({**result.output,'receipt':result.receipt,'worker_id':worker.node_id_hex,
                'transport':'local' if local else 'relay','encrypted':not local,'seconds':1 if local else 2,
                'encoded_data_bytes':0 if local else 200,'artifact_initially_local':local})
        return {'success':True,'operation':'language.gradient.v3','training_policy_version':2,'native_physical_requester':True,
            'requester_id':own,'base_name':'qwen3-0.6b','head':chain[-2],'history':chain[:-1],
            'signed_results':results,'contributions':rows,'physical_workers':sorted({row['worker_id'] for row in rows}),
            'receive_bandwidth_mbps':2.5,'worker_slot_limits':{worker.node_id_hex:2 if worker==workers[0] else 1 for worker in workers[:2 if distributed else 1]},
            'contribution_wall_seconds':3 if distributed else 4}
    before,after=make(False),make(True)
    if failure=='remote_cache':after['contributions'][1]['artifact_initially_local']=True
    elif failure=='local_bytes':before['contributions'][0]['encoded_data_bytes']=200
    elif failure=='false_local':after['contributions'][1].update(transport='local',encrypted=False,encoded_data_bytes=0)
    elif failure=='different_mode':after['native_physical_requester']=False
    elif failure=='reused_signed_job':
        after['signed_results'][0]=before['signed_results'][0]
        after['contributions'][0]['receipt']=before['signed_results'][0]['receipt']
    if failure:
        with pytest.raises(ValueError):compare_runs(before,after)
    else:
        result=compare_runs(before,after)
        assert result['tokens']==256 and result['observed_speedup']==4/3
