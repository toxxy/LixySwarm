"""Signed synthetic work only; no physical speedup is asserted by these tests."""
from copy import deepcopy
from dataclasses import replace

import pytest

from src.learning.language_scaling import compare_runs
from src.network.work_protocol import ResultReceipt,WorkResult
from test_language_versions import fixture


def reports(monkeypatch,tmp_path):
    _store,first,update,publisher=fixture(monkeypatch,tmp_path);_second,evaluation=update(first)
    results=evaluation['contributions']
    def trial(results,seconds):
        return {'success':True,'operation':'language.gradient.v2','head':first,'base_name':'qwen3-0.6b',
            'signed_results':results,'physical_workers':sorted({row['receipt']['worker_node_id'] for row in results}),
            'contributions':[{**row['output'],'receipt':row['receipt'],'worker_id':row['receipt']['worker_node_id'],
                'seconds':seconds/2,'encrypted':True} for row in results],'contribution_wall_seconds':seconds}
    single=[]
    for row in results:
        result=WorkResult.from_dict(row)
        single.append(replace(result,receipt=ResultReceipt.create(result,update.workers[0],publisher.node_id_hex).to_dict()).to_dict())
    return trial(single,10),trial(results,6)


def test_identical_signed_work_reports_a_measured_gain_or_slowdown(monkeypatch,tmp_path):
    baseline,distributed=reports(monkeypatch,tmp_path)
    value=compare_runs(baseline,distributed)
    assert value['tokens']==32 and value['observed_speedup']==10/6 and value['training_throughput_improved']
    distributed['contribution_wall_seconds']=12
    assert not compare_runs(baseline,distributed)['training_throughput_improved']
    assert not value['new_model_promoted']


@pytest.mark.parametrize('failure',['signature','model','timing','nan','row','workers','incomplete','tokens','duplicates','warmup'])
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
    else:baseline['concurrent_initial_model_copy']=True
    with pytest.raises((ValueError,RuntimeError)):compare_runs(baseline,distributed)
