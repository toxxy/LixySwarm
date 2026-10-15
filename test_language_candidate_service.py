"""Synthetic physical receipts and tensor-format fixtures; no learning claim."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from src.contribution import ContributionPolicy,ResourceRequirements
from src.network.client_state import ClientSettings
from src.network.compute_process import ComputeProcessError
from src.network.language_candidate import run_candidate
from src.network.language_training import LanguageTraining
from test_language_versions import fixture as versions_fixture


def service(monkeypatch,tmp_path,*,policy_version=1):
    root=tmp_path/'public/language'
    store,first,update,publisher=versions_fixture(monkeypatch,root)
    second,report=update(first)
    if policy_version==3:
        from src.learning.language_policy import HALF_POLICY_ID
        from src.network.work_protocol import WorkResult,ResultReceipt
        report.update(version=3)
        for index,row in enumerate(report['contributions']):
            value=replace(WorkResult.from_dict(row),output={**row['output'],'training_policy_id':HALF_POLICY_ID,
                'quantization_relative_l2_error':.0002})
            report['contributions'][index]=replace(value,receipt=ResultReceipt.create(value,update.workers[index],publisher.node_id_hex).to_dict()).to_dict()
    knowledge=SimpleNamespace(allowed=True,work=SimpleNamespace(register_handler=lambda *_a,**_kw:None))
    families=SimpleNamespace(root=root,store=store,artifacts=SimpleNamespace(fetch=lambda *_a,**_kw:pytest.fail('all gradients already local')))
    training=LanguageTraining(knowledge,families,tmp_path,{'has_gpu':True,'gpu_vram_gb':8,'cpu_cores':4})
    training.ready=True;training.engine.enabled=True;training.engine.candidates_enabled=True
    training.engine.requesters=[publisher.node_id_hex]
    training.settings=ClientSettings(policy=ContributionPolicy.for_mode('maximum'))
    if policy_version==3:training.engine.batching_enabled=True;training.engine.half_precision_enabled=True
    monkeypatch.setattr(training,'_head',lambda:first)
    base=tmp_path/'base';base.mkdir()
    config=store.object_path(first['descriptor']['files'][next(i for i,r in enumerate(first['descriptor']['files']) if r['name']=='config.json')]['artifact_id'])
    (base/'config.json').write_bytes(config.read_bytes())
    from src.network import language_candidate
    monkeypatch.setattr(language_candidate,'materialize_version',lambda *_a,**_kw:base)
    payload={'base_name':'qwen3-0.6b','release_id':first['release']['release_id'],
        'adapter_id':first['release']['model_artifact_id'],'dataset_id':report['dataset_id'],
        'contributions':deepcopy(report['contributions']),'learning_rate':.001}
    if policy_version==3:payload['training_policy_id']=HALF_POLICY_ID
    execution=SimpleNamespace(work=SimpleNamespace(origin_node_id=publisher.node_id_hex,job_id='f'*64,payload=payload),
        resource_requirements=lambda:ResourceRequirements(kind='training',gpu_required=True,ram_gb=4),
        deadline=time.time()+60,raise_if_cancelled=lambda:None)
    def run(_command,request,_execution,**kwargs):
        assert kwargs['allowed']() and request['action']=='candidate'
        if policy_version==3:
            assert request['training_policy_id']==HALF_POLICY_ID
            assert all('quantization_relative_l2_error' in row for row in request['contributions'])
        Path(request['output_file']).write_bytes(store.object_path(second['release']['model_artifact_id']).read_bytes())
        return {**deepcopy(report),'adapter_bytes':second['descriptor']['adapter_bytes']}
    monkeypatch.setattr(training.supervisor,'run',run)
    return training,payload,execution,report


@pytest.mark.parametrize('failure',[None,'single_worker','regression'])
def test_candidate_keeps_only_eligible_weights_and_never_promotes(monkeypatch,tmp_path,failure):
    training,payload,execution,report=service(monkeypatch,tmp_path)
    if failure=='single_worker':payload['contributions']=payload['contributions'][:1]
    if failure=='regression':report['quality_after'][0]['answer']='Madrid'
    before=training.families.store.total_bytes()
    result=run_candidate(training,payload,execution)
    summary=result['summary'];directory=tmp_path/'language-candidates'/execution.work.job_id
    assert summary['eligible_for_publication']==(failure is None)
    assert not summary['published'] and training.completed==1 and training.active==0
    assert (directory/'evaluation.json').is_file()
    assert (directory/'adapter.safetensors').exists()==(failure is None)
    assert training.families.store.total_bytes()==before
    assert not (training.families.root/'versions/qwen3-0.6b/head.json').exists()
    assert training._permit_candidate_cached(execution,result)
    training.engine.candidates_enabled=False
    assert not training._permit_candidate_cached(execution,result)


def test_parent_change_during_candidate_discards_local_trial(monkeypatch,tmp_path):
    training,payload,execution,report=service(monkeypatch,tmp_path)
    original=training.supervisor.run
    def run(*args,**kwargs):
        result=original(*args,**kwargs)
        monkeypatch.setattr(training,'_head',lambda:{'release':{'release_id':'e'*64}})
        return result
    monkeypatch.setattr(training.supervisor,'run',run)
    with pytest.raises(ComputeProcessError,match='parent changed'):run_candidate(training,payload,execution)
    assert training.completed==0 and training.active==0
    assert not (tmp_path/'language-candidates'/execution.work.job_id).exists()


def test_candidate_requires_explicit_local_operation_permission(monkeypatch,tmp_path):
    training,payload,execution,_=service(monkeypatch,tmp_path)
    training.engine.candidates_enabled=False
    with pytest.raises(ComputeProcessError,match='not locally approved'):run_candidate(training,payload,execution)
    assert not (tmp_path/'language-candidates').exists()


@pytest.mark.parametrize('withdraw_during_compute',[False,True])
def test_half_candidate_binds_policy_and_stops_when_precision_permission_changes(monkeypatch,tmp_path,withdraw_during_compute):
    from src.learning.language_policy import HALF_POLICY_ID
    training,payload,execution,_=service(monkeypatch,tmp_path,policy_version=3)
    training.engine.half_precision_enabled=False
    with pytest.raises(ComputeProcessError,match='locally enabled'):run_candidate(training,payload,execution,policy_version=3)
    training.engine.half_precision_enabled=True
    if withdraw_during_compute:
        original=training.supervisor.run
        def changed(*args,**kw):
            result=original(*args,**kw);training.engine.half_precision_enabled=False;return result
        monkeypatch.setattr(training.supervisor,'run',changed)
        with pytest.raises(ComputeProcessError,match='permission or parent'):run_candidate(training,payload,execution,policy_version=3)
        assert training.completed==0 and not (tmp_path/'language-candidates'/execution.work.job_id).exists()
    else:
        result=run_candidate(training,payload,execution,policy_version=3)
        assert result['summary']['training_policy_id']==HALF_POLICY_ID and result['evaluation']['version']==3
        assert result['summary']['eligible_for_publication'] and not result['summary']['published']
        assert training._permit_candidate_cached(execution,result)
        training.engine.half_precision_enabled=False
        assert not training._permit_candidate_cached(execution,result)
