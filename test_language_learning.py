"""Signed fixtures exercise crash boundaries; no physical learning claim."""
import json
import threading
from dataclasses import asdict,replace
from types import SimpleNamespace

import pytest

from src.network import language_learning as learning
from src.network.work_protocol import ResultReceipt,WorkResult
from src.release import language_family,language_publish
from test_language_versions import fixture


def setup(monkeypatch,tmp_path):
    root=tmp_path/'node';store,head,update,signer=fixture(monkeypatch,root/'public/language')
    second,evaluation=update(head);workers=update.workers
    monkeypatch.setattr(learning,'WORKERS',{worker.node_id_hex for worker in workers})
    monkeypatch.setattr(learning,'SIGNER',signer.node_id_hex)
    monkeypatch.setattr(language_publish,'BASES',language_family.BASES)
    identity=workers[0];key=tmp_path/'publisher.key';signer.save(str(key))
    results=[]
    for index,row in enumerate(evaluation['contributions']):
        result=WorkResult.from_dict(row)
        result=replace(result,receipt=ResultReceipt.create(result,workers[index],identity.node_id_hex).to_dict())
        results.append(result)
    evaluation['contributions']=[result.to_dict() for result in results]
    calls=[];controls=learning.LanguageLearningSettings(enabled=True,workers=[worker.node_id_hex for worker in workers],
        signing_key_file=str(key),base_name='qwen3-0.6b',examples_per_worker=4,shards_per_cycle=2,allocation='balanced')
    controls.save(root)
    def execute(operation,payload,*_a,**kwargs):
        calls.append(operation)
        if operation=='language.gradient.v2':return results[payload['start']//4]
        assert operation=='language.candidate.v1'
        directory=root/'language-candidates'/('f'*64);directory.mkdir(parents=True,exist_ok=True)
        (directory/'adapter.safetensors').write_bytes(store.object_path(second['release']['model_artifact_id']).read_bytes())
        (directory/'evaluation.json').write_text(json.dumps(evaluation))
        result=WorkResult(job_id='f'*64,status='ok',output={'summary':{'eligible_for_publication':True},'evaluation':evaluation})
        return replace(result,receipt=ResultReceipt.create(result,identity,identity.node_id_hex).to_dict())
    work=SimpleNamespace(execute_local=execute,submit=execute)
    node=SimpleNamespace(identity=identity,connected_node_ids=[workers[1].node_id_hex])
    knowledge=SimpleNamespace(node=node,work=work,allowed=True)
    training=SimpleNamespace(families=SimpleNamespace(root=root/'public/language',store=store),ready=True,
        engine=SimpleNamespace(candidates_enabled=True,base_name='qwen3-0.6b'),required_ram=lambda:4)
    monkeypatch.setattr(learning,'fetch_history',lambda *_a,**_kw:[head])
    coordinator=learning.LanguageLearningCoordinator(knowledge,training,root);coordinator.settings=controls
    return coordinator,controls,head,calls


def test_round_advances_durable_cursor_and_publishes_only_evaluated_update(monkeypatch,tmp_path):
    coordinator,controls,head,calls=setup(monkeypatch,tmp_path)
    coordinator._run(threading.Event())
    assert coordinator.ledger['error'] is None
    assert coordinator.ledger['round']==1 and coordinator.ledger['cursor']==8
    assert coordinator.ledger['accepted_sequence']==1 and coordinator.ledger['pending'] is None
    assert calls.count('language.gradient.v2')==2 and calls.count('language.candidate.v1')==1
    assert coordinator._registry().head()['release']['previous_release_id']==head['release']['release_id']
    restored=learning.LanguageLearningCoordinator(coordinator.knowledge,coordinator.training,coordinator.directory)
    restored.settings=controls;restored._run(threading.Event())
    assert restored.ledger['round']==1 and restored.ledger['cursor']==8 and len(calls)==3
    assert restored.ledger['pending'] is None
    imported=learning.LanguageLearningCoordinator(coordinator.knowledge,coordinator.training,tmp_path/'new-progress')
    imported.settings=controls;imported._initialize()
    assert imported.ledger['cursor']==8 and imported.ledger['accepted_sequence']==1


@pytest.mark.parametrize('crash_after_commit',[False,True])
def test_publication_journal_recovers_once_on_either_side_of_head_commit(monkeypatch,tmp_path,crash_after_commit):
    coordinator,controls,head,calls=setup(monkeypatch,tmp_path)
    original=learning.publish_update
    def interrupted(*args,**kwargs):
        if crash_after_commit:
            original(*args,**kwargs);raise RuntimeError('injected after head commit')
        journal=kwargs['before_install']
        def stop(bundle):journal(bundle);raise RuntimeError('injected before head commit')
        return original(*args,**{**kwargs,'before_install':stop})
    monkeypatch.setattr(learning,'publish_update',interrupted)
    coordinator._run(threading.Event())
    assert coordinator.ledger['round']==0 and coordinator.ledger['cursor']==0
    assert coordinator.ledger['pending']['bundle'] is not None
    assert coordinator._registry().head()['release']['sequence']==int(crash_after_commit)
    restored=learning.LanguageLearningCoordinator(coordinator.knowledge,coordinator.training,coordinator.directory)
    restored.settings=controls;restored._run(threading.Event());restored._recover_publication()
    assert restored.ledger['round']==1 and restored.ledger['cursor']==8
    assert restored._registry().head()['release']['sequence']==1 and len(calls)==3


def test_signed_receipts_survive_failed_fetch_without_recalculating(monkeypatch,tmp_path):
    coordinator,controls,head,calls=setup(monkeypatch,tmp_path)
    store=coordinator.training.families.store
    original=store.has
    monkeypatch.setattr(store,'has',lambda _identifier:False)
    def fail(*_a,**_kw):raise TimeoutError('injected transfer interruption')
    coordinator.training.families.artifacts=SimpleNamespace(fetch=fail)
    coordinator._run(threading.Event())
    assert coordinator.ledger['round']==0 and coordinator.ledger['cursor']==0
    assert len(coordinator.ledger['pending']['contributions'])==2
    assert calls==['language.gradient.v2','language.gradient.v2']
    monkeypatch.setattr(store,'has',original)
    restored=learning.LanguageLearningCoordinator(coordinator.knowledge,coordinator.training,coordinator.directory)
    restored.settings=controls;restored._run(threading.Event())
    assert restored.ledger['round']==1 and restored.ledger['cursor']==8
    assert calls.count('language.gradient.v2')==2


def test_disabled_controls_and_quality_stop_do_not_schedule_work(monkeypatch,tmp_path):
    coordinator,controls,head,calls=setup(monkeypatch,tmp_path)
    replace(controls,enabled=False).save(coordinator.directory);coordinator.update()
    assert coordinator.thread is None and coordinator.ledger is None and not calls
    controls.save(coordinator.directory);coordinator.settings=controls;coordinator._initialize()
    coordinator._save(rejections=controls.stop_after_rejections)
    coordinator.update()
    assert coordinator.ledger['status']=='no_improvement' and coordinator.thread is None


def test_tampered_saved_receipt_stops_before_new_work(monkeypatch,tmp_path):
    coordinator,controls,head,calls=setup(monkeypatch,tmp_path)
    coordinator._initialize();coordinator._prepare_round()
    pending=coordinator.ledger['pending'];pending['contributions']['0']={'invalid':'checkpoint'}
    coordinator._save(pending=pending)
    coordinator._run(threading.Event())
    assert not calls and coordinator.ledger['round']==0 and coordinator.ledger['cursor']==0
    assert coordinator.ledger['error'] and coordinator._registry().head()==head


def test_replica_wait_preserves_attempt_budget_and_cursor(monkeypatch,tmp_path):
    coordinator,controls,head,calls=setup(monkeypatch,tmp_path)
    def unavailable(*_a,**_kw):raise TimeoutError('fixture replica reconnecting')
    monkeypatch.setattr(learning,'fetch_history',unavailable)
    for _ in range(4):coordinator._run(threading.Event())
    assert coordinator.ledger['status']=='waiting_replicas'
    assert coordinator.ledger['pending']['attempts']==0 and coordinator.ledger['cursor']==0 and calls==[]
    monkeypatch.setattr(learning,'fetch_history',lambda *_a,**_kw:[head])
    coordinator._run(threading.Event())
    assert coordinator.ledger['round']==1 and coordinator.ledger['cursor']==8


def test_quality_rejection_is_recorded_without_publishing(monkeypatch,tmp_path):
    coordinator,controls,head,calls=setup(monkeypatch,tmp_path)
    execute=coordinator.knowledge.work.execute_local
    def reject(operation,*args,**kwargs):
        result=execute(operation,*args,**kwargs)
        if operation=='language.candidate.v1':
            result=replace(result,output={**result.output,'summary':{'eligible_for_publication':False}})
            identity=coordinator.knowledge.node.identity
            result=replace(result,receipt=ResultReceipt.create(result,identity,identity.node_id_hex).to_dict())
        return result
    coordinator.knowledge.work.execute_local=reject
    monkeypatch.setattr(learning,'publish_update',lambda *_a,**_kw:pytest.fail('rejected candidate published'))
    coordinator._run(threading.Event())
    assert coordinator.ledger['round']==1 and coordinator.ledger['cursor']==8 and coordinator.ledger['rejections']==1
    assert coordinator._registry().head()==head


def test_corpus_boundary_and_failed_round_do_not_lose_examples(monkeypatch,tmp_path):
    coordinator,controls,head,calls=setup(monkeypatch,tmp_path)
    coordinator._initialize();coordinator._save(cursor=192);coordinator._prepare_round()
    pending=coordinator.ledger['pending']
    assert [(row['start'],row['count']) for row in pending['plan']]==[(192,4),(196,4)]
    assert coordinator.ledger['cursor']==192
    # A fully evaluated rejection consumes these examples without promoting.
    coordinator._finish(False)
    assert coordinator.ledger['cursor']==0 and coordinator.ledger['epoch']==1
    assert coordinator.ledger['round']==1 and coordinator._registry().head()==head


def test_old_controls_migrate_without_enabling_observed_allocation(monkeypatch,tmp_path):
    coordinator,controls,head,calls=setup(monkeypatch,tmp_path)
    old=asdict(controls);old['version']=1;old.pop('shards_per_cycle');old.pop('allocation')
    path=coordinator.directory/'language-learning-controls.json';path.write_text(json.dumps(old))
    migrated=learning.LanguageLearningSettings.load(coordinator.directory)
    assert migrated.enabled==controls.enabled and migrated.version==2
    assert migrated.shards_per_cycle==2 and migrated.allocation=='balanced'
    assert migrated.workers==controls.workers and migrated.signing_key_file==controls.signing_key_file


def test_withdrawn_remote_capacity_waits_without_dispatch_or_cursor_change(monkeypatch,tmp_path):
    coordinator,controls,head,calls=setup(monkeypatch,tmp_path)
    coordinator.knowledge.node.peers=lambda:[{'node_id':controls.workers[1],'resources':{'max_gpu_jobs':0,'work':{'training':False}}}]
    coordinator._run(threading.Event())
    assert coordinator.ledger['status']=='waiting_capacity' and coordinator.ledger['pending'] is None
    assert coordinator.ledger['cursor']==0 and calls==[]


def test_coordinator_plans_disjoint_work_using_verified_capacity_history(monkeypatch,tmp_path):
    coordinator,controls,head,calls=setup(monkeypatch,tmp_path)
    coordinator.settings=replace(controls,allocation='observed',shards_per_cycle=4)
    measured={'workers':{worker:{'examples_per_request_second':4 if index==0 else 1}
        for index,worker in enumerate(controls.workers)}}
    monkeypatch.setattr(coordinator.capacity,'estimate',lambda assigned_head,workers:measured if assigned_head==head and workers==controls.workers else pytest.fail('wrong capacity context'))
    coordinator._initialize();coordinator._prepare_round()
    plan=coordinator.ledger['pending']['plan']
    assert [row['worker_id'] for row in plan].count(controls.workers[0])==3
    assert [(row['start'],row['count']) for row in plan]==[(0,4),(4,4),(8,4),(12,4)]
    assert coordinator.ledger['cursor']==0 and coordinator.ledger['pending']['end']==16 and not calls


@pytest.mark.parametrize('change',[{'cycles':11},{'examples_per_worker':9},{'learning_rate':float('nan')},
    {'workers':['a'*64,'b'*64]},{'stop_after_rejections':0}])
def test_unbounded_or_foreign_local_controls_are_rejected(monkeypatch,tmp_path,change):
    _coordinator,controls,_head,_calls=setup(monkeypatch,tmp_path)
    with pytest.raises(ValueError):replace(controls,**change).validate()
