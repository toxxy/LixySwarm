"""Versioned useful batches, immutable legacy gates and old/new discovery views."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import struct
import threading
from types import SimpleNamespace

import pytest

from src.contribution import ContributionPolicy, ResourceGovernor
from src.learning.language_capacity import allocate_shards
from src.learning.language_policy import BATCH_POLICY_ID, BATCH_CLIENT_VERSION, retention_plan
from src.network.artifact_store import digest_file
from src.network.client_state import ClientSettings
from src.network.compute_process import ComputeProcessError
from src.network.language_learning import LanguageLearningSettings
from src.network.lsp import LSPIdentity
from src.network.lsp_v3 import LSPNodeV3
from src.network.work_protocol import WorkCoordinator, WorkResult, ResultReceipt
from src.release import language_versions as versions
from src.release.language_family import canonical
from test_language_training import versioned_service
from test_language_versions import fixture


def batch_fixture(monkeypatch, root):
    store, initial, update, publisher = fixture(monkeypatch, root)
    previous, _ = update(initial)
    draft, report = update(previous)
    def put(content, kind):
        path=store.incoming/'batch-fixture';path.write_bytes(content)
        return store.commit_generated(path,kind=kind,media_type='application/octet-stream')
    content=store.object_path(draft['release']['model_artifact_id']).read_bytes()
    model=put(content[:-4]+struct.pack('<f', .11),'model')
    report.update(version=2,gate_id=versions.gate_id(2),adapter_id=model.artifact_id)
    for index,row in enumerate(report['contributions']):
        output={**row['output'],'start':index*64,'count':64,'tokens':128,'training_policy_id':BATCH_POLICY_ID}
        result=replace(WorkResult.from_dict(row),output=output)
        report['contributions'][index]=replace(result,receipt=ResultReceipt.create(result,update.workers[index],publisher.node_id_hex).to_dict()).to_dict()
    evaluation=put(canonical(report),'evaluation')
    bundle=versions.build_update([initial,previous],model,evaluation,report,publisher)
    put(canonical(bundle['descriptor']),'other')
    return store,[initial,previous,bundle],report,publisher,update.workers


def test_historical_gate_bytes_and_default_batch_permission_remain_unchanged():
    assert versions.gate_id()=='8b1163d42606b34937f70442e08f81eb024ea8c8099f970e0e4b1827e75f41c7'
    assert versions.gate_id(2)!=versions.gate_id()
    from src.network.language_training import LanguageTrainingSettings
    assert not LanguageTrainingSettings().batching_enabled


def test_batch_versions_preserve_old_evaluation_and_survive_registry_restart(monkeypatch,tmp_path):
    store,chain,report,_,_=batch_fixture(monkeypatch,tmp_path)
    assert chain[-1]['descriptor']['version']==3
    assert chain[-1]['release']['minimum_client_version']==BATCH_CLIENT_VERSION
    registry=versions.LanguageVersionRegistry(tmp_path,'qwen3-0.6b');registry.install(chain,store)
    assert versions.LanguageVersionRegistry(tmp_path,'qwen3-0.6b').head()==chain[-1]
    assert versions.verify_evaluation(report,chain[-2],chain[-1])['tokens']==256
    with pytest.raises(ValueError):
        versions.verify_contributions(report['contributions'],base_name='qwen3-0.6b',adapter_id=chain[-2]['release']['model_artifact_id'],dataset_id=report['dataset_id'])


@pytest.mark.parametrize('problem',['count','count_type','tokens','tokens_per_example','policy','missing_policy','overlap','signature','legacy_gate','quality'])
def test_larger_batches_cannot_bypass_original_quality_or_boundaries(monkeypatch,tmp_path,problem):
    _,chain,report,publisher,workers=batch_fixture(monkeypatch,tmp_path);report=deepcopy(report)
    row=WorkResult.from_dict(report['contributions'][0]);output=dict(row.output)
    if problem=='count':output['count']=65
    elif problem=='count_type':output['count']=True
    elif problem=='tokens':output['tokens']=12289
    elif problem=='tokens_per_example':output.update(count=1,tokens=193)
    elif problem=='policy':output['training_policy_id']='f'*64
    elif problem=='missing_policy':output.pop('training_policy_id')
    elif problem=='overlap':output['start']=64
    elif problem=='signature':output['loss']+=1
    elif problem=='legacy_gate':report['gate_id']=versions.gate_id()
    else:report['quality_after'][0]['answer']='Madrid'
    row=replace(row,output=output)
    if problem!='signature':row=replace(row,receipt=ResultReceipt.create(row,workers[0],publisher.node_id_hex).to_dict())
    report['contributions'][0]=row.to_dict()
    with pytest.raises(ValueError):versions.verify_evaluation(report,chain[-2],chain[-1])


def test_group_retention_is_token_weighted_and_large_ranges_remain_disjoint():
    counts=[1]*8+[3]*8
    assert retention_plan(64,counts)==[(8,.25),(9,.75)]
    plan=allocate_shards(['a'*64,'b'*64],64,3,policy_version=2)
    assert [(row['start'],row['count']) for row in plan]==[(0,64),(64,64),(128,64)]
    assert [row['worker_id'] for row in plan]==['a'*64,'b'*64,'a'*64]
    with pytest.raises(ValueError):allocate_shards(['a'*64,'b'*64],64,3)
    with pytest.raises(ValueError):retention_plan(180,[1]*64)


def test_migrating_a13_coordinator_controls_does_not_enable_large_batches(tmp_path):
    controls=LanguageLearningSettings();old=asdict(controls);old['version']=2;old.pop('training_policy_version')
    (tmp_path/'language-learning-controls.json').write_text(json.dumps(old))
    migrated=LanguageLearningSettings.load(tmp_path)
    assert migrated.version==3 and migrated.training_policy_version==1 and migrated.examples_per_worker==8
    assert not migrated.enabled and migrated.shards_per_cycle==controls.shards_per_cycle


def test_batch_worker_requires_local_permission_and_binds_the_child_policy(monkeypatch,tmp_path):
    from src.network import language_install
    service,head,payload,execution=versioned_service(monkeypatch,tmp_path)
    payload.update(count=64,training_policy_id=BATCH_POLICY_ID)
    with pytest.raises(ComputeProcessError,match='locally enabled'):service.batched_gradient(payload,execution)
    service.engine.batching_enabled=True
    monkeypatch.setattr(language_install,'materialize_version',lambda *_a,**_kw:tmp_path)
    def compute(_command,request,_execution,**kwargs):
        assert request['action']=='gradient-batch' and request['training_policy_id']==BATCH_POLICY_ID
        assert kwargs['allowed']()
        path=Path(request['output_file']);path.write_bytes(b'signed synthetic batch gradient')
        return {'base_hash':__import__('src.learning.public_bases',fromlist=['base_hash']).base_hash(payload['base_name']),
            **{k:payload[k] for k in ['adapter_id','dataset_id','start','count','training_policy_id']},
            'tokens':128,'loss':.2,'retention_loss':.1,'gradient_id':digest_file(path)[0],'gradient_bytes':path.stat().st_size}
    monkeypatch.setattr(service.supervisor,'run',compute)
    result=service.batched_gradient(payload,execution)
    assert result['training_policy_id']==BATCH_POLICY_ID and service.completed==1
    assert service._permit_versioned_cached(execution,result)
    service.engine.batching_enabled=False
    assert not service._permit_versioned_cached(execution,result)
    old={k:v for k,v in payload.items() if k!='training_policy_id'}
    with pytest.raises(ValueError):service.versioned_gradient(old,execution)


def test_withdrawn_local_batch_permission_keeps_pending_attempt_and_cursor(monkeypatch,tmp_path):
    from test_language_learning import setup
    coordinator,controls,head,calls=setup(monkeypatch,tmp_path)
    settings=replace(controls,training_policy_version=2,examples_per_worker=64,shards_per_cycle=3)
    settings.save(coordinator.directory);coordinator.settings=settings
    coordinator.training.engine.batching_enabled=True
    coordinator.knowledge.node.peers=lambda:[{'node_id':settings.workers[1],
        'capabilities':{'language_gradient_batches_v1':True},'resources':{'max_gpu_jobs':1}}]
    coordinator._initialize();coordinator._prepare_round()
    assert coordinator.ledger['pending']['training_policy_version']==2
    coordinator.training.engine.batching_enabled=False
    coordinator._run(threading.Event())
    assert coordinator.ledger['status']=='waiting_capacity' and coordinator.ledger['pending']['attempts']==0
    assert coordinator.ledger['cursor']==0 and not calls


def test_publication_journal_recovers_before_a_new_policy_replans_work(monkeypatch,tmp_path):
    from test_language_learning import setup
    from src.network import language_learning as learning
    coordinator,controls,head,calls=setup(monkeypatch,tmp_path)
    original=learning.publish_update
    def interrupted(*args,**kwargs):
        original(*args,**kwargs);raise RuntimeError('injected after legacy commit')
    monkeypatch.setattr(learning,'publish_update',interrupted)
    coordinator._run(threading.Event())
    assert coordinator.ledger['pending']['bundle'] is not None and coordinator.ledger['round']==0
    changed=replace(controls,training_policy_version=2);changed.save(coordinator.directory)
    restored=learning.LanguageLearningCoordinator(coordinator.knowledge,coordinator.training,coordinator.directory)
    restored.settings=changed;restored._run(threading.Event())
    assert restored.ledger['round']==1 and restored.ledger['cursor']==8
    assert restored.ledger['pending'] is None and len(calls)==3


def test_new_and_legacy_clients_discover_their_supported_head_over_real_lsp(monkeypatch,tmp_path):
    from src.network.language_families import LanguageFamilies
    from src.network.language_version_service import fetch_history
    root=tmp_path/'source';store,chain,report,_,_=batch_fixture(monkeypatch,root/'public/language')
    versions.LanguageVersionRegistry(root/'public/language','qwen3-0.6b').install(chain,store)
    nodes=[];works=[]
    try:
        for name in ['source','receiver']:
            directory=tmp_path/name
            node=LSPNodeV3(LSPIdentity.generate(),host='127.0.0.1',port=0,address_book_path=directory/'peers.json',target_outbound=0,allow_private=True)
            nodes.append(node)
            governor=ResourceGovernor(ContributionPolicy.for_mode('relay'),hardware=SimpleNamespace(cpu_cores=1,ram_gb=1,disk_gb=1,has_gpu=False,gpu_vram_gb=0),storage_path=directory)
            works.append(WorkCoordinator(node,governor))
        knowledge=SimpleNamespace(work=works[0],allowed=True,budget=SimpleNamespace(wait=lambda *_a,**_kw:None),cancel=threading.Event())
        LanguageFamilies(knowledge,root)
        for node in nodes:node.start()
        assert nodes[1].connect_peer('127.0.0.1',nodes[0].port)
        provider=nodes[0].identity.node_id_hex
        assert fetch_history(works[1],'qwen3-0.6b',provider)==chain
        old_view=SimpleNamespace(node=SimpleNamespace(identity=nodes[1].identity),submit=works[1].submit)
        assert fetch_history(old_view,'qwen3-0.6b',provider)==chain[:-1]
        from src.network.chat_versions import ChatVersions
        chat=ChatVersions(tmp_path/'chat','qwen3-0.6b')
        chat.synchronize(works[1],provider,cancel_event=threading.Event(),preflight=lambda *_:None)
        assert chat.head()==chain[-1] and not chat.store.has(chain[-1]['release']['model_artifact_id'])
    finally:
        for work in works:work.close()
        for node in nodes:node.stop()
