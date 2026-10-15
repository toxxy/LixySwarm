"""Signed precision transitions and compatibility; synthetic metadata fixtures."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import struct
import threading
from types import SimpleNamespace

import pytest

from src.learning.language_policy import (BATCH_POLICY_ID,HALF_POLICY_ID,HALF_CLIENT_VERSION,
    descriptor_policy,policy_from_id)
from src.network.artifact_store import digest_file
from src.network.compute_process import ComputeProcessError
from src.network.language_training import LanguageTrainingSettings
from src.network.work_protocol import WorkResult,ResultReceipt
from src.release import language_versions as versions
from src.release.language_family import canonical
from test_language_batches import batch_fixture
from test_language_training import versioned_service


def half_fixture(monkeypatch,root):
    store,chain,report,publisher,workers=batch_fixture(monkeypatch,root)
    def put(content,kind):
        path=store.incoming/'half-fixture';path.write_bytes(content)
        return store.commit_generated(path,kind=kind,media_type='application/octet-stream')
    parent=chain[-1]
    content=store.object_path(parent['release']['model_artifact_id']).read_bytes()
    model=put(content[:-4]+struct.pack('<f',.12),'model')
    report=deepcopy(report);report.update(version=3,gate_id=versions.gate_id(3),
        parent_release_id=parent['release']['release_id'],adapter_id=model.artifact_id)
    for index,row in enumerate(report['contributions']):
        value=replace(WorkResult.from_dict(row),output={**row['output'],
            'adapter_id':parent['release']['model_artifact_id'],'training_policy_id':HALF_POLICY_ID,
            'quantization_relative_l2_error':.0002})
        report['contributions'][index]=replace(value,receipt=ResultReceipt.create(value,workers[index],publisher.node_id_hex).to_dict()).to_dict()
    evaluation=put(canonical(report),'evaluation')
    bundle=versions.build_update(chain,model,evaluation,report,publisher)
    put(canonical(bundle['descriptor']),'other')
    return store,chain+[bundle],report,publisher,workers


def test_precision_policy_is_explicit_and_legacy_gates_are_immutable(tmp_path):
    assert BATCH_POLICY_ID=='f68c8074efb7ae0a0e5aed7440c7d9092aab6ef5dea6a49ec7766aa8de08935e'
    assert versions.gate_id()=='8b1163d42606b34937f70442e08f81eb024ea8c8099f970e0e4b1827e75f41c7'
    assert versions.gate_id(2)=='54dee8396ee05693b28affb0c08c24910f0e302992a1e1b37d48cb96c2b95a69'
    assert HALF_POLICY_ID!=BATCH_POLICY_ID and versions.gate_id(3)!=versions.gate_id(2)
    old=LanguageTrainingSettings();data=old.__dict__.copy();data.pop('half_precision_enabled')
    (tmp_path/'language-training.json').write_text(json.dumps(data))
    migrated=LanguageTrainingSettings.load(tmp_path)
    assert not migrated.half_precision_enabled and not migrated.batching_enabled
    with pytest.raises(ValueError):policy_from_id('f'*64)


def test_half_release_survives_restart_and_cannot_be_relabelled(monkeypatch,tmp_path):
    store,chain,report,publisher,workers=half_fixture(monkeypatch,tmp_path)
    assert chain[-1]['descriptor']['version']==4 and descriptor_policy(chain[-1]['descriptor'])==3
    assert chain[-1]['release']['minimum_client_version']==HALF_CLIENT_VERSION
    registry=versions.LanguageVersionRegistry(tmp_path,'qwen3-0.6b');registry.install(chain,store)
    assert versions.LanguageVersionRegistry(tmp_path,'qwen3-0.6b').head()==chain[-1]
    assert versions.verify_evaluation(report,chain[-2],chain[-1])['tokens']==256
    with pytest.raises(ValueError):versions.verify_contributions(report['contributions'],base_name='qwen3-0.6b',
        adapter_id=chain[-2]['release']['model_artifact_id'],dataset_id=report['dataset_id'],policy_version=2)
    # Even a correctly signed legacy-style transition cannot downgrade this parent.
    downgraded=deepcopy(report);downgraded.update(version=2,gate_id=versions.gate_id(2),parent_release_id=chain[-1]['release']['release_id'])
    content=store.object_path(chain[-1]['release']['model_artifact_id']).read_bytes()
    path=store.incoming/'downgrade-model';path.write_bytes(content[:-4]+struct.pack('<f',.13))
    model=store.commit_generated(path,kind='model',media_type='application/octet-stream')
    downgraded['adapter_id']=model.artifact_id
    for index,row in enumerate(downgraded['contributions']):
        out={**row['output'],'adapter_id':chain[-1]['release']['model_artifact_id'],'training_policy_id':BATCH_POLICY_ID}
        out.pop('quantization_relative_l2_error')
        value=replace(WorkResult.from_dict(row),output=out)
        downgraded['contributions'][index]=replace(value,receipt=ResultReceipt.create(value,workers[index],publisher.node_id_hex).to_dict()).to_dict()
    versions.verify_contributions(downgraded['contributions'],base_name='qwen3-0.6b',
        adapter_id=chain[-1]['release']['model_artifact_id'],dataset_id=report['dataset_id'],policy_version=2)
    path=store.incoming/'downgrade-evaluation';path.write_bytes(canonical(downgraded))
    evaluation=store.commit_generated(path,kind='evaluation',media_type='application/json')
    with pytest.raises(ValueError,match='transition'):versions.build_update(chain,model,evaluation,downgraded,publisher)


@pytest.mark.parametrize('problem',['missing_error','excess_error','negative_error','nan_error','bool_error','policy','signature','quality','gate'])
def test_half_policy_cannot_bypass_precision_signature_or_quality(monkeypatch,tmp_path,problem):
    _,chain,report,publisher,workers=half_fixture(monkeypatch,tmp_path);report=deepcopy(report)
    row=WorkResult.from_dict(report['contributions'][0]);out=dict(row.output)
    if problem=='missing_error':out.pop('quantization_relative_l2_error')
    elif problem=='excess_error':out['quantization_relative_l2_error']=.002
    elif problem=='negative_error':out['quantization_relative_l2_error']=-.001
    elif problem=='nan_error':out['quantization_relative_l2_error']=float('nan')
    elif problem=='bool_error':out['quantization_relative_l2_error']=True
    elif problem=='policy':out['training_policy_id']=BATCH_POLICY_ID
    elif problem=='signature':out['loss']+=1
    elif problem=='quality':report['quality_after'][0]['answer']='Madrid'
    else:report['gate_id']=versions.gate_id(2)
    row=replace(row,output=out)
    # A nonfinite value cannot even be signed canonically.
    if problem=='nan_error':
        with pytest.raises(ValueError):ResultReceipt.create(row,workers[0],publisher.node_id_hex)
        return
    if problem!='signature':row=replace(row,receipt=ResultReceipt.create(row,workers[0],publisher.node_id_hex).to_dict())
    report['contributions'][0]=row.to_dict()
    with pytest.raises(ValueError):versions.verify_evaluation(report,chain[-2],chain[-1])


def test_worker_and_its_cache_stop_when_half_permission_is_withdrawn(monkeypatch,tmp_path):
    from src.network import language_install
    service,head,payload,execution=versioned_service(monkeypatch,tmp_path)
    payload.update(count=64,training_policy_id=HALF_POLICY_ID)
    with pytest.raises(ComputeProcessError,match='locally enabled'):service.half_gradient(payload,execution)
    service.engine.batching_enabled=True;service.engine.half_precision_enabled=True
    monkeypatch.setattr(language_install,'materialize_version',lambda *_a,**_kw:tmp_path)
    def compute(command,request,context,**kw):
        assert request['action']=='gradient-half' and request['training_policy_id']==HALF_POLICY_ID and kw['allowed']()
        path=Path(request['output_file']);path.write_bytes(b'unit fixture; numerical tensor format is tested separately')
        from src.learning.public_bases import base_hash
        return {'base_hash':base_hash(payload['base_name']),**{k:payload[k] for k in ['adapter_id','dataset_id','start','count','training_policy_id']},
            'tokens':128,'loss':.2,'retention_loss':.1,'gradient_id':digest_file(path)[0],'gradient_bytes':path.stat().st_size,
            'quantization_relative_l2_error':.0002}
    monkeypatch.setattr(service.supervisor,'run',compute)
    result=service.half_gradient(payload,execution)
    assert service.completed==1 and service._permit_versioned_cached(execution,result)
    service.engine.half_precision_enabled=False
    assert not service._permit_versioned_cached(execution,result)
    with pytest.raises(ComputeProcessError):service.half_gradient(payload,execution)


def test_selecting_half_policy_does_not_rearm_a_stopped_learning_window(monkeypatch,tmp_path):
    from test_language_learning import setup
    coordinator,controls,_,calls=setup(monkeypatch,tmp_path)
    coordinator.settings=controls;coordinator._initialize()
    coordinator._save(round=5,cursor=96,rejections=3,status='no_improvement')
    coordinator.training.engine.batching_enabled=True;coordinator.training.engine.half_precision_enabled=True
    changed=replace(controls,cycles=10,training_policy_version=3,examples_per_worker=64)
    changed.save(coordinator.directory);coordinator.update()
    assert coordinator.ledger['round']==5 and coordinator.ledger['cursor']==96 and coordinator.ledger['rejections']==3
    assert coordinator.ledger['status']=='no_improvement' and coordinator.thread is None and not calls


def test_compatible_head_views_and_chat_metadata_use_real_local_lsp(monkeypatch,tmp_path):
    from src.contribution import ContributionPolicy,ResourceGovernor
    from src.network.lsp import LSPIdentity
    from src.network.lsp_v3 import LSPNodeV3
    from src.network.work_protocol import WorkCoordinator
    from src.network.language_families import LanguageFamilies
    from src.network.language_version_service import fetch_history
    from src.network.chat_versions import ChatVersions
    root=tmp_path/'source';store,chain,_,_,_=half_fixture(monkeypatch,root/'public/language')
    versions.LanguageVersionRegistry(root/'public/language','qwen3-0.6b').install(chain,store)
    nodes=[];works=[]
    try:
        for name in ['source','receiver']:
            directory=tmp_path/name
            node=LSPNodeV3(LSPIdentity.generate(),host='127.0.0.1',port=0,address_book_path=directory/'peers.json',target_outbound=0,allow_private=True)
            nodes.append(node)
            governor=ResourceGovernor(ContributionPolicy.for_mode('relay'),hardware=SimpleNamespace(cpu_cores=1,ram_gb=1,disk_gb=1,has_gpu=False,gpu_vram_gb=0),storage_path=directory)
            works.append(WorkCoordinator(node,governor,max_offers_per_minute=600))
        LanguageFamilies(SimpleNamespace(work=works[0],allowed=True,budget=SimpleNamespace(wait=lambda *_a,**_kw:None),cancel=threading.Event()),root)
        for node in nodes:node.start()
        assert nodes[1].connect_peer('127.0.0.1',nodes[0].port)
        provider=nodes[0].identity.node_id_hex
        assert fetch_history(works[1],'qwen3-0.6b',provider)==chain
        batch_view=SimpleNamespace(node=SimpleNamespace(identity=nodes[1].identity,peers=lambda:[{'node_id':provider,'capabilities':{'language_versions_v2':True}}]),submit=works[1].submit)
        assert fetch_history(batch_view,'qwen3-0.6b',provider)==chain[:-1]
        legacy_view=SimpleNamespace(node=SimpleNamespace(identity=nodes[1].identity),submit=works[1].submit)
        assert fetch_history(legacy_view,'qwen3-0.6b',provider)==chain[:-2]
        chat=ChatVersions(tmp_path/'chat','qwen3-0.6b')
        chat.synchronize(works[1],provider,cancel_event=threading.Event(),preflight=lambda *_:None)
        assert chat.head()==chain[-1] and not chat.store.has(chain[-1]['release']['model_artifact_id'])
        from src.network.language_compute import LanguageComputeService
        local=LanguageComputeService(SimpleNamespace(work=SimpleNamespace(register_handler=lambda *_a,**_kw:None)),root,{})
        local.ready=True;local.engine.base_name='qwen3-0.6b';local.engine.adapter_id=chain[-1]['release']['model_artifact_id']
        assert chain[-1]['release']['model_artifact_id'] in local.approved_models()
        assert chain[-1]['release']['model_artifact_id'] not in local.approved_models(maximum_descriptor=3)
        assert chain[-2]['release']['model_artifact_id'] in local.approved_models(maximum_descriptor=3)
    finally:
        for work in works:work.close()
        for node in nodes:node.stop()
