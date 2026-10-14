"""Synthetic metadata/signature fixtures; no physical-learning claim."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import time
import struct
from pathlib import Path

import pytest

from src.learning.public_conversation import public_corpus,corpus_digest
from src.learning.public_quality import quality_cases
from src.network.artifact_store import ArtifactStore,ArtifactManifest
from src.network.lsp import LSPIdentity
from src.network.work_protocol import WorkResult,ResultReceipt
from src.release import language_family as genesis
from src.release import language_versions as versions
from src.release.manifest import ReleaseManifest,ReleaseError


def fixture(monkeypatch,tmp_path):
    publisher=LSPIdentity.generate();workers=[LSPIdentity.generate(),LSPIdentity.generate()]
    monkeypatch.setattr(genesis,'SIGNER',publisher.node_id_hex)
    monkeypatch.setattr(versions,'WORKERS',{worker.node_id_hex for worker in workers})
    name='qwen3-0.6b';store=ArtifactStore(tmp_path/'artifacts')
    def artifact(content,kind='other'):
        path=store.incoming/str(time.time_ns());path.write_bytes(content)
        return store.commit_generated(path,kind=kind,media_type='application/octet-stream')
    model=artifact(b'g'*4599712,'model');evaluation=artifact(b'legacy public evaluation','evaluation')
    pins={};sizes={}
    for filename in genesis.BASES[name]:
        content=('synthetic public '+filename).encode()
        if filename=='config.json':content=json.dumps({'num_hidden_layers':1,'hidden_size':2,'num_attention_heads':1,'num_key_value_heads':1,'head_dim':2}).encode()
        value=artifact(content)
        pins[filename]=value.artifact_id;sizes[filename]=value.size
    monkeypatch.setattr(genesis,'BASES',{name:pins});monkeypatch.setattr(versions,'BASES',{name:pins})
    monkeypatch.setattr(genesis,'COMMON_SIZES',sizes);monkeypatch.setattr(genesis,'SIZES',{name:{}})
    monkeypatch.setattr(genesis,'ADAPTERS',{name:model.artifact_id})
    monkeypatch.setattr(genesis,'EVALUATION_BYTES',{name:evaluation.size})
    descriptor={'version':1,'base_name':name,'base_hash':genesis.base_hash(name),
        'files':[{'name':key,'artifact_id':value,'bytes':sizes[key]} for key,value in pins.items()],
        'adapter_id':model.artifact_id,'adapter_bytes':model.size,'evaluation_id':evaluation.artifact_id}
    config=artifact(genesis.canonical(descriptor))
    release=ReleaseManifest.create(model_artifact_id=model.artifact_id,model_format='safetensors-v1',sequence=0,
        config_artifact_id=config.artifact_id,tokenizer_artifact_id=pins['tokenizer.json']).sign(publisher)
    first={'release':release.to_dict(),'descriptor':descriptor}
    monkeypatch.setattr(genesis,'INITIAL_BUNDLES',{name:first})
    def update(parent):
        header={'__metadata__':{'format':'lixyswarm-lora-v1','base_model_hash':descriptor['base_hash'],'rank':'8','alpha':'16','targets':'q_proj,v_proj'}}
        values=[]
        for projection in ['q_proj','v_proj']:
            for suffix,shape in [('lora_a',[8,2]),('lora_b',[2,8])]:
                start=len(values)*4;values.extend([.1]*16)
                header[f'model.layers.0.self_attn.{projection}.{suffix}']={'dtype':'F32','shape':shape,'data_offsets':[start,len(values)*4]}
        encoded=json.dumps(header).encode();encoded+=b' '*((-len(encoded))%8)
        model=artifact(struct.pack('<Q',len(encoded))+encoded+struct.pack('<'+'f'*len(values),*values),'model')
        results=[]
        for index,worker in enumerate(workers):
            gradient=artifact(b'gradient'+bytes([index]),'gradient')
            output={'base_hash':descriptor['base_hash'],'adapter_id':parent['release']['model_artifact_id'],
                'dataset_id':corpus_digest(public_corpus()),'start':index*4,'count':4,'tokens':16,'loss':.5,
                'retention_loss':.1,'gradient_id':gradient.artifact_id,'gradient_bytes':gradient.size,'gradient':gradient.to_dict()}
            result=WorkResult(job_id=hashlib.sha256((str(index)+str(time.time_ns())).encode()).hexdigest(),status='ok',output=output)
            result=replace(result,receipt=ResultReceipt.create(result,worker,publisher.node_id_hex).to_dict())
            results.append(result.to_dict())
        rows=[{'case':case['case'],'answer':case['accepted'][0]} for case in quality_cases()]
        sources=Path(__file__).parent/'experiments/reports'
        seed=json.loads((sources/'conversation_adapter_17b_v2_20261014.json').read_text())['candidate']['cases']
        dialogue=json.loads((sources/'conversation_dialogue_17b_v2_20261014.json').read_text())['results']['candidate']['cases']
        instructions=[{'case':row['case'],'answer':row['answer']} for row in seed+dialogue]
        report={'version':1,'gate_id':versions.gate_id(),'private_inputs':False,'base_name':name,'base_hash':descriptor['base_hash'],
            'dataset_id':corpus_digest(public_corpus()),'parent_release_id':parent['release']['release_id'],
            'adapter_id':model.artifact_id,'baseline_nll':.5,'candidate_nll':.4,
            'quality_before':rows,'quality_after':deepcopy(rows),'instruction_before':instructions,
            'instruction_after':deepcopy(instructions),'contributions':results}
        evaluation=artifact(genesis.canonical(report),'evaluation')
        fields={**descriptor,'version':2,'adapter_id':model.artifact_id,'adapter_bytes':model.size,
            'evaluation_id':evaluation.artifact_id,'evaluation_bytes':evaluation.size}
        config=artifact(genesis.canonical(fields))
        release=ReleaseManifest.create(model_artifact_id=model.artifact_id,model_format='safetensors-v1',
            sequence=parent['release']['sequence']+1,previous_release_id=parent['release']['release_id'],
            config_artifact_id=config.artifact_id,tokenizer_artifact_id=pins['tokenizer.json']).sign(publisher)
        return {'release':release.to_dict(),'descriptor':fields},report
    update.workers=workers
    return store,first,update,publisher


def test_signed_history_installs_restarts_and_rejects_rollback_or_fork(monkeypatch,tmp_path):
    store,first,update,publisher=fixture(monkeypatch,tmp_path)
    second,report=update(first)
    registry=versions.LanguageVersionRegistry(tmp_path,'qwen3-0.6b')
    registry.install([first,second],store)
    restored=versions.LanguageVersionRegistry(tmp_path,'qwen3-0.6b')
    assert restored.head()==second
    assert restored.history(second['release']['release_id'])==[first,second]
    with pytest.raises(ReleaseError,match='backwards'):restored.install([first],store)
    fork,_=update(first)
    with pytest.raises(ReleaseError,match='fork'):restored.install([first,fork],store)
    invalid=deepcopy(second);invalid['release']['signatures']=[]
    with pytest.raises(ReleaseError):versions.verify_chain([first,invalid])
    with pytest.raises(ReleaseError):versions.verify_chain([second])


@pytest.mark.parametrize('failure',['private','foreign_data','gate','fact','context','duplicate_receipt','single_worker','same_physical_worker','stale_model','overlap','no_improvement'])
def test_promotion_rejects_false_learning_evidence(monkeypatch,tmp_path,failure):
    _,first,update,publisher=fixture(monkeypatch,tmp_path);second,report=update(first)
    if failure=='private':report['private_inputs']=True
    elif failure=='foreign_data':report['dataset_id']='a'*64
    elif failure=='gate':report['gate_id']='b'*64
    elif failure=='fact':report['quality_after'][0]['answer']='Madrid'
    elif failure=='context':next(row for row in report['instruction_after'] if row['case']=='es_context')['answer']='wrong'
    elif failure=='duplicate_receipt':report['contributions'][1]=report['contributions'][0]
    elif failure=='single_worker':report['contributions']=report['contributions'][:1]
    elif failure=='same_physical_worker':pass
    elif failure=='stale_model':report['contributions'][0]['output']['adapter_id']='b'*64
    elif failure=='overlap':report['contributions'][1]['output']['start']=0
    else:report['candidate_nll']=report['baseline_nll']
    if failure in {'stale_model','overlap','same_physical_worker'}:
        index=0 if failure=='stale_model' else 1
        row=WorkResult.from_dict(report['contributions'][index])
        worker=update.workers[0 if failure=='same_physical_worker' else index]
        report['contributions'][index]=replace(row,receipt=ResultReceipt.create(row,worker,publisher.node_id_hex).to_dict()).to_dict()
    with pytest.raises(ValueError):versions.verify_evaluation(report,first,second)


def test_missing_support_data_leaves_previous_head_unchanged(monkeypatch,tmp_path):
    store,first,update,_=fixture(monkeypatch,tmp_path);second,_=update(first)
    registry=versions.LanguageVersionRegistry(tmp_path,'qwen3-0.6b')
    store.object_path(second['descriptor']['evaluation_id']).write_bytes(b'corrupt')
    with pytest.raises(ReleaseError,match='incomplete or corrupt'):registry.install([first,second],store)
    assert registry.head()==first


def test_update_cannot_replace_pinned_upstream_files_even_with_local_signature(monkeypatch,tmp_path):
    _,first,update,_=fixture(monkeypatch,tmp_path);second,_=update(first)
    second['descriptor']['files'][0]['name']='../personal.enc'
    with pytest.raises(ReleaseError):versions.validate_bundle(second)


def test_complete_version_and_evidence_acquire_over_encrypted_lsp(monkeypatch,tmp_path):
    import threading
    from types import SimpleNamespace
    from src.contribution import ContributionPolicy,ResourceGovernor
    from src.network.lsp_v3 import LSPNodeV3
    from src.network.work_protocol import WorkCoordinator
    from src.network.language_families import LanguageFamilies
    from src.network.language_version_service import acquire_history
    store,first,update,_=fixture(monkeypatch,tmp_path/'source')
    second,_=update(first)
    nodes=[];works=[]
    try:
        for label in ['provider','receiver']:
            root=tmp_path/label;root.mkdir()
            node=LSPNodeV3(LSPIdentity.generate(),host='127.0.0.1',port=0,
                address_book_path=root/'peers.json',target_outbound=0,allow_private=True)
            node.start();nodes.append(node)
            governor=ResourceGovernor(ContributionPolicy.for_mode('relay'),hardware=SimpleNamespace(
                cpu_cores=1,ram_gb=1,has_gpu=False,gpu_vram_gb=0,disk_gb=10),storage_path=root)
            works.append(WorkCoordinator(node,governor,max_offers_per_minute=600))
        knowledge=SimpleNamespace(work=works[0],allowed=True,cancel=threading.Event(),
            budget=SimpleNamespace(wait=lambda *_a,**_kw:None))
        families=LanguageFamilies(knowledge,tmp_path/'provider')
        families.store=store;families.artifacts.store=store
        versions.LanguageVersionRegistry(families.root,'qwen3-0.6b').install([first,second],store)
        assert nodes[1].connect_peer('127.0.0.1',nodes[0].port)
        target=ArtifactStore(tmp_path/'receiver/artifacts')
        acquired=acquire_history(works[1],target,tmp_path/'receiver','qwen3-0.6b',nodes[0].identity.node_id_hex)
        assert acquired.release_id==second['release']['release_id']
        assert target.has(acquired.model_artifact_id)
        restored=versions.LanguageVersionRegistry(tmp_path/'receiver','qwen3-0.6b')
        assert restored.head()==second
        assert nodes[1].peers()[0]['encrypted']
        from src.network.chat_versions import ChatVersions
        light=ChatVersions(tmp_path/'light-client','qwen3-0.6b')
        light.synchronize(works[1],nodes[0].identity.node_id_hex,cancel_event=None,preflight=lambda _size:None)
        assert light.head()==second
        assert light.store.has(second['descriptor']['evaluation_id'])
        assert not light.store.has(second['release']['model_artifact_id'])
    finally:
        for work in works:work.close()
        for node in nodes:node.stop()


def test_publisher_signs_only_after_all_public_gates_pass(monkeypatch,tmp_path):
    store,first,update,publisher=fixture(monkeypatch,tmp_path);second,report=update(first)
    signed=[];original=publisher.sign
    monkeypatch.setattr(publisher,'sign',lambda message:signed.append(True) or original(message))
    adapter=store.manifest(second['release']['model_artifact_id'])
    evaluation=store.manifest(second['descriptor']['evaluation_id'])
    bundle=versions.build_update([first],adapter,evaluation,report,publisher)
    assert versions.verify_chain([first,bundle]).sequence==1 and len(signed)==1
    report['quality_after'][0]['answer']='Madrid'
    path=store.incoming/'bad-evaluation';path.write_bytes(genesis.canonical(report))
    bad=store.commit_generated(path,kind='evaluation',media_type='application/json')
    with pytest.raises(ReleaseError,match='factual'):
        versions.build_update([first],adapter,bad,report,publisher)
    assert len(signed)==1


def test_concurrent_acceptance_cannot_replace_another_transaction_head(monkeypatch,tmp_path):
    from src.network.client_state import NodeStateLock
    store,first,update,_=fixture(monkeypatch,tmp_path);second,_=update(first)
    registry=versions.LanguageVersionRegistry(tmp_path,'qwen3-0.6b')
    with NodeStateLock(registry.root/'transaction'):
        with pytest.raises(RuntimeError):registry.install([first,second],store)
    assert registry.head()==first
    registry.install([first,second],store)
    assert registry.head()==second
