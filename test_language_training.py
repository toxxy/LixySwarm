from types import SimpleNamespace

import pytest

from src.network.language_training import LanguageTraining,LanguageTrainingSettings
from src.network.client_state import ClientSettings
from src.network.compute_process import ComputeProcessError
from src.learning.public_conversation import public_corpus,corpus_digest
from src.release.language_family import ADAPTERS


def versioned_service(monkeypatch,tmp_path):
    import time
    from src.contribution import ContributionPolicy,ResourceRequirements
    from src.network.artifact_store import ArtifactStore,digest_file
    knowledge=SimpleNamespace(allowed=True,work=SimpleNamespace(register_handler=lambda *_a,**_kw:None))
    families=SimpleNamespace(root=tmp_path/'public/language',store=ArtifactStore(tmp_path/'public/language/artifacts'))
    service=LanguageTraining(knowledge,families,tmp_path,{'has_gpu':True,'gpu_vram_gb':8,'cpu_cores':4})
    service.ready=True;service.engine.requesters=['a'*64];service.settings=ClientSettings(policy=ContributionPolicy.for_mode('maximum'))
    head={'release':{'release_id':'b'*64,'model_artifact_id':ADAPTERS['qwen3-0.6b']}}
    monkeypatch.setattr(service,'_head',lambda:head)
    payload={'base_name':'qwen3-0.6b','release_id':'b'*64,'adapter_id':ADAPTERS['qwen3-0.6b'],
        'dataset_id':corpus_digest(public_corpus()),'start':0,'count':4}
    execution=SimpleNamespace(work=SimpleNamespace(origin_node_id='a'*64,job_id='c'*64,payload=payload),
        resource_requirements=lambda:ResourceRequirements(kind='training',gpu_required=True,ram_gb=4),deadline=time.time()+30)
    return service,head,payload,execution


def test_old_local_controls_migrate_without_enabling_more_resources(tmp_path):
    import json
    old={'version':1,'enabled':False,'command':[],'base_directory':'','adapter_file':'','requesters':[]}
    (tmp_path/'language-training.json').write_text(json.dumps(old))
    migrated=LanguageTrainingSettings.load(tmp_path)
    assert migrated.version==2 and migrated.base_name=='qwen3-0.6b' and not migrated.enabled


def test_unapproved_versions_and_peer_paths_reject_before_materialization(monkeypatch,tmp_path):
    from src.network import language_install
    service,head,payload,execution=versioned_service(monkeypatch,tmp_path)
    monkeypatch.setattr(language_install,'materialize_version',lambda *_a,**_k:pytest.fail('must reject before reading models'))
    payload['release_id']='d'*64
    with pytest.raises(ValueError,match='stale'):service.versioned_gradient(payload,execution)
    payload['release_id']=head['release']['release_id'];payload['base_directory']='private'
    with pytest.raises(ValueError,match='payload'):service.versioned_gradient(payload,execution)


def test_head_change_during_compute_prevents_gradient_commit_and_cache_reuse(monkeypatch,tmp_path):
    from pathlib import Path
    from src.network import language_install
    from src.network.artifact_store import digest_file
    from src.learning.public_bases import base_hash
    service,head,payload,execution=versioned_service(monkeypatch,tmp_path)
    monkeypatch.setattr(language_install,'materialize_version',lambda *_a,**_kw:tmp_path)
    def run(command,request,execution,**kwargs):
        path=Path(request['output_file']);path.write_bytes(b'synthetic gradient fixture')
        assert kwargs['allowed']()
        head['release']['release_id']='e'*64
        return {**{k:payload[k] for k in ['adapter_id','dataset_id','start','count']},
            'base_hash':base_hash('qwen3-0.6b'),'tokens':16,'loss':.2,'retention_loss':.1,
            'gradient_id':digest_file(path)[0],'gradient_bytes':path.stat().st_size}
    monkeypatch.setattr(service.supervisor,'run',run)
    assert service._permit_versioned_cached(execution,{'adapter_id':payload['adapter_id'],'dataset_id':payload['dataset_id']})
    with pytest.raises(ComputeProcessError,match='version changed'):service.versioned_gradient(payload,execution)
    assert not service._permit_versioned_cached(execution,{'adapter_id':payload['adapter_id'],'dataset_id':payload['dataset_id']})
    assert service.completed==0 and service.families.store.total_bytes()==0


def test_large_family_uses_current_adapter_and_requires_its_larger_lease(monkeypatch,tmp_path):
    from pathlib import Path
    from src.contribution import ResourceRequirements
    from src.network import language_install
    from src.network.artifact_store import digest_file
    from src.learning.public_bases import base_hash
    service,head,payload,execution=versioned_service(monkeypatch,tmp_path)
    service.engine.base_name='qwen3-1.7b'
    head['release']['model_artifact_id']='f'*64
    payload.update(base_name='qwen3-1.7b',adapter_id='f'*64)
    monkeypatch.setattr(language_install,'materialize_version',lambda *_a,**_k:pytest.fail('insufficient RAM must reject first'))
    with pytest.raises(ComputeProcessError,match='full GPU/RAM lease'):service.versioned_gradient(payload,execution)
    execution.resource_requirements=lambda:ResourceRequirements(kind='training',gpu_required=True,ram_gb=6)
    monkeypatch.setattr(language_install,'materialize_version',lambda *_a,**_kw:tmp_path/'large')
    def run(command,request,execution,**kwargs):
        assert request['action']=='gradient' and request['base_name']=='qwen3-1.7b'
        assert request['base_directory']==str(tmp_path/'large')
        assert request['adapter_file']==str(tmp_path/'large/adapter.safetensors')
        assert kwargs['ram_gb']==6 and kwargs['allowed']()
        path=Path(request['output_file']);path.write_bytes(b'synthetic gradient fixture')
        return {**{k:payload[k] for k in ['adapter_id','dataset_id','start','count']},
            'base_hash':base_hash('qwen3-1.7b'),'tokens':16,'loss':.2,'retention_loss':.1,
            'gradient_id':digest_file(path)[0],'gradient_bytes':path.stat().st_size}
    monkeypatch.setattr(service.supervisor,'run',run)
    result=service.versioned_gradient(payload,execution)
    assert result['adapter_id']=='f'*64 and service.families.store.has(result['gradient_id'])
    assert service.completed==1
    assert not service._permit_cached(execution,result)


def test_unknown_requester_or_peer_supplied_data_cannot_trigger_language_training(tmp_path):
    handlers={};knowledge=SimpleNamespace(allowed=True,work=SimpleNamespace(register_handler=lambda name,kind,handler,**kw:handlers.update({name:handler})))
    service=LanguageTraining(knowledge,SimpleNamespace(),tmp_path,{'has_gpu':True,'gpu_vram_gb':8,'cpu_cores':4})
    service.ready=True;service.engine.requesters=['a'*64]
    execution=SimpleNamespace(work=SimpleNamespace(origin_node_id='b'*64),resource_requirements=lambda:SimpleNamespace(gpu_required=True,ram_gb=4))
    payload={'adapter_id':ADAPTERS['qwen3-0.6b'],'dataset_id':corpus_digest(public_corpus()),'start':0,'count':1}
    with pytest.raises(ComputeProcessError,match='known requester'):service.gradient(payload,execution)
    payload['dataset_id']='c'*64
    with pytest.raises(ValueError,match='public.*corpus'):service.gradient(payload,execution)
    payload['personal_path']='private-data'
    with pytest.raises(ValueError,match='payload'):service.gradient(payload,execution)


def test_disabled_or_invalid_local_training_controls_do_not_become_ready(tmp_path):
    settings=LanguageTrainingSettings();settings.save(tmp_path)
    assert not LanguageTrainingSettings.load(tmp_path).enabled
    settings.enabled=True
    with pytest.raises(ValueError):settings.save(tmp_path)


def test_cached_gradient_is_unavailable_after_requester_or_permission_withdrawal(tmp_path):
    handlers={}
    def register(name,kind,handler,**kwargs):handlers[name]=kwargs
    knowledge=SimpleNamespace(allowed=True,work=SimpleNamespace(register_handler=register))
    service=LanguageTraining(knowledge,SimpleNamespace(),tmp_path,{'has_gpu':True,'gpu_vram_gb':8,'cpu_cores':4})
    service.ready=True;service.engine.requesters=['a'*64]
    execution=SimpleNamespace(work=SimpleNamespace(origin_node_id='a'*64))
    output={'adapter_id':ADAPTERS['qwen3-0.6b'],'dataset_id':corpus_digest(public_corpus())}
    permit=handlers['language.gradient.v1']['permit_cached']
    assert permit(execution,output)
    service.engine.requesters=[]
    assert not permit(execution,output)
    service.engine.requesters=['a'*64];knowledge.allowed=False
    assert not permit(execution,output)
    knowledge.allowed=True;service.ready=False
    assert not permit(execution,output)
