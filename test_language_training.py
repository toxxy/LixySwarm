from types import SimpleNamespace

import pytest

from src.network.language_training import LanguageTraining,LanguageTrainingSettings
from src.network.client_state import ClientSettings
from src.network.compute_process import ComputeProcessError
from src.learning.public_conversation import public_corpus,corpus_digest
from src.release.language_family import ADAPTERS


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
    def register(name,kind,handler,**kwargs):handlers.update(kwargs)
    knowledge=SimpleNamespace(allowed=True,work=SimpleNamespace(register_handler=register))
    service=LanguageTraining(knowledge,SimpleNamespace(),tmp_path,{'has_gpu':True,'gpu_vram_gb':8,'cpu_cores':4})
    service.ready=True;service.engine.requesters=['a'*64]
    execution=SimpleNamespace(work=SimpleNamespace(origin_node_id='a'*64))
    output={'adapter_id':ADAPTERS['qwen3-0.6b'],'dataset_id':corpus_digest(public_corpus())}
    permit=handlers['permit_cached']
    assert permit(execution,output)
    service.engine.requesters=[]
    assert not permit(execution,output)
    service.engine.requesters=['a'*64];knowledge.allowed=False
    assert not permit(execution,output)
    knowledge.allowed=True;service.ready=False
    assert not permit(execution,output)
