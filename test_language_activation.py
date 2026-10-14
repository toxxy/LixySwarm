"""Selection and compatibility fixtures; no physical model-learning claim."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import pytest

from src.contribution import ContributionPolicy,ResourceRequirements
from src.network.client_state import ClientSettings
from src.network.language_activation import activate_version
from src.network.language_compute import LanguageComputeService,LanguageSettings
from src.release.language_versions import LanguageVersionRegistry
from test_language_versions import fixture


def setup(monkeypatch,tmp_path):
    root=tmp_path/'node';public=root/'public/language'
    store,first,update,_=fixture(monkeypatch,public);second,_=update(first)
    LanguageVersionRegistry(public,'qwen3-0.6b').install([first,second],store)
    base=root/'base';base.mkdir()
    engine=LanguageSettings(enabled=True,command=[sys.executable],base_name='qwen3-0.6b',base_directory=str(base),
        adapter_file=str(store.object_path(first['release']['model_artifact_id'])),adapter_id=first['release']['model_artifact_id'])
    engine.save(root)
    controls=ClientSettings(policy=ContributionPolicy.for_mode('balanced'));controls.policy.max_ram_gb=6;controls.save(root)
    return root,store,first,second,engine,controls


def test_selection_preserves_all_permissions_and_the_locally_chosen_command(monkeypatch,tmp_path):
    root,store,first,second,engine,controls=setup(monkeypatch,tmp_path)
    before=(root/'controls.json').read_bytes()
    result=activate_version(root,'qwen3-0.6b',second['release']['release_id'],expected_adapter_id=engine.adapter_id)
    selected=LanguageSettings.load(root)
    assert selected.enabled and selected.command==engine.command
    assert selected.adapter_id==second['release']['model_artifact_id']
    assert Path(selected.adapter_file).read_bytes()==store.object_path(selected.adapter_id).read_bytes()
    assert (root/'controls.json').read_bytes()==before and not result['resource_permissions_changed']
    assert LanguageVersionRegistry(root/'public/language','qwen3-0.6b').head()==second


def test_disabled_wrong_parent_or_stale_version_cannot_activate(monkeypatch,tmp_path):
    root,store,first,second,engine,controls=setup(monkeypatch,tmp_path)
    before=(root/'language-engine.json').read_bytes()
    for release,previous in [(first['release']['release_id'],engine.adapter_id),(second['release']['release_id'],'d'*64)]:
        with pytest.raises(ValueError):activate_version(root,'qwen3-0.6b',release,expected_adapter_id=previous)
        assert (root/'language-engine.json').read_bytes()==before
    replace(engine,enabled=False).save(root)
    with pytest.raises(ValueError,match='locally enabled'):
        activate_version(root,'qwen3-0.6b',second['release']['release_id'],expected_adapter_id=engine.adapter_id)
    assert not LanguageSettings.load(root).enabled


def test_operator_change_during_preparation_is_preserved(monkeypatch,tmp_path):
    from src.network import language_activation
    root,store,first,second,engine,controls=setup(monkeypatch,tmp_path)
    def changed(*_a,**_kw):
        replace(engine,enabled=False).save(root)
        return root/'prepared'
    monkeypatch.setattr(language_activation,'materialize_version',changed)
    with pytest.raises(ValueError,match='controls changed'):
        activate_version(root,'qwen3-0.6b',second['release']['release_id'],expected_adapter_id=engine.adapter_id)
    current=LanguageSettings.load(root)
    assert not current.enabled and current.adapter_id==engine.adapter_id


def test_rejected_small_genesis_cannot_be_selected_for_chat(monkeypatch,tmp_path):
    root=tmp_path/'node';store,first,_update,_publisher=fixture(monkeypatch,root/'public/language')
    engine=LanguageSettings(enabled=True,command=[sys.executable],base_name='qwen3-0.6b',base_directory=str(root),
        adapter_file=str(store.object_path(first['release']['model_artifact_id'])),adapter_id=first['release']['model_artifact_id'])
    engine.save(root);ClientSettings().save(root)
    with pytest.raises(ValueError,match='factual preservation'):
        activate_version(root,'qwen3-0.6b',first['release']['release_id'],expected_adapter_id=engine.adapter_id)


def test_downloading_head_does_not_serve_it_until_selection_and_ancestors_still_answer(monkeypatch,tmp_path):
    root,store,first,second,engine,controls=setup(monkeypatch,tmp_path)
    handlers={}
    def register(name,kind,handler,**kwargs):handlers[name]=kwargs
    service=LanguageComputeService(SimpleNamespace(work=SimpleNamespace(register_handler=register)),root,
        {'cpu_cores':8,'has_gpu':True,'gpu_vram_gb':32})
    service.update(controls)
    assert service.approved_models()==[engine.adapter_id]
    execution=SimpleNamespace(deadline=time.time()+30,resource_requirements=lambda:ResourceRequirements(kind='inference',gpu_required=True,ram_gb=6),
        raise_if_cancelled=lambda:None)
    payload={'model_artifact_id':second['release']['model_artifact_id'],'messages':[{'role':'user','content':'public fixture question'}]}
    with pytest.raises(ValueError,match='locally approved'):service.inference(payload,execution)
    activate_version(root,'qwen3-0.6b',second['release']['release_id'],expected_adapter_id=engine.adapter_id)
    service.update(controls)
    assert set(service.approved_models())=={engine.adapter_id,second['release']['model_artifact_id']}
    def run(_command,request,_execution,**kwargs):
        assert kwargs['allowed']() and request['adapter_id']==engine.adapter_id
        assert Path(request['adapter_file']).read_bytes()==store.object_path(engine.adapter_id).read_bytes()
        return {'model_id':engine.adapter_id,'answer':'old client remains supported','output_tokens':4}
    monkeypatch.setattr(service.supervisor,'run',run)
    payload['model_artifact_id']=engine.adapter_id
    assert service.inference(payload,execution)['model_id']==engine.adapter_id
    permit=handlers['language.inference.v1']['permit_cached']
    assert permit(execution,{'model_id':engine.adapter_id})
    service.ready=False
    assert not permit(execution,{'model_id':engine.adapter_id})
