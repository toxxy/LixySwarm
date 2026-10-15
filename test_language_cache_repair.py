"""Cache-only recovery preserves expert selection and operator resource consent."""
import json
import threading
from types import SimpleNamespace

import pytest

from src.contribution import ContributionPolicy
from src.network.artifact_store import ArtifactStore,ArtifactError
from src.network.client_state import ClientSettings
from src.network.language_cache_repair import LanguageCacheRepair
from src.network.lsp import LSPIdentity
from src.release.language_versions import LanguageVersionRegistry,dependencies
from test_language_versions import fixture


def repair_fixture(monkeypatch,tmp_path):
    from src.network import language_replication
    source,first,update,_=fixture(monkeypatch,tmp_path/'source');second,_=update(first)
    directory=tmp_path/'node';root=directory/'public/language';store=ArtifactStore(root/'artifacts')
    for identifier in {i for bundle in [first,second] for i,_ in dependencies(bundle)}:
        m=source.manifest(identifier);store.import_file(source.object_path(identifier),kind=m.kind,media_type=m.media_type)
    registry=LanguageVersionRegistry(root,'qwen3-0.6b');registry.install([first,second],store)
    model=second['release']['model_artifact_id'];store.object_path(model).unlink()
    settings=ClientSettings(policy=ContributionPolicy.for_mode('relay'));settings.save(directory)
    (directory/'language-engine.json').write_bytes(b'operator selected expert; preserve')
    (directory/'language-training.json').write_bytes(b'operator disabled training; preserve')
    failures=[False];calls=[]
    class Transfer:
        def __init__(self,*a,**kw):pass
        def fetch(self,identifier,**kw):
            calls.append(identifier)
            if failures[0]:raise ArtifactError('synthetic provider disconnected')
            m=source.manifest(identifier);part,offset=store.begin_receive(m)
            with part.open('ab' if offset else 'wb') as f:f.write(source.object_path(identifier).read_bytes()[offset:])
            return store.finalize_receive(m,part)
    monkeypatch.setattr(language_replication,'ArtifactService',Transfer)
    monkeypatch.setattr(language_replication,'fetch_history',lambda *a,**k:pytest.fail('Cache repair changed metadata policy'))
    node=SimpleNamespace(identity=LSPIdentity.generate(),peers=lambda:[],connect_node=lambda *a,**kw:True)
    knowledge=SimpleNamespace(allowed=True,work=None)
    families=SimpleNamespace(root=root,store=store)
    return directory,model,store,node,knowledge,families,failures,calls


def finish(service):
    service.thread.join(timeout=5);assert not service.thread.is_alive()


def test_idle_node_does_not_create_or_repeat_a_repair_request(monkeypatch,tmp_path):
    d,model,store,node,knowledge,families,failures,calls=repair_fixture(monkeypatch,tmp_path)
    service=LanguageCacheRepair(node,knowledge,families,d)
    try:
        for _ in range(5):service.update()
        assert service.snapshot()['state']=='idle' and service.thread is None
        assert not service.path.exists() and not service.intent.exists() and not calls
    finally:service.close()


def test_cache_repair_needs_no_compute_engine_and_changes_no_selected_expert_or_permissions(monkeypatch,tmp_path):
    d,model,store,node,knowledge,families,failures,calls=repair_fixture(monkeypatch,tmp_path)
    before={name:(d/name).read_bytes() for name in ['controls.json','language-engine.json','language-training.json']}
    head=(families.root/'versions/qwen3-0.6b/head.json').read_bytes()
    service=LanguageCacheRepair(node,knowledge,families,d)
    try:
        service.start('qwen3-0.6b');finish(service)
        assert service.snapshot()['state']=='complete' and store.has(model) and calls==[model]
        assert {name:(d/name).read_bytes() for name in before}==before
        assert (families.root/'versions/qwen3-0.6b/head.json').read_bytes()==head
        assert not service.snapshot()['active_expert_changed']
        assert not json.loads(service.intent.read_text())['requested']
    finally:service.close()


def test_restart_retries_a_disconnected_provider_without_discarding_the_request(monkeypatch,tmp_path):
    d,model,store,node,knowledge,families,failures,calls=repair_fixture(monkeypatch,tmp_path)
    failures[0]=True;first=LanguageCacheRepair(node,knowledge,families,d)
    first.start('qwen3-0.6b');finish(first)
    assert first.snapshot()['state']=='waiting' and json.loads(first.intent.read_text())['requested']
    first.close();failures[0]=False
    second=LanguageCacheRepair(node,knowledge,families,d)
    try:
        second.update();finish(second)
        assert second.snapshot()['state']=='complete' and store.has(model)
    finally:second.close()


def test_storage_limit_and_withdrawn_permission_prevent_any_repair_download(monkeypatch,tmp_path):
    d,model,store,node,knowledge,families,failures,calls=repair_fixture(monkeypatch,tmp_path)
    settings=ClientSettings.load(d);settings.policy.max_disk_gb=.0001;settings.save(d)
    service=LanguageCacheRepair(node,knowledge,families,d)
    try:
        with pytest.raises(ValueError,match='espacio'):service.start('qwen3-0.6b')
        assert not calls and not store.has(model)
        settings.paused=True;settings.save(d)
        with pytest.raises(ValueError,match='Conecta'):service.start('qwen3-0.6b')
        assert not calls
    finally:service.close()
