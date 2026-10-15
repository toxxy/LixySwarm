import hashlib
from types import SimpleNamespace

import pytest

from src.network import language_install as installation
from src.network.artifact_store import ArtifactStore
from src.network.public_knowledge import managed_public_bytes


def fixture(monkeypatch,tmp_path):
    store=ArtifactStore(tmp_path/'public/language/artifacts')
    objects=[]
    for name,content in [('config.json',b'public-config'),('model.safetensors',b'public-model'),('adapter.safetensors',b'public-adapter')]:
        source=tmp_path/name;source.write_bytes(content);manifest=store.import_file(source)
        objects.append((name,manifest.artifact_id,len(content)))
    release=SimpleNamespace(release_id='a'*64,model_artifact_id=objects[-1][1])
    descriptor={'files':[{'name':name,'artifact_id':identifier,'bytes':size} for name,identifier,size in objects[:-1]],'adapter_bytes':objects[-1][2]}
    monkeypatch.setattr(installation,'load_bundle',lambda *_: {})
    monkeypatch.setattr(installation,'verify_bundle',lambda *_:(release,descriptor))
    monkeypatch.setattr(installation,'dependencies',lambda *_:[(identifier,size) for _,identifier,size in objects])
    return store,objects


def test_verified_files_materialize_without_changing_active_expert_or_resource_controls(monkeypatch,tmp_path):
    store,objects=fixture(monkeypatch,tmp_path)
    previous=b'operator-owned-current-settings'
    (tmp_path/'language-engine.json').write_bytes(previous);(tmp_path/'controls.json').write_bytes(previous)
    before=managed_public_bytes(tmp_path)
    target=installation.materialize_family(tmp_path,'fixture')
    assert (target/'model.safetensors').read_bytes()==b'public-model'
    assert (tmp_path/'language-engine.json').read_bytes()==previous
    assert (tmp_path/'controls.json').read_bytes()==previous
    assert installation.materialize_family(tmp_path,'fixture')==target
    if (target/'model.safetensors').stat().st_ino==store.object_path(objects[1][1]).stat().st_ino:
        assert managed_public_bytes(tmp_path)==before


def test_incomplete_corrupt_or_cancelled_family_is_not_exposed(monkeypatch,tmp_path):
    store,objects=fixture(monkeypatch,tmp_path)
    with pytest.raises(RuntimeError,match='detenida'):installation.materialize_family(tmp_path,'fixture',allowed=lambda:False)
    store.object_path(objects[1][1]).write_bytes(b'corrupted-model')
    with pytest.raises(ValueError,match='verificada'):installation.materialize_family(tmp_path,'fixture')
    assert not (tmp_path/'language-models'/('a'*64)).exists()


def test_only_current_verified_version_materializes(monkeypatch,tmp_path):
    from src.release import language_versions
    store,objects=fixture(monkeypatch,tmp_path)
    release,descriptor=installation.verify_bundle({})
    head={'release':{'release_id':release.release_id}}
    monkeypatch.setattr(language_versions,'LanguageVersionRegistry',lambda *_:SimpleNamespace(head=lambda:head))
    monkeypatch.setattr(language_versions,'validate_bundle',lambda *_:(release,descriptor))
    monkeypatch.setattr(language_versions,'dependencies',lambda *_:[(identifier,size) for _,identifier,size in objects])
    with pytest.raises(ValueError,match='current verified'):
        installation.materialize_version(tmp_path,'fixture','b'*64)
    assert not (tmp_path/'language-models').exists()
    target=installation.materialize_version(tmp_path,'fixture',release.release_id)
    assert (target/'adapter.safetensors').read_bytes()==b'public-adapter'
    with pytest.raises(RuntimeError,match='detenida'):
        installation.materialize_version(tmp_path,'fixture',release.release_id,allowed=lambda:False)


def test_model_link_probe_cleans_up_and_required_links_never_fall_back_to_an_unreserved_copy(monkeypatch,tmp_path):
    store,objects=fixture(monkeypatch,tmp_path)
    assert installation.model_hardlinks_available(tmp_path,store)
    assert not list(store.incoming.glob('.link-probe-*'))
    assert not list((tmp_path/'language-models').glob('.link-probe-*'))
    release,descriptor=installation.verify_bundle({})
    def failed_link(*a,**kw):raise OSError('link support disappeared')
    monkeypatch.setattr(installation.os,'link',failed_link)
    assert not installation.model_hardlinks_available(tmp_path,store)
    def no_copy(*a,**kw):pytest.fail('A required-link plan cannot silently allocate a second model')
    monkeypatch.setattr(installation.shutil,'copyfile',no_copy)
    with pytest.raises(ValueError,match='reserva'):
        installation._materialize(tmp_path,store,{},release,descriptor,[(i,n) for _,i,n in objects],lambda:True,require_hardlinks=True)
    assert not (tmp_path/'language-models'/release.release_id).exists()
    assert all(store.has(identifier) for _,identifier,_ in objects)
