"""Synthetic signed histories; no extra physical replicas are claimed."""
import pytest

from src.network.artifact_store import ArtifactError,ArtifactStore
from src.network.language_replication import replicate_version
from src.release.language_versions import LanguageVersionRegistry
from src.release.manifest import ReleaseError
from test_language_versions import fixture


def setup(monkeypatch,tmp_path):
    from src.network import language_replication
    source,first,update,_=fixture(monkeypatch,tmp_path/'source');second,report=update(first)
    target=ArtifactStore(tmp_path/'receiver/artifacts');calls=[]
    monkeypatch.setattr(language_replication,'fetch_history',lambda _work,_name,provider,**_kw:[first,second])
    class Transfer:
        def __init__(self,*_a,**_kw):pass
        def fetch(self,identifier,**kwargs):
            provider=kwargs['peer_id'];calls.append((provider,identifier))
            if provider=='unavailable':raise ArtifactError('fixture provider disconnected')
            manifest=source.manifest(identifier)
            partial,offset=target.begin_receive(manifest)
            with partial.open('ab' if offset else 'wb') as output:
                output.write(source.object_path(identifier).read_bytes()[offset:])
            return target.finalize_receive(manifest,partial)
    monkeypatch.setattr(language_replication,'ArtifactService',Transfer)
    return source,target,first,second,report,calls,update


def test_download_can_fail_over_and_restart_preserves_the_verified_head(monkeypatch,tmp_path):
    source,target,first,second,report,calls,_update=setup(monkeypatch,tmp_path)
    result=replicate_version(None,target,tmp_path/'receiver','qwen3-0.6b',['unavailable','working'])
    assert result['release_id']==second['release']['release_id']
    assert set(result['acquired_objects'].values())=={'working'}
    assert calls[0][1]==second['descriptor']['evaluation_id']
    assert not result['active_expert_changed'] and not result['resource_permissions_changed']
    assert LanguageVersionRegistry(tmp_path/'receiver','qwen3-0.6b').head()==second
    calls.clear()
    again=replicate_version(None,target,tmp_path/'receiver','qwen3-0.6b',['working'])
    assert again['acquired_objects']=={} and calls==[]


def test_missing_content_is_repaired_without_reverting_a_newer_head(monkeypatch,tmp_path):
    source,target,first,second,report,calls,_update=setup(monkeypatch,tmp_path)
    replicate_version(None,target,tmp_path/'receiver','qwen3-0.6b',['working'])
    model=second['release']['model_artifact_id'];target.object_path(model).write_bytes(b'corrupt public fixture')
    from src.network import language_replication
    monkeypatch.setattr(language_replication,'fetch_history',lambda *_a,**_kw:[first])
    calls.clear();result=replicate_version(None,target,tmp_path/'receiver','qwen3-0.6b',['working'])
    assert result['release_id']==second['release']['release_id'] and result['acquired_objects']=={model:'working'}
    assert target.object_path(model).read_bytes()==source.object_path(model).read_bytes()


def test_storage_preflight_and_permission_withdrawal_prevent_activation(monkeypatch,tmp_path):
    source,target,first,second,report,calls,_update=setup(monkeypatch,tmp_path)
    def denied(_chain):raise ValueError('operator storage limit')
    with pytest.raises(ValueError,match='storage'):
        replicate_version(None,target,tmp_path/'receiver','qwen3-0.6b',['working'],preflight=denied)
    assert calls==[]
    with pytest.raises(ArtifactError,match='permission'):
        replicate_version(None,target,tmp_path/'receiver','qwen3-0.6b',['working'],allowed=lambda:False)
    assert LanguageVersionRegistry(tmp_path/'receiver','qwen3-0.6b').head()==first


def test_valid_signed_forks_are_not_silently_chosen(monkeypatch,tmp_path):
    from src.network import language_replication
    from src.release.language_versions import verify_chain
    source,target,first,second,report,calls,update=setup(monkeypatch,tmp_path)
    fork,_report=update(first)
    histories={'one':[first,second],'two':[first,fork]}
    def history(_work,_name,provider,**_kw):
        verify_chain(histories[provider]);return histories[provider]
    monkeypatch.setattr(language_replication,'fetch_history',history)
    with pytest.raises(ReleaseError,match='fork'):
        replicate_version(None,target,tmp_path/'receiver','qwen3-0.6b',['one','two'])
    assert not calls


def test_invalid_evaluation_prevents_model_download_and_head_change(monkeypatch,tmp_path):
    from src.network import language_replication
    source,target,first,second,report,calls,_update=setup(monkeypatch,tmp_path)
    def reject(*_a):raise ReleaseError('fixture quality regression')
    monkeypatch.setattr(language_replication,'verify_evaluation',reject)
    with pytest.raises(ReleaseError,match='regression'):
        replicate_version(None,target,tmp_path/'receiver','qwen3-0.6b',['working'])
    assert calls==[('working',second['descriptor']['evaluation_id'])]
    assert not target.has(second['release']['model_artifact_id'])
    assert LanguageVersionRegistry(tmp_path/'receiver','qwen3-0.6b').head()==first


def test_encrypted_transfer_recovers_missing_object_from_second_provider(monkeypatch,tmp_path):
    import threading
    from types import SimpleNamespace
    from src.contribution import ContributionPolicy,ResourceGovernor
    from src.network.language_families import LanguageFamilies
    from src.network.lsp import LSPIdentity
    from src.network.lsp_v3 import LSPNodeV3
    from src.network.work_protocol import WorkCoordinator
    source,first,update,_=fixture(monkeypatch,tmp_path/'source');second,_report=update(first)
    nodes=[];works=[]
    try:
        for label in ['first','second','receiver']:
            directory=tmp_path/label
            node=LSPNodeV3(LSPIdentity.generate(),host='127.0.0.1',port=0,target_outbound=0,
                address_book_path=directory/'peers.json',allow_private=True)
            node.start();nodes.append(node)
            works.append(WorkCoordinator(node,ResourceGovernor(ContributionPolicy.for_mode('relay'),
                hardware=SimpleNamespace(cpu_cores=1,ram_gb=1,disk_gb=10,has_gpu=False,gpu_vram_gb=0),storage_path=directory),max_offers_per_minute=600))
        for index in [0,1]:
            knowledge=SimpleNamespace(work=works[index],allowed=True,cancel=threading.Event(),
                budget=SimpleNamespace(wait=lambda *_a,**_kw:None))
            family=LanguageFamilies(knowledge,tmp_path/['first','second'][index])
            family.store=source;family.artifacts.store=source
            LanguageVersionRegistry(family.root,'qwen3-0.6b').install([first,second],source)
        def unavailable(_payload,_execution):raise ArtifactError('fixture first provider lost this object')
        works[0].register_handler('language.artifact.read-chunk.v1','artifact',unavailable)
        for node in nodes[:2]:assert nodes[2].connect_peer('127.0.0.1',node.port)
        target=ArtifactStore(tmp_path/'receiver/artifacts')
        result=replicate_version(works[2],target,tmp_path/'receiver','qwen3-0.6b',[node.identity.node_id_hex for node in nodes[:2]])
        assert result['release_id']==second['release']['release_id']
        assert set(result['acquired_objects'].values())=={nodes[1].identity.node_id_hex}
        assert all(peer['encrypted'] for peer in nodes[2].peers())
    finally:
        for work in works:work.close()
        for node in nodes:node.stop()
