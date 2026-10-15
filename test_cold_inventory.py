"""Cache eviction requires complete evidence and survives an interrupted unlink."""
from copy import deepcopy
import os
from pathlib import Path

import pytest

from experiments import cold_inventory as cold
from src.network.artifact_store import ArtifactStore
from src.network.lsp import LSPIdentity
from src.release.language_versions import LanguageVersionRegistry,dependencies
from test_cold_recovery import checkpoint
from test_language_versions import fixture


def state(monkeypatch,tmp_path):
    source,first,_,_=fixture(monkeypatch,tmp_path/'source')
    monkeypatch.setattr(cold,'FAMILIES',('qwen3-0.6b',))
    node=tmp_path/'node';store=ArtifactStore(node/'public/language/artifacts')
    for identifier,_ in dependencies(first):
        m=source.manifest(identifier);store.import_file(source.object_path(identifier),kind=m.kind)
    LanguageVersionRegistry(node/'public/language','qwen3-0.6b').install([first],store)
    checkpoint(node,LSPIdentity.generate(),42)
    (node/'personal.enc').write_bytes(b'private canary stays local')
    (node/'controls.json').write_bytes(b'operator controls stay intact')
    folder=node/'language-models'/first['release']['release_id'];folder.mkdir(parents=True)
    os.link(store.object_path(first['release']['model_artifact_id']),folder/'adapter.safetensors')
    planned=cold.inventory(node);provider=deepcopy(planned);provider['node_id']=LSPIdentity.generate().node_id_hex
    return node,store,planned,provider


def test_incomplete_or_same_identity_replica_cannot_authorize_eviction(monkeypatch,tmp_path):
    node,store,planned,provider=state(monkeypatch,tmp_path)
    for bad in [planned,{**provider,'verified_transferable':[]}]:
        with pytest.raises(ValueError,match='distinct'):cold.evict_weights(node,planned,bad)
    assert all(store.object_path(i).exists() for i in planned['weights'])


def test_external_hardlink_blocks_all_eviction_before_the_first_unlink(monkeypatch,tmp_path):
    node,store,planned,provider=state(monkeypatch,tmp_path)
    identifier=next(iter(planned['weights']));os.link(store.object_path(identifier),tmp_path/'external-copy')
    with pytest.raises(ValueError,match='external hardlink'):cold.evict_weights(node,planned,provider)
    assert all(store.object_path(i).exists() for i in planned['weights'])


def test_corrupt_weight_aborts_before_any_good_weight_is_removed(monkeypatch,tmp_path):
    node,store,planned,provider=state(monkeypatch,tmp_path)
    identifier=next(iter(planned['weights']));path=store.object_path(identifier)
    with path.open('r+b') as f:f.write(b'X')
    with pytest.raises(ValueError,match='signed descriptor'):cold.evict_weights(node,planned,provider)
    assert all(store.object_path(i).exists() for i in planned['weights'])


def test_power_loss_during_eviction_can_retry_without_touching_other_data(monkeypatch,tmp_path):
    node,store,planned,provider=state(monkeypatch,tmp_path)
    protected={p:p.read_bytes() for p in node.rglob('*') if p.is_file() and str(p.relative_to(node)) not in
        {name for g in planned['inode_groups'] for name in g['paths']}}
    real=Path.unlink;calls=[]
    def interrupted(path,*a,**kw):
        calls.append(path)
        if len(calls)==2:raise OSError('synthetic power loss')
        return real(path,*a,**kw)
    with monkeypatch.context() as scoped:
        scoped.setattr(Path,'unlink',interrupted)
        with pytest.raises(OSError,match='power loss'):cold.evict_weights(node,planned,provider)
    cold.evict_weights(node,planned,provider)
    assert not cold.inventory(node)['verified_present']
    assert {p:p.read_bytes() for p in protected}==protected
    assert cold.evict_weights(node,planned,provider)['removed_public_links']==0
