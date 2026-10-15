"""Failure recovery must preserve public payloads, consent and replay counters."""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.cold_recovery import (accounted_bytes,merge_artifacts,merge_partial,
    merge_engine_downloads,newest_status,restore_counters,repair_view)
from src.network.artifact_store import ArtifactStore
from src.network.lsp import LSPIdentity
from src.network.telemetry import build_status,sign_status,verify_status


def artifact(store,folder,data):
    path=folder/hashlib.sha256(data).hexdigest();path.write_bytes(data)
    return store.import_file(path,kind='model')


def test_merge_interrupted_between_objects_retries_without_copy_or_lost_payload(tmp_path,monkeypatch):
    import experiments.cold_recovery as recovery
    a=ArtifactStore(tmp_path/'trial/public/language/artifacts');b=ArtifactStore(tmp_path/'old/public/language/artifacts')
    values=[artifact(a,tmp_path,data) for data in [b'public-model-one',b'public-model-two']]
    for name,raw in [('controls.json',b'operator-control'),('personal.enc',b'private encrypted canary'),('identity.key',b'own-private-key')]:
        (tmp_path/'old'/name).write_bytes(raw)
    before={p.name:p.read_bytes() for p in (tmp_path/'old').iterdir() if p.is_file()}
    real=recovery.move_owned;calls=[]
    def interrupted(source,destination):
        calls.append(source)
        if len(calls)==2:raise OSError('simulated power loss')
        real(source,destination)
    monkeypatch.setattr(recovery,'move_owned',interrupted)
    with pytest.raises(OSError,match='power loss'):merge_artifacts(a,b,allowed_partials={})
    assert sum(b.has(v.artifact_id) for v in values)==1
    monkeypatch.setattr(recovery,'move_owned',real)
    merge_artifacts(a,b,allowed_partials={});merge_artifacts(a,b,allowed_partials={})
    assert all(b.has(v.artifact_id) for v in values) and not list(a.objects.glob('*/*'))
    assert {p.name:p.read_bytes() for p in (tmp_path/'old').iterdir() if p.is_file()}==before


def test_corrupt_completed_object_is_rejected_before_moving_anything(tmp_path):
    a=ArtifactStore(tmp_path/'a');b=ArtifactStore(tmp_path/'b')
    good=artifact(a,tmp_path,b'good');bad=artifact(a,tmp_path,b'corrupt-me')
    a.object_path(bad.artifact_id).write_bytes(b'corrupted!')
    with pytest.raises(ValueError):merge_artifacts(a,b,allowed_partials={})
    assert a.has(good.artifact_id) and not list(b.objects.glob('*/*'))


def test_partial_recovery_keeps_longer_prefix_and_rejects_a_conflict(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';a.write_bytes(b'abcdefgh');b.write_bytes(b'abcd')
    merge_partial(a,b,maximum=10);assert not a.exists() and b.read_bytes()==b'abcdefgh'
    a.write_bytes(b'abc');merge_partial(a,b,maximum=10);assert not a.exists()
    a.write_bytes(b'abX')
    with pytest.raises(ValueError,match='conflict'):merge_partial(a,b,maximum=10)
    assert a.read_bytes()==b'abX' and b.read_bytes()==b'abcdefgh'


def test_only_signed_dependency_partials_are_adopted_and_opaque_work_is_retained(tmp_path):
    a=ArtifactStore(tmp_path/'a');b=ArtifactStore(tmp_path/'b');identifier=hashlib.sha256(b'complete').hexdigest()
    (a.incoming/(identifier+'.part')).write_bytes(b'comp')
    (a.incoming/'uncommitted-training.tmp').write_bytes(b'retain public optimizer work')
    r=merge_artifacts(a,b,allowed_partials={identifier:8})
    assert (b.incoming/(identifier+'.part')).read_bytes()==b'comp'
    assert r['retained_incoming']==['uncommitted-training.tmp']
    assert (a.incoming/'uncommitted-training.tmp').read_bytes()==b'retain public optimizer work'


def test_quota_counts_shared_inodes_once_across_node_and_research_roots(tmp_path):
    a=tmp_path/'a/public';b=tmp_path/'b/language-models';extra=tmp_path/'research'
    for p in [a,b,extra]:p.mkdir(parents=True)
    (a/'tensor').write_bytes(b'12345678');os.link(a/'tensor',b/'tensor');os.link(a/'tensor',extra/'tensor')
    (extra/'optimizer').write_bytes(b'1234')
    assert accounted_bytes([a.parent,b.parent],[extra])==12


def checkpoint(directory,identity,sequence,state='connected'):
    directory.mkdir(exist_ok=True)
    node=SimpleNamespace(identity=identity,peers=lambda:[])
    status=build_status(node,sequence=sequence,role='participant',state=state,uptime_s=1,
        resources={'cpu_cores':4,'ram_gb':16.,'gpu_vram_gb':8.,'has_gpu':True})
    value=sign_status(identity,status);(directory/'status.json').write_text(json.dumps(value));return value


def test_restore_sequence_preserves_signed_checkpoint_and_rejects_an_identity_fork(tmp_path):
    identity=LSPIdentity.generate();a=tmp_path/'old';b=tmp_path/'trial'
    checkpoint(a,identity,12345);new=checkpoint(b,identity,12400)
    assert newest_status(a,b,identity.node_id_hex)==12400
    assert verify_status(json.loads((a/'status.json').read_text()))['sequence']==12400
    assert newest_status(a,b,identity.node_id_hex)==12400
    checkpoint(b,LSPIdentity.generate(),99999)
    with pytest.raises(ValueError):newest_status(a,b,identity.node_id_hex)
    assert json.loads((a/'status.json').read_text())==new


def test_counter_merge_is_absolute_and_does_not_reset_or_double_credit(tmp_path):
    a=tmp_path/'old';b=tmp_path/'trial';a.mkdir();b.mkdir();name='language-training-status.json'
    (a/name).write_text('{"completed":84}');(b/name).write_text('{"completed":1}')
    before={name:84}
    assert restore_counters(a,b,before)[name]==85
    assert restore_counters(a,b,before)[name]==85
    (a/name).write_text('{"completed":86}')
    with pytest.raises(ValueError,match='changed'):restore_counters(a,b,before)
    assert json.loads((a/name).read_text())['completed']==86


def test_repair_only_fills_declared_public_view_files_without_replacing_unknown_data(tmp_path):
    store=ArtifactStore(tmp_path/'public/language/artifacts');m=artifact(store,tmp_path,b'known-model')
    release='a'*64;folder=tmp_path/'language-models'/release;folder.mkdir(parents=True)
    (folder/'private.txt').write_bytes(b'never replace this')
    with pytest.raises(ValueError,match='unknown'):repair_view(tmp_path,store,release,{'model.safetensors':(m.artifact_id,m.size)})
    assert (folder/'private.txt').read_bytes()==b'never replace this' and not (folder/'model.safetensors').exists()
    other=repair_view(tmp_path,store,'b'*64,{'model.safetensors':(m.artifact_id,m.size)})
    assert (other/'model.safetensors').stat().st_ino==store.object_path(m.artifact_id).stat().st_ino


def test_app_pinned_engine_and_its_resumable_cache_move_without_reinstalling(tmp_path):
    from test_engine_download import archive,Response
    from src.network.engine_download import prepare_engine
    source=tmp_path/'trial';target=tmp_path/'old';entry,data=archive()
    command=prepare_engine(entry,source,allowed=lambda:True,rate_mbps=lambda:1000000,
        opener=lambda *a,**kw:Response(data))
    binary=Path(command);inode=binary.stat().st_ino
    (source/'engine-downloads'/(entry.sha256+'.part')).write_bytes(data[:23])
    merge_engine_downloads(source,target,[entry])
    assert (target/'engines'/entry.manifest_sha256/binary.name).stat().st_ino==inode
    assert (target/'engine-downloads'/(entry.sha256+'.part')).read_bytes()==data[:23]
