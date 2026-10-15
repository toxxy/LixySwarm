"""Interrupted native-trial recovery restores public state and operator consent."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from experiments import cold_native_trial as cold
from experiments.cold_checkpoint import ColdCheckpoint
from src.network.artifact_store import ArtifactStore
from src.network.client_state import ClientSettings,NodeStateLock
from src.network.lsp import LSPIdentity
from src.network.telemetry import atomic_json
from test_cold_inventory import state
from test_cold_recovery import checkpoint
from test_engine_download import archive


def test_kill_after_merge_and_control_suspension_can_recover_without_duplicate_credit(monkeypatch,tmp_path):
    old,store,planned,provider=state(monkeypatch,tmp_path)
    identity=LSPIdentity.generate();checkpoint(old,identity,10)
    ClientSettings().save(old);controls=(old/'controls.json').read_bytes();planned=cold.inventory(old)
    monkeypatch.setattr(cold,'LAPTOP',identity.node_id_hex);monkeypatch.setattr(cold,'PROVIDER',provider['node_id'])
    monkeypatch.setattr(cold,'FAMILIES',('qwen3-0.6b',));monkeypatch.setattr(cold,'extras',lambda:[])
    folder=tmp_path/'test';trial=folder/'node';folder.mkdir()
    with NodeStateLock(old),NodeStateLock(trial):
        c=ColdCheckpoint.capture(old,trial,folder/'checkpoint.json')
        received=ArtifactStore(trial/'public/language/artifacts')
        for identifier in planned['weights']:
            m=store.manifest(identifier);received.import_file(store.object_path(identifier),kind=m.kind)
        for group in planned['inode_groups']:
            for relative in group['paths']:(old/relative).unlink()
    (trial/'language-training-status.json').write_text('{"completed":1}')
    (old/'language-training-status.json').write_text('{"completed":0}')
    checkpoint(trial,identity,20)
    session=folder/'session.json'
    atomic_json(session,{'schema':'lixyswarm-native-cold-session-v1','original':str(old),'trial':str(trial),
        'physical_id':identity.node_id_hex,'provider_id':provider['node_id'],'timeout':1800,'planned_inventory':planned})
    entry,_=archive();monkeypatch.setattr(cold,'language_engine',lambda _:entry)
    def interrupted(*a):raise OSError('simulated kill after control checkpoint')
    monkeypatch.setattr(cold,'start_native',interrupted)
    with pytest.raises(OSError,match='simulated kill'):cold.recover(session)
    assert not (folder/'recovered.json').exists()
    assert ClientSettings.load(old).policy.cpu_percent==0
    assert all(store.has(i) for i in planned['weights'])
    class Process:
        ended=False
        def poll(self):return 0 if self.ended else None
        def terminate(self):self.ended=True
        def wait(self,**kw):return 0
    process=Process();monkeypatch.setattr(cold,'start_native',lambda *a:process)
    monkeypatch.setattr(cold,'fresh',lambda *a:True)
    class Controller:
        def __init__(self,*a,**kw):pass
        def repair_language_cache(self,name):
            assert ClientSettings.load(old).policy.cpu_percent==0
            assert ClientSettings.load(old).policy.max_disk_gb==json.loads(controls)['policy']['max_disk_gb']
        def cache_repair_status(self):return {'state':'complete'}
    monkeypatch.setattr(cold,'NodeController',Controller)
    result=cold.recover(session)
    assert result['recovered'] and result['original_knowledge_verified'] and process.ended
    assert result['metadata']['counters']['language-training-status.json']==1
    assert all(result['metadata']['private_files_unchanged'].values())
    assert (old/'controls.json').read_bytes()==controls
    assert (old/'personal.enc').read_bytes()==b'private canary stays local'
    assert cold.recover(session)==result
    assert json.loads((old/'language-training-status.json').read_text())['completed']==1


def test_code_cleanup_retains_download_resume_bytes_and_rejects_unknown_install_paths(tmp_path):
    entry,_=archive();trial=tmp_path/'trial';root=trial/'engine-downloads';root.mkdir(parents=True)
    part=root/(entry.sha256+'.part');part.write_bytes(b'partial code download')
    scratch=root/(entry.sha256+'.unpacked');scratch.mkdir();(scratch/'partial-code').write_bytes(b'unpacked code')
    cold.prune_owned_code(trial,entry)
    assert part.read_bytes()==b'partial code download' and not scratch.exists()
    unknown=trial/'engines/.install-12345678';unknown.mkdir(parents=True);(unknown/'private.txt').write_bytes(b'preserve unexpected file')
    with pytest.raises(ValueError,match='Unexpected engine installation'):cold.prune_owned_code(trial,entry)
    assert (unknown/'private.txt').read_bytes()==b'preserve unexpected file'
