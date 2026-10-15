import hashlib
import json
import os
from pathlib import Path

import pytest

from src.network.engine_catalog import EngineDownload
from src.network.engine_package import install_package,verify_package
from src.network.engine_cache import candidate_cache_bytes,inspect_cache,reclaim_inactive
from src.network.client_state import NodeStateLock
from src.network.reference_compute import EngineSettings
from test_engine_package import package


def cached_cpu(tmp_path,*,state=None):
    tmp_path.mkdir(parents=True,exist_ok=True);root,value,manifest_id=package(tmp_path)
    state=tmp_path/'state' if state is None else state;state.mkdir(parents=True,exist_ok=True)
    command=install_package(root,manifest_id,state)
    # A tiny pinned transport fixture; this test never decodes or executes it.
    raw=b'verified public code archive fixture'*40;identifier=hashlib.sha256(raw).hexdigest()
    entry=EngineDownload('https://example.invalid/code.zip',identifier,len(raw),manifest_id,
        sum(p.stat().st_size for p in root.iterdir()))
    archive=state/'engine-downloads'/(identifier+'.zip');archive.parent.mkdir();archive.write_bytes(raw)
    return state,entry,archive,Path(command)


def test_redundant_archive_retires_with_selected_engine_models_and_settings_intact(tmp_path):
    state,entry,archive,command=cached_cpu(tmp_path)
    (state/'identity.key').write_bytes(b'private fixture identity')
    model=state/'public/model';model.parent.mkdir();model.write_bytes(b'learned fixture must remain')
    partial=archive.with_suffix('.part');partial.write_bytes(b'resumable fixture')
    unknown=archive.parent/'unknown.zip';unknown.write_bytes(b'unknown operator data')
    before=(state/'compute-engine.json').read_bytes()
    assert candidate_cache_bytes(state,[entry])==entry.bytes
    assert len(inspect_cache(state,[entry]))==1 and archive.exists()
    result=reclaim_inactive(state,[entry])
    assert result['removed_cache_bytes']==entry.bytes and not result['installed_engines_removed']
    assert not archive.exists() and command.read_bytes()==b'trusted-test-package'
    assert model.read_bytes()==b'learned fixture must remain' and partial.read_bytes()==b'resumable fixture'
    assert unknown.read_bytes()==b'unknown operator data' and (state/'compute-engine.json').read_bytes()==before
    assert (state/'identity.key').read_bytes()==b'private fixture identity'
    assert reclaim_inactive(state,[entry])['removed_cache_bytes']==0


@pytest.mark.parametrize('corruption',['archive','installed','manifest'])
def test_unverified_code_or_archive_cannot_be_pruned(tmp_path,corruption):
    state,entry,archive,command=cached_cpu(tmp_path)
    victim=archive if corruption=='archive' else command if corruption=='installed' else command.parent/'engine-package.json'
    raw=victim.read_bytes();victim.write_bytes(bytes([raw[0]^1])+raw[1:])
    assert reclaim_inactive(state,[entry])['removed_cache_bytes']==0
    assert archive.exists()


@pytest.mark.parametrize('lock_name',['node','preparation'])
def test_maintenance_respects_live_node_and_preparation_ownership(tmp_path,lock_name):
    state,entry,archive,_=cached_cpu(tmp_path)
    with NodeStateLock(state if lock_name=='node' else state/'preparation-lock'):
        with pytest.raises(RuntimeError,match='already uses'):reclaim_inactive(state,[entry])
    assert archive.exists()


def test_selected_archive_reference_and_unfinished_downloads_are_preserved(tmp_path):
    state,entry,archive,_=cached_cpu(tmp_path)
    EngineSettings(enabled=False,command=[str(archive)]).save(state)
    assert candidate_cache_bytes(state,[entry])==0
    assert reclaim_inactive(state,[entry])['removed_cache_bytes']==0 and archive.exists()
    archive.rename(archive.with_suffix('.part'))
    assert reclaim_inactive(state,[entry])['removed_cache_bytes']==0 and archive.with_suffix('.part').exists()


def test_alias_state_and_cache_links_cannot_delete_foreign_data(tmp_path):
    state,entry,archive,_=cached_cpu(tmp_path)
    alias=tmp_path/'alias'
    try:alias.symlink_to(state,target_is_directory=True)
    except OSError:pytest.skip('Symlink fixture unavailable')
    with pytest.raises(ValueError,match='canonical'):reclaim_inactive(alias,[entry])
    foreign=tmp_path/'foreign';archive.rename(foreign);archive.symlink_to(foreign)
    assert reclaim_inactive(state,[entry])['removed_cache_bytes']==0 and foreign.is_file() and archive.is_symlink()


def test_cancellation_interrupts_verification_before_deletion(tmp_path):
    state,entry,archive,_=cached_cpu(tmp_path);calls=0
    def check():
        nonlocal calls
        calls+=1
        if calls==4:raise RuntimeError('permission withdrawn')
    with pytest.raises(RuntimeError,match='withdrawn'):reclaim_inactive(state,[entry],check=check)
    assert archive.exists()


@pytest.mark.parametrize('changed',['archive','installed'])
def test_mutation_after_preview_cannot_remove_the_verified_backup(tmp_path,monkeypatch,changed):
    from src.network import engine_cache
    state,entry,archive,command=cached_cpu(tmp_path);original=engine_cache.inspect_cache
    def inspect(*args,**kwargs):
        result=original(*args,**kwargs);path=archive if changed=='archive' else command
        stat=path.stat();data=path.read_bytes();path.write_bytes(data[:-1]+bytes([data[-1]^1]))
        os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns))
        return result
    monkeypatch.setattr(engine_cache,'inspect_cache',inspect)
    with pytest.raises(ValueError):reclaim_inactive(state,[entry])
    assert archive.exists()


@pytest.mark.parametrize('corrupt',[False,True])
def test_gpu_preparation_reclaims_only_verified_cache_before_downloading(monkeypatch,tmp_path,corrupt):
    from src.network import language_preparation as preparation
    from src.network.client_state import ClientSettings
    from src.network.public_knowledge import managed_public_bytes
    from test_language_preparation import context,finish
    holder={}
    def first_network_read():assert not holder['archive'].exists()
    service,target,_,opens=context(monkeypatch,tmp_path,hook=first_network_read)
    state,old,archive,command=cached_cpu(tmp_path/'old-fixture',state=tmp_path);holder['archive']=archive
    monkeypatch.setattr(preparation,'local_entries',lambda:(old,))
    settings=ClientSettings.load(state)
    required=preparation.remaining_budget(state,service.families.store,target)
    settings.policy.max_disk_gb=(managed_public_bytes(state)+required-old.bytes/2)/1024**3;settings.save(state)
    before=(state/'controls.json').read_bytes()
    if corrupt:
        raw=archive.read_bytes();archive.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
    service.start();finish(service)
    assert (state/'controls.json').read_bytes()==before and command.is_file()
    if corrupt:
        assert service.snapshot()['state']=='waiting' and not opens and archive.exists()
    else:
        assert service.snapshot()['state']=='complete' and opens and not archive.exists()
