"""Signed synthetic replication fixtures; no extra physical nodes claimed."""
from dataclasses import replace
from types import SimpleNamespace
import threading

import pytest

from src.contribution import ContributionPolicy
from src.network.client_state import ClientSettings
from src.network.language_compute import LanguageSettings
from src.network import language_follow as follow
from src.release.language_versions import LanguageVersionRegistry
from test_language_replication import setup as replication_setup


def setup(monkeypatch,tmp_path,*,select=False):
    source,target,first,second,evaluation,calls,update=replication_setup(monkeypatch,tmp_path)
    directory=tmp_path/'receiver';provider='a'*64
    node=SimpleNamespace(identity=SimpleNamespace(node_id_hex='b'*64),connected_node_ids=[provider],
        peers=lambda:[{'node_id':provider,'host':'8.8.8.8','transport':'relay','encrypted':True,'relay_host':'1.1.1.1'}])
    knowledge=SimpleNamespace(node=node,work=None,allowed=True,budget=SimpleNamespace(wait=lambda *_a,**_kw:None))
    ClientSettings(policy=ContributionPolicy.for_mode('maximum')).save(directory)
    settings=follow.LanguageFollowSettings(enabled=True,base_name='qwen3-0.6b',providers=[provider],select_updates=select)
    settings.save(directory)
    follower=follow.LanguageVersionFollower(knowledge,SimpleNamespace(root=directory,store=target),directory)
    follower.settings=settings
    monkeypatch.setattr(follow,'fetch_history',lambda *_a,**_kw:[first,second])
    return follower,source,target,first,second,calls,update


def test_following_verifies_new_head_and_avoids_rehashing_unchanged_objects(monkeypatch,tmp_path):
    follower,source,target,first,second,calls,update=setup(monkeypatch,tmp_path)
    original=follow.replicate_version;verified=[]
    def verify(*args,**kwargs):verified.append(True);return original(*args,**kwargs)
    monkeypatch.setattr(follow,'replicate_version',verify)
    follower._run(threading.Event())
    assert follower.snapshot()['state']=='current' and follower.snapshot()['sequence']==1
    assert (follower.directory/'language-follow/versions'/f"{second['release']['release_id']}.json").is_file()
    assert LanguageVersionRegistry(follower.families.root,'qwen3-0.6b').head()==second
    assert not LanguageSettings.load(follower.directory).enabled
    count=len(calls);follower._run(threading.Event())
    assert verified==[True] and len(calls)==count
    # A same-size write invalidates the stat fingerprint and forces hash repair.
    path=target.object_path(second['release']['model_artifact_id'])
    path.write_bytes(b'x'*path.stat().st_size)
    follower._run(threading.Event())
    assert len(verified)==2 and path.read_bytes()==source.object_path(second['release']['model_artifact_id']).read_bytes()


@pytest.mark.parametrize('failure',['quota','revoked','fork'])
def test_storage_limits_revocation_and_signed_forks_do_not_replace_the_head(monkeypatch,tmp_path,failure):
    follower,source,target,first,second,calls,update=setup(monkeypatch,tmp_path)
    if failure=='quota':
        controls=ClientSettings.load(follower.directory);controls.policy.max_disk_gb=.001;controls.save(follower.directory)
    elif failure=='revoked':
        original=follow.replicate_version
        def revoke(*args,**kwargs):
            controls=ClientSettings.load(follower.directory);controls.paused=True;controls.save(follower.directory)
            return original(*args,**kwargs)
        monkeypatch.setattr(follow,'replicate_version',revoke)
    else:
        follower._run(threading.Event());calls.clear();first=second
        initial=follow.LanguageVersionRegistry(follower.families.root,'qwen3-0.6b').history(second['release']['release_id'])[0]
        fork,_report=update(initial)
        monkeypatch.setattr(follow,'fetch_history',lambda *_a,**_kw:[initial,fork])
    follower._run(threading.Event())
    assert follower.snapshot()['state'] in {'waiting','paused'} and follower.snapshot()['error']
    assert LanguageVersionRegistry(follower.families.root,'qwen3-0.6b').head()==first
    assert calls==[] and not LanguageSettings.load(follower.directory).enabled


@pytest.mark.parametrize('operator_change',[False,True])
def test_optional_selection_preserves_local_expert_and_compute_consent(monkeypatch,tmp_path,operator_change):
    follower,source,target,first,second,calls,update=setup(monkeypatch,tmp_path,select=True)
    engine=LanguageSettings(enabled=True,command=[str(tmp_path/'local-engine')],base_name='qwen3-0.6b',
        base_directory=str(tmp_path/'base'),adapter_file=str(tmp_path/'base/adapter.safetensors'),adapter_id=first['release']['model_artifact_id'])
    engine.save(follower.directory);controls=ClientSettings.load(follower.directory)
    selected=[]
    def activate(directory,name,release_id,*,expected_adapter_id):
        selected.append((name,release_id,expected_adapter_id))
        replace(LanguageSettings.load(directory),adapter_id=second['release']['model_artifact_id']).save(directory)
    monkeypatch.setattr(follow,'activate_version',activate)
    if operator_change:
        original=follow.replicate_version
        def change(*args,**kwargs):
            result=original(*args,**kwargs)
            replace(engine,adapter_id='e'*64).save(follower.directory)
            return result
        monkeypatch.setattr(follow,'replicate_version',change)
    follower._run(threading.Event())
    assert ClientSettings.load(follower.directory)==controls
    if operator_change:
        assert selected==[] and LanguageSettings.load(follower.directory).adapter_id=='e'*64
        assert follower.snapshot()['selection']=='local_expert_changed'
    else:
        assert selected==[('qwen3-0.6b',second['release']['release_id'],engine.adapter_id)]
        assert LanguageSettings.load(follower.directory).enabled


def test_following_defaults_off_and_cannot_enable_an_unconfigured_expert(monkeypatch,tmp_path):
    assert not follow.LanguageFollowSettings().enabled
    follower,source,target,first,second,calls,update=setup(monkeypatch,tmp_path,select=True)
    follower._run(threading.Event())
    assert follower.snapshot()['selection']=='expert_not_enabled'
    assert not LanguageSettings.load(follower.directory).enabled
