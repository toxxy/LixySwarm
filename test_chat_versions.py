from copy import deepcopy
import json

import pytest

from src.network.chat_versions import ChatVersions
from src.release.manifest import ReleaseError
from test_language_versions import fixture


def test_light_client_accepts_evaluated_history_without_model_weights(monkeypatch,tmp_path):
    store,first,update,_=fixture(monkeypatch,tmp_path/'source')
    second,report=update(first)
    client=ChatVersions(tmp_path/'client','qwen3-0.6b')
    identifier=second['descriptor']['evaluation_id']
    assert client.install([first,second],{identifier:report})==second
    assert ChatVersions(tmp_path/'client','qwen3-0.6b').head()==second
    assert client.store.total_bytes()==0
    assert not client.store.has(second['release']['model_artifact_id'])
    assert client.install([first],{})==second
    assert client.head()==second


def test_unverified_reports_forks_and_tampered_cache_do_not_change_chat_head(monkeypatch,tmp_path):
    store,first,update,_=fixture(monkeypatch,tmp_path/'source');second,report=update(first)
    client=ChatVersions(tmp_path/'client','qwen3-0.6b');identifier=second['descriptor']['evaluation_id']
    wrong=deepcopy(report);wrong['quality_after'][0]['answer']='Madrid'
    with pytest.raises(ReleaseError,match='identity'):client.install([first,second],{identifier:wrong})
    assert client.head()==first
    client.install([first,second],{identifier:report})
    fork,fork_report=update(first)
    with pytest.raises(ReleaseError,match='fork'):
        client.install([first,fork],{fork['descriptor']['evaluation_id']:fork_report})
    data=json.loads(client.path.read_text());data['chain'][-1]['release']['signatures']=[]
    client.path.write_text(json.dumps(data))
    with pytest.raises(ReleaseError):client.head()


def test_only_evaluation_artifacts_are_fetched_and_disk_preflight_runs_first(monkeypatch,tmp_path):
    from src.network import chat_versions
    store,first,update,_=fixture(monkeypatch,tmp_path/'source');second,report=update(first)
    client=ChatVersions(tmp_path/'client','qwen3-0.6b');fetched=[]
    monkeypatch.setattr(chat_versions,'fetch_history',lambda *_a,**_kw:[first,second])
    class Transfer:
        def __init__(self,*_a,**_kw):pass
        def fetch(self,identifier,**kwargs):
            fetched.append(identifier)
            assert identifier==second['descriptor']['evaluation_id']
            source=store.object_path(identifier)
            client.store.import_file(source,kind='evaluation',media_type='application/json')
            return client.store.object_path(identifier)
    monkeypatch.setattr(chat_versions,'ArtifactService',Transfer)
    def no_space(_size):raise ValueError('no authorized space')
    with pytest.raises(ValueError,match='space'):
        client.synchronize(None,'provider',cancel_event=None,preflight=no_space)
    assert not fetched
    client.synchronize(None,'provider',cancel_event=None,preflight=lambda _size:None)
    assert fetched==[second['descriptor']['evaluation_id']] and client.head()==second
