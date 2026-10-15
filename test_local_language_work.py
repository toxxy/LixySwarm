"""Native owner work preserves physical identity, leases, consent and loopback auth."""
import json
import threading
import time
from types import SimpleNamespace
import urllib.error
import urllib.request

import pytest

from src.network import local_language_work as native
from src.network.chat_service import LocalChatAPI
from src.network.client_state import ClientSettings
from src.contribution import ContributionPolicy
from test_language_learning import setup


def fixture(monkeypatch,tmp_path):
    coordinator,controls,head,calls=setup(monkeypatch,tmp_path)
    training=coordinator.training;training.settings=ClientSettings(policy=ContributionPolicy.for_mode('maximum'));training._head=lambda:head
    training.engine.batching_enabled=False
    coordinator.knowledge.budget=SimpleNamespace(wait=lambda *_args,**_kwargs:None)
    monkeypatch.setattr(native,'WORKERS',set(controls.workers))
    manager=native.LocalLanguageWork(coordinator.knowledge,training,coordinator)
    value={'request_id':'1'*32,'worker_id':controls.workers[0],'operation':'language.gradient.v2',
        'payload':{'base_name':'qwen3-0.6b','release_id':head['release']['release_id'],
            'adapter_id':head['release']['model_artifact_id'],'dataset_id':coordinator._registry().head()['descriptor'].get('dataset_id',''),
            'start':0,'count':4}}
    from src.learning.public_conversation import public_corpus,corpus_digest
    value['payload']['dataset_id']=corpus_digest(public_corpus())
    return manager,value,coordinator,calls


def wait_done(manager,identifier):
    until=time.monotonic()+3
    while time.monotonic()<until:
        state=manager.status(identifier)
        if state['state']!='running':return state
        time.sleep(.01)
    raise AssertionError('Native synthetic work did not finish')


def test_existing_physical_node_executes_without_network_or_shadow_requester(monkeypatch,tmp_path):
    manager,value,coordinator,calls=fixture(monkeypatch,tmp_path)
    try:
        manager.start(value);state=wait_done(manager,value['request_id'])
        assert state['state']=='complete' and state['route']['transport']=='local'
        assert state['encoded_data_bytes']==0 and state['artifact_initially_local']
        assert state['result']['receipt']['requester_node_id']==value['worker_id']
        assert calls==['language.gradient.v2'] and coordinator.ledger is None
    finally:manager.close()


@pytest.mark.parametrize('problem',['worker','operation','private_path','dataset','parent','batch_permission','pending_coordinator'])
def test_native_owner_bridge_rejects_unapproved_work_before_dispatch(monkeypatch,tmp_path,problem):
    manager,value,coordinator,calls=fixture(monkeypatch,tmp_path)
    try:
        if problem=='worker':value['worker_id']='f'*64
        elif problem=='operation':value['operation']='shell.execute'
        elif problem=='private_path':value['payload']['personal_path']='/private/memory'
        elif problem=='dataset':value['payload']['dataset_id']='f'*64
        elif problem=='parent':value['payload']['release_id']='f'*64
        elif problem=='batch_permission':value['operation']='language.gradient.v3'
        else:coordinator._initialize();coordinator._prepare_round()
        with pytest.raises((ValueError,RuntimeError)):manager.start(value)
        assert not calls and not manager.jobs
    finally:manager.close()


def test_owner_disappearance_cancels_a_compute_lease(monkeypatch,tmp_path):
    manager,value,coordinator,calls=fixture(monkeypatch,tmp_path);entered=threading.Event();cancelled=threading.Event()
    def block(*_args,**kwargs):
        entered.set()
        assert kwargs['cancel_event'].wait(3)
        cancelled.set();raise RuntimeError('synthetic cancellation')
    coordinator.knowledge.work.execute_local=block
    try:
        manager.start(value);assert entered.wait(1)
        with manager.lock:manager.jobs[value['request_id']]['lease']=time.monotonic()-1
        assert cancelled.wait(2)
        assert wait_done(manager,value['request_id'])['state']=='cancelled'
    finally:manager.close()


def test_native_work_api_requires_owner_token_host_and_no_web_origin(monkeypatch,tmp_path):
    manager,value,coordinator,calls=fixture(monkeypatch,tmp_path)
    broker=SimpleNamespace(snapshot=lambda:{},close=lambda:None)
    api=LocalChatAPI(broker,tmp_path/'api',language_work=manager);api.start()
    url=f'http://127.0.0.1:{api.server.server_port}'
    def send(path,payload=None,**headers):
        data=None if payload is None else json.dumps(payload).encode()
        req=urllib.request.Request(url+path,data=data,headers={'Content-Type':'application/json',**headers})
        return urllib.request.urlopen(req,timeout=2)
    try:
        for headers in [{},{'Authorization':'Bearer '+api.token,'Origin':'https://untrusted.invalid'},
                        {'Authorization':'Bearer '+api.token,'Host':'untrusted.invalid'}]:
            with pytest.raises(urllib.error.HTTPError) as error:send('/language-work/start',value,**headers)
            assert error.value.code==403
        assert not calls
        with send('/language-work/start',value,Authorization='Bearer '+api.token) as response:assert response.status==200
        state=wait_done(manager,value['request_id']);assert state['state']=='complete'
        with send('/language-work/'+value['request_id'],Authorization='Bearer '+api.token) as response:
            result=json.loads(response.read());assert api.token not in json.dumps(result)
    finally:api.close()
