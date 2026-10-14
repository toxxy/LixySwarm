import hashlib
import json
import os
import threading
import urllib.request
import urllib.error
from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.network import chat_service
from src.network.chat_service import ChatBroker,LocalChatAPI,MODEL_ID
from src.network.lsp import LSPIdentity
from src.network.work_protocol import WorkResult,ResultReceipt
from src.network.client_state import ClientSettings
from desktop.chat_history import ChatHistory,bounded_context


def broker_fixture(monkeypatch):
    requester=LSPIdentity.generate();worker=LSPIdentity.generate();sent=[]
    monkeypatch.setattr(chat_service,"PROVIDERS",{worker.node_id_hex})
    node=SimpleNamespace(identity=requester,peers=lambda:[{"node_id":worker.node_id_hex,
        "capabilities":{"language_inference_v1":True},"resources":{"models":[MODEL_ID],"has_gpu":True,"ram_gb":5}}])
    def submit(operation,payload,*_,**__):
        sent.append(payload)
        result=WorkResult(job_id=hashlib.sha256(b"chat-test").hexdigest(),status="ok",
                          output={"model_id":MODEL_ID,"answer":"public answer","output_tokens":2})
        return replace(result,receipt=ResultReceipt.create(result,worker,requester.node_id_hex).to_dict())
    work=SimpleNamespace(submit=submit)
    language=SimpleNamespace(ready=False,engine=SimpleNamespace(adapter_id=MODEL_ID))
    broker=ChatBroker(node,work,language);broker.update(ClientSettings())
    return broker,sent


def call(api,path,value=None,*,token=None,origin=None):
    headers={"Authorization":"Bearer "+(api.token if token is None else token),"Content-Type":"application/json"}
    if origin:headers["Origin"]=origin
    request=urllib.request.Request(f"http://127.0.0.1:{api.server.server_port}{path}",data=json.dumps(value).encode() if value is not None else None,headers=headers)
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request,timeout=5) as response:return json.load(response)


def test_current_question_only_is_sent_by_default_and_ipc_writes_no_plaintext(monkeypatch,tmp_path):
    broker,sent=broker_fixture(monkeypatch);api=LocalChatAPI(broker,tmp_path);api.start()
    private="private-history-canary-9493"
    value={"request_id":"a"*32,"remote_context":False,"messages":[{"role":"user","content":private},{"role":"assistant","content":"old answer"},{"role":"user","content":"public question"}]}
    try:
        result=call(api,"/ask",value)
        assert result["answer"]=="public answer" and result["context_sent"] is False
        assert sent[-1]["messages"]==[{"role":"user","content":"public question"}]
        value["request_id"]="b"*32;value["remote_context"]=True
        assert call(api,"/ask",value)["context_sent"] is True
        assert sent[-1]["messages"]==value["messages"]
        assert private not in json.dumps(call(api,"/status"))
        assert all(private.encode() not in path.read_bytes() for path in tmp_path.iterdir() if path.is_file())
        if os.name!="nt":assert (tmp_path/"local-api.json").stat().st_mode&0o777==0o600
    finally:api.close()
    assert not (tmp_path/"local-api.json").exists()


def test_expert_preparation_ipc_requires_auth_and_accepts_no_programs_or_paths(monkeypatch,tmp_path):
    broker,_=broker_fixture(monkeypatch);started=[]
    preparation=SimpleNamespace(snapshot=lambda:{'state':'idle'},start=lambda:started.append(True) or {'state':'starting'},stop=lambda:{'state':'stopped'})
    api=LocalChatAPI(broker,tmp_path,preparation);api.start()
    try:
        assert call(api,'/preparation/status')=={'state':'idle'}
        for kwargs in [{'token':'wrong'},{'origin':'https://untrusted.example'}]:
            with pytest.raises(urllib.error.HTTPError) as error:call(api,'/preparation/start',{},**kwargs)
            assert error.value.code==403
        with pytest.raises(urllib.error.HTTPError) as error:
            call(api,'/preparation/start',{'command':['peer-code'],'base_directory':'private'})
        assert error.value.code==400 and not started
        assert call(api,'/preparation/start',{})=={'state':'starting'}
        assert call(api,'/preparation/stop',{})=={'state':'stopped'}
    finally:api.close()


def test_browser_origin_bad_auth_and_unknown_chat_fields_are_rejected(monkeypatch,tmp_path):
    broker,sent=broker_fixture(monkeypatch);api=LocalChatAPI(broker,tmp_path);api.start()
    try:
        for kwargs in [{"token":"bad"},{"origin":"https://untrusted.example"}]:
            with pytest.raises(urllib.error.HTTPError) as error:call(api,"/status",**kwargs)
            assert error.value.code==403
        value={"request_id":"a"*32,"remote_context":False,"messages":[{"role":"user","content":"public question"}],"model_path":"peer-program"}
        with pytest.raises(urllib.error.HTTPError) as error:call(api,"/ask",value)
        assert error.value.code==400 and not sent
    finally:api.close()


def test_invalid_receipt_and_worker_error_do_not_echo_private_input(monkeypatch,tmp_path):
    broker,_=broker_fixture(monkeypatch)
    broker.work.submit=lambda *_a,**_k:WorkResult(job_id="a"*64,status="ok",output={"model_id":MODEL_ID,"answer":"unsigned"})
    api=LocalChatAPI(broker,tmp_path);api.start()
    value={"request_id":"a"*32,"remote_context":False,"messages":[{"role":"user","content":"private-canary-error"}]}
    try:
        with pytest.raises(urllib.error.HTTPError) as error:call(api,"/ask",value)
        assert b"private-canary-error" not in error.value.read()
    finally:api.close()


def test_cancellation_before_dispatch_and_disconnection_send_no_question(monkeypatch):
    broker,sent=broker_fixture(monkeypatch)
    value={"request_id":"a"*32,"remote_context":False,"messages":[{"role":"user","content":"private question"}]}
    broker.cancel(value["request_id"])
    with pytest.raises(RuntimeError,match="cancelada"):broker.ask(value)
    broker.update(ClientSettings(connected=False))
    value["request_id"]="b"*32
    with pytest.raises(RuntimeError,match="Conecta"):broker.ask(value)
    assert not sent


def test_active_chat_can_be_cancelled_through_authenticated_ipc(monkeypatch,tmp_path):
    broker,_=broker_fixture(monkeypatch);entered=threading.Event();errors=[]
    def blocked(*_args,**kwargs):
        entered.set()
        assert kwargs["cancel_event"].wait(3)
        raise chat_service.ChatUnavailable("Consulta cancelada")
    broker.work.submit=blocked
    api=LocalChatAPI(broker,tmp_path);api.start()
    value={"request_id":"c"*32,"remote_context":False,"messages":[{"role":"user","content":"bounded question"}]}
    def ask():
        try:call(api,"/ask",value)
        except urllib.error.HTTPError as error:errors.append(error.code)
    thread=threading.Thread(target=ask)
    try:
        thread.start();assert entered.wait(3)
        assert call(api,"/cancel",{"request_id":value["request_id"]})["cancelled"]
        thread.join(timeout=3)
        assert not thread.is_alive() and errors==[409]
        assert call(api,"/status")["active"]==0
    finally:
        broker.close();thread.join(timeout=3);api.close()


def test_new_requester_prioritizes_approved_expert_without_blocking_controls(monkeypatch):
    broker,_=broker_fixture(monkeypatch);broker.node.peers=lambda:[]
    connected=[];attempted=threading.Event()
    def connect(provider,timeout):
        connected.append(provider);attempted.set();return True
    broker.node.connect_node=connect
    try:
        broker.update(ClientSettings());assert attempted.wait(2)
        broker.discovery_thread.join(timeout=2)
        broker.update(ClientSettings())
        assert connected==sorted(chat_service.PROVIDERS)
        assert broker.snapshot()["available"] is False
    finally:broker.close()


class KeyStore:
    def __init__(self):self.values={}
    def get_password(self,service,account):return self.values.get((service,account))
    def set_password(self,service,account,value):self.values[service,account]=value


def test_chat_history_survives_restart_encrypted_and_preserves_other_personal_banks(tmp_path):
    backend=KeyStore();history=ChatHistory(tmp_path,backend=backend)
    messages=[{"role":"user","content":"personal-context-canary-2305"},{"role":"assistant","content":"answer"}]
    history.store.save({"personal_bank":{"keep":True}})
    history.save(messages,True)
    assert b"personal-context-canary-2305" not in (tmp_path/"personal.enc").read_bytes()
    restored=ChatHistory(tmp_path,backend=backend)
    assert restored.messages==messages and restored.share_context
    assert restored.store.load()["personal_bank"]=={"keep":True}
    assert bounded_context(messages,"next question")[-1]["content"]=="next question"
