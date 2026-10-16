"""Native own work preserves owner authorization, leases and selected state."""
import json
from pathlib import Path
import time
from types import SimpleNamespace
import urllib.error

import pytest

from src.contribution import ContributionPolicy,ResourceRequirements
from src.learning.swarm_distributed import WORKERS,policy_id,state_id
from src.learning.public_math import ARTIFACTS
from src.network.client_state import ClientSettings
from src.network.own_swarm import OwnSwarmService,OwnSwarmSettings
from test_chat_service import broker_fixture,call
from src.network.chat_service import LocalChatAPI


def fixture(tmp_path):
    handlers={}
    node=SimpleNamespace(identity=SimpleNamespace(node_id_hex=WORKERS[0]))
    knowledge=SimpleNamespace(allowed=True,node=node,budget=SimpleNamespace(wait=lambda *_a,**_kw:None),
        work=SimpleNamespace(node=node,register_handler=lambda name,kind,handler,**kw:handlers.update({name:handler})))
    service=OwnSwarmService(knowledge,tmp_path,{'cpu_cores':4})
    model=tmp_path/'model';model.write_bytes(b'own model fixture')
    opt=tmp_path/'optimizer';opt.write_bytes(b'own optimizer fixture')
    mid=service.store.import_file(model,kind='model').artifact_id
    oid=service.store.import_file(opt,kind='other').artifact_id
    command=tmp_path/'installed-engine';command.write_text('fixture')
    OwnSwarmSettings(enabled=True,command=[str(command)],model_id=mid,optimizer_id=oid,
        step=1024,requesters=list(WORKERS)).save(tmp_path)
    for path in [service.training_path(),service.evaluation_path()]:path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture')
    settings=ClientSettings(policy=ContributionPolicy.for_mode('maximum'))
    service.update(settings);assert service.ready
    payload={'training_policy_id':policy_id(),'state_id':state_id(mid,oid,1024),'model_id':mid,
        'dataset_id':ARTIFACTS['train'][0],'start':0,'count':128}
    execution=SimpleNamespace(origin_node_id=WORKERS[0],payload=payload,
        cancel_event=__import__('threading').Event(),raise_if_cancelled=lambda:None,
        deadline=time.time()+60,resource_requirements=lambda:ResourceRequirements(kind='training',ram_gb=4,disk_gb=.02))
    return service,payload,execution,handlers


def test_native_typed_compute_commits_only_its_own_bounded_output_and_persists_counter(tmp_path,monkeypatch):
    from src.network.artifact_store import digest_file
    service,payload,execution,_=fixture(tmp_path)
    def compute(command,request,_execution,**kw):
        assert kw['allowed']() and request['action']=='own-gradient'
        path=Path(request['output_directory'])/'gradient.safetensors';path.parent.mkdir();path.write_bytes(b'public gradient fixture')
        identifier,size=digest_file(path)
        return {**payload,'tokens':128,'loss':2.,'gradient_id':identifier,'gradient_bytes':size,'gradient_file':str(path)}
    monkeypatch.setattr(service.supervisor,'run',compute)
    output=service.gradient(payload,execution)
    assert service.store.has(output['gradient_id']) and service.completed==1 and service.active==0
    assert not list((service.root/'jobs').iterdir())
    again=OwnSwarmService(service.knowledge,tmp_path,{'cpu_cores':4})
    assert again.completed==1 and not again.ready
    assert service.permit_cached(execution,output)
    service.engine.step+=1
    assert not service.permit_cached(execution,output)


@pytest.mark.parametrize('change',['unknown_requester','private_path','stale_state','low_ram','gpu','paused'])
def test_invalid_or_unconsented_own_work_never_launches_a_tensor_process(tmp_path,monkeypatch,change):
    service,payload,execution,_=fixture(tmp_path)
    monkeypatch.setattr(service.supervisor,'run',lambda *_a,**_kw:pytest.fail('unapproved child launched'))
    if change=='unknown_requester':execution.origin_node_id='f'*64
    if change=='private_path':payload['personal_path']='/private/memory'
    if change=='stale_state':payload['state_id']='f'*64
    if change=='low_ram':execution.resource_requirements=lambda:ResourceRequirements(kind='training',ram_gb=1,disk_gb=.02)
    if change=='gpu':execution.resource_requirements=lambda:ResourceRequirements(kind='training',gpu_required=True,ram_gb=4,disk_gb=.02)
    if change=='paused':service.knowledge.allowed=False
    with pytest.raises(ValueError):service.gradient(payload,execution)
    assert service.completed==0


def test_own_ipc_uses_existing_loopback_auth_and_closes_owned_work(monkeypatch,tmp_path):
    broker,_=broker_fixture(monkeypatch);started=[];closed=[]
    work=SimpleNamespace(start=lambda v:started.append(v) or {'state':'running'},
        status=lambda _: {'state':'complete'},cancel=lambda _:True,close=lambda:closed.append(True))
    api=LocalChatAPI(broker,tmp_path,own_work=work);api.start()
    try:
        for options in [{'token':'wrong'},{'origin':'https://untrusted.invalid'}]:
            with pytest.raises(urllib.error.HTTPError) as error:call(api,'/own-work/start',{},**options)
            assert error.value.code==403
        assert not started
        assert call(api,'/own-work/start',{'typed':'fixture'})=={'state':'running'}
        assert call(api,'/own-work/'+'a'*32)=={'state':'complete'}
        assert call(api,'/own-work/cancel',{'request_id':'a'*32})=={'cancelled':True}
    finally:api.close()
    assert closed==[True]


def test_own_training_state_changed_during_child_rejects_the_result_before_commit(tmp_path,monkeypatch):
    from src.network.compute_process import ComputeProcessError
    from src.network.artifact_store import digest_file
    service,payload,execution,_=fixture(tmp_path)
    def compute(_command,request,_execution,**_kw):
        path=Path(request['output_directory'])/'gradient';path.parent.mkdir();path.write_bytes(b'old-state gradient')
        identifier,size=digest_file(path);service.engine.step+=1
        return {**payload,'tokens':128,'loss':2.,'gradient_id':identifier,'gradient_bytes':size,'gradient_file':str(path)}
    monkeypatch.setattr(service.supervisor,'run',compute)
    with pytest.raises(ComputeProcessError,match='state or permissions changed'):
        service.gradient(payload,execution)
    assert service.completed==0 and service.active==0
