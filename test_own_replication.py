"""Native adoption enforces local opt-in, public transport, replay and atomic selection."""
from dataclasses import asdict,replace
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from src.contribution import ContributionPolicy,ResourceRequirements
from src.learning.public_math import canonical
from src.learning.swarm_distributed import policy_id,state_id
from src.network import own_replication,own_learning_settings,own_swarm,own_state
from src.learning import swarm_state_audit
from src.network.artifact_store import ArtifactStore
from src.network.client_state import ClientSettings
from src.network.own_learning_settings import OwnLearningSettings
from src.network.work_protocol import WorkResult,ResultReceipt
from test_swarm_distributed import fixture_records


def fixture(tmp_path,monkeypatch):
    records,workers,owner,mid,parent=fixture_records();ids=[v.node_id_hex for v in workers]
    monkeypatch.setattr(own_swarm,'WORKERS',ids);monkeypatch.setattr(own_learning_settings,'WORKERS',ids)
    verify=own_state.verify_transition
    for module in [own_state,own_replication,swarm_state_audit]:
        monkeypatch.setattr(module,'verify_transition',lambda r,b:verify(r,b,workers=ids))
    source=ArtifactStore(tmp_path/'remote-provider')
    def content(label,kind='other'):
        p=tmp_path/label;p.write_bytes(label.encode());return source.import_file(p,kind=kind)
    for index,worker in enumerate(workers):
        gradient=content('gradient-'+str(index),'gradient');result=WorkResult.from_dict(records[index]['result'])
        result=replace(result,output={**result.output,'gradient_id':gradient.artifact_id,'gradient_bytes':gradient.size})
        result=replace(result,receipt=ResultReceipt.create(result,worker,owner.node_id_hex).to_dict())
        records[index]['result']=result.to_dict()
    model=content('candidate-model','model');optimizer=content('candidate-optimizer')
    metrics={'nll':2.,'byte_accuracy':.4,'predicted_bytes':57123,'documents':128}
    shared={'accepted':True,'training_policy_id':policy_id(),'state_id':parent,'model_id':mid,'optimizer_id':'b'*64,
        'before':metrics,'after':{**metrics,'nll':1.99,'byte_accuracy':.41},'previous_optimizer_step':1024,
        'candidate_optimizer_step':1025,'prediction_bytes':32768}
    following=state_id(model.artifact_id,optimizer.artifact_id,1025)
    p=tmp_path/'evaluation';p.write_bytes(canonical({**shared,'candidate_state_id':following,'contributions':records}))
    evaluation=source.import_file(p,kind='evaluation')
    output={**shared,'candidate_model_id':model.artifact_id,'candidate_model_bytes':model.size,
        'candidate_optimizer_id':optimizer.artifact_id,'candidate_optimizer_bytes':optimizer.size,
        'evaluation_id':evaluation.artifact_id,'evaluation_bytes':evaluation.size}
    result=WorkResult(job_id='f'*64,status='ok',output=output)
    result=replace(result,receipt=ResultReceipt.create(result,owner,owner.node_id_hex).to_dict())
    directory=tmp_path/'follower';directory.mkdir();command=tmp_path/'installed-engine';command.write_text('fixture')
    own_swarm.OwnSwarmSettings(enabled=True,command=[str(command)],model_id=mid,optimizer_id='b'*64,
        step=1024,requesters=ids).save(directory)
    settings=ClientSettings(policy=ContributionPolicy.for_mode('maximum'));settings.save(directory)
    OwnLearningSettings(follow_enabled=True,owner_id=owner.node_id_hex).save(directory)
    node=SimpleNamespace(identity=SimpleNamespace(node_id_hex=ids[1]),connected_node_ids={ids[0]},
        peers=lambda:[{'node_id':ids[0],'host':'31.97.9.54','relay_host':'31.97.9.54','transport':'relay','encrypted':True}])
    knowledge=SimpleNamespace(allowed=True,node=node,work=SimpleNamespace(register_handler=lambda *_a,**_kw:None))
    store=ArtifactStore(directory/'public/own/artifacts');fetches=[]
    def fetch(identifier,**options):
        fetches.append(identifier);return store.import_file(source.object_path(identifier),kind='other')
    service=SimpleNamespace(directory=directory,root=directory/'public/own',knowledge=knowledge,store=store,
        ready=True,settings=settings,engine=own_swarm.OwnSwarmSettings.load(directory),
        resources={'cpu_cores':4},active=0,lock=threading.RLock(),changed=threading.Event(),
        transfer=SimpleNamespace(fetch=fetch))
    service.snapshot=lambda:{'state_id':state_id(service.engine.model_id,service.engine.optimizer_id,service.engine.step)}
    service.update=lambda _s:setattr(service,'engine',own_swarm.OwnSwarmSettings.load(directory))
    audit={'success':True,'state_id':following,'model_id':model.artifact_id,'optimizer_id':optimizer.artifact_id,
        'training_policy_id':policy_id(),'gradient_replay_verified':True,'saved_state_unchanged':True,'optimizer_step':1025}
    def replay(_command,request,_execution,**options):
        assert request['action']=='own-audit' and options['allowed']() and request['device']=='cpu'
        assert 'identity.key' not in json.dumps(request)
        return audit
    service.supervisor=SimpleNamespace(run=replay)
    execution=SimpleNamespace(origin_node_id=owner.node_id_hex,cancel_event=threading.Event(),
        deadline=time.time()+1200,raise_if_cancelled=lambda:None,
        resource_requirements=lambda:ResourceRequirements(kind='training',ram_gb=4,disk_gb=.12))
    return own_replication.OwnStateReplication(service),{'candidate_result':result.to_dict()},execution,fetches,audit


def test_native_replica_verifies_all_public_objects_before_replay_and_keeps_consent(tmp_path,monkeypatch):
    replica,payload,execution,fetches,audit=fixture(tmp_path,monkeypatch);service=replica.service
    before=(service.directory/'controls.json').read_bytes();old=asdict(service.engine)
    result=replica.adopt(payload,execution)
    assert result['success'] and result['state_id']==audit['state_id'] and len(fetches)==5
    assert service.engine.step==1025 and service.active==0
    assert (service.directory/'controls.json').read_bytes()==before and service.engine.command==old['command']
    assert replica.adopt(payload,execution)['already_selected'] and len(fetches)==5
    assert replica.permit_cached(execution,result)


@pytest.mark.parametrize('problem',['private_route','wrong_owner','withdrawn_during_replay','low_ram'])
def test_unapproved_or_interrupted_adoption_preserves_the_parent_state(tmp_path,monkeypatch,problem):
    replica,payload,execution,fetches,_=fixture(tmp_path,monkeypatch);service=replica.service
    parent=service.snapshot()['state_id']
    if problem=='private_route':service.knowledge.node.peers=lambda:[{'node_id':execution.origin_node_id,
        'host':'100.84.3.4','transport':'direct','encrypted':True}]
    if problem=='wrong_owner':execution.origin_node_id='d'*64
    if problem=='low_ram':execution.resource_requirements=lambda:ResourceRequirements(kind='training',ram_gb=1,disk_gb=.12)
    if problem=='withdrawn_during_replay':
        original=service.supervisor.run
        def withdraw(*args,**kwargs):
            audit=original(*args,**kwargs);OwnLearningSettings().save(service.directory);return audit
        service.supervisor.run=withdraw
    with pytest.raises(ValueError):replica.adopt(payload,execution)
    assert service.snapshot()['state_id']==parent and service.engine.step==1024 and service.active==0
    if problem!='withdrawn_during_replay':assert fetches==[]
