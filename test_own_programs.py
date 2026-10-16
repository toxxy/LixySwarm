"""Durable new-food admission and preservation of earlier quality stops."""
from dataclasses import asdict
import hashlib
import json
from types import SimpleNamespace

import pytest
from src.learning.own_food_catalog import PROGRAMS,policy_identifier
from src.learning.swarm_policy import definition,network_policy_id,verdict,validate_payload,state_id
from src.network.own_network import OwnNetworkCoordinator
from src.network.own_network_settings import OwnNetworkSettings
from src.network.own_swarm import OwnSwarmSettings
from src.network.client_state import ClientSettings
from src.network.public_knowledge import KnowledgeSettings
from src.network import own_programs as programs
from src.network.telemetry import atomic_json


def test_food_policy_is_prospective_and_cannot_relax_legacy_or_retention():
    identifier=policy_identifier(PROGRAMS[0]['program_id']);p=definition(identifier)
    old={'nll':2.,'byte_accuracy':.4,'predicted_bytes':57123,'documents':128}
    before={**old,'predicted_bytes':p['evaluation_prediction_bytes'],'retention':old}
    after={**before,'nll':1.99,'retention':{**old,'nll':1.999}}
    assert verdict(before,after,identifier)
    assert not verdict(before,{**after,'retention':{**old,'nll':2.00001}},identifier)
    assert not verdict(before,{**after,'retention':{**old,'byte_accuracy':.399}},identifier)
    with pytest.raises(ValueError):verdict(before,after,network_policy_id())
    for invalid in [None,'','f'*64]:
        with pytest.raises(ValueError):definition(invalid)
    payload={'training_policy_id':identifier,'state_id':'a'*64,'model_id':'b'*64,
        'dataset_id':p['training_dataset_id'],'start':p['training_units']-1,'count':1}
    validate_payload(payload)
    with pytest.raises(ValueError):validate_payload({**payload,'count':2})
    with pytest.raises(ValueError):validate_payload({**payload,'training_policy_id':None})


def prepared(tmp_path,monkeypatch):
    node_id='1'*64
    selected=OwnSwarmSettings(version=2,enabled=True,command=['/own/LixySwarmEngineCPU'],
        model_id='a'*64,optimizer_id='b'*64,step=1033,requesters=[node_id]);selected.save(tmp_path)
    controls=OwnNetworkSettings(enabled=True,follow_enabled=True,owner_id=node_id,
        initial_state_id=state_id('a'*64,'b'*64,1033));controls.save(tmp_path)
    ClientSettings().save(tmp_path);KnowledgeSettings(enabled=True,consented_at=1).save(tmp_path)
    policy=definition(policy_identifier(PROGRAMS[0]['program_id']))
    sizes={policy[k+'_dataset_id']:policy[k+'_bytes'] for k in ['training','evaluation','retention']}
    monkeypatch.setattr(programs,'supports_programs',lambda _:True)
    monkeypatch.setattr(programs,'digest_file',lambda path:(path.name,sizes[path.name]))
    service=SimpleNamespace(directory=tmp_path,root=tmp_path/'public/own',engine=selected,active=0,
        store=SimpleNamespace(object_path=lambda i:tmp_path/i),ready=True,
        knowledge=SimpleNamespace(allowed=True,node=SimpleNamespace(identity=SimpleNamespace(node_id_hex=node_id))))
    return service,selected,controls


def test_program_selection_recovers_interrupted_two_file_commit_without_losing_state(tmp_path,monkeypatch):
    service,selected,controls=prepared(tmp_path,monkeypatch);original=programs.recover_selection
    monkeypatch.setattr(programs,'recover_selection',lambda directory:None)
    programs.select_program(service,PROGRAMS[0]['program_id'],controls.owner_id,coordinator=True,check=lambda:None)
    pending=json.loads((tmp_path/'own-program-pending.json').read_text())
    # Simulate process death after only the coordinator file was committed.
    OwnNetworkSettings(**pending['controls']).save(tmp_path)
    original(tmp_path)
    new=OwnSwarmSettings.load(tmp_path);current=OwnNetworkSettings.load(tmp_path)
    assert new.version==3 and new.program_id==PROGRAMS[0]['program_id']
    assert (new.model_id,new.optimizer_id,new.step)==(selected.model_id,selected.optimizer_id,1033)
    assert current.parent_program_id==controls.contract() and current.initial_state_id==controls.initial_state_id
    original(tmp_path)  # Idempotent recovery.
    assert OwnSwarmSettings.load(tmp_path)==new
    assert json.loads((tmp_path/'own-network/admissions'/new.program_id/'previous-controls.json').read_text())==asdict(controls)


def test_new_program_ledger_does_not_reset_or_overwrite_terminal_old_ledger(tmp_path,monkeypatch):
    service,selected,controls=prepared(tmp_path,monkeypatch)
    old={'version':2,'contract_id':controls.contract(),'training_policy_id':network_policy_id(),
        'round':5,'cursor':3584,'workers':{},'rejections':3,'accepted':1,'pending':None,'status':'no_improvement'}
    path=tmp_path/'own-network/ledger.json';atomic_json(path,old);before=path.read_bytes()
    programs.select_program(service,PROGRAMS[0]['program_id'],controls.owner_id,coordinator=True,check=lambda:None)
    c=OwnNetworkCoordinator(service,None)
    assert c.ledger is None and c.path!=path and c.settings.initial_cursor==0
    assert path.read_bytes()==before
    # A restarted new coordinator loads its own progress rather than the old stop.
    atomic_json(c.path,{**old,'version':3,'contract_id':c.settings.contract(),
        'training_policy_id':c.settings.policy_id(),'round':1,'cursor':64,'rejections':0})
    resumed=OwnNetworkCoordinator(service,None)
    assert resumed.ledger['cursor']==64 and resumed.ledger['round']==1 and path.read_bytes()==before


def test_paused_or_changed_training_state_blocks_incomplete_admission(tmp_path,monkeypatch):
    service,selected,controls=prepared(tmp_path,monkeypatch)
    original=programs.recover_selection;monkeypatch.setattr(programs,'recover_selection',lambda _:None)
    programs.select_program(service,PROGRAMS[0]['program_id'],controls.owner_id,coordinator=True,check=lambda:None)
    client=ClientSettings.load(tmp_path);client.paused=True;client.save(tmp_path)
    with pytest.raises(ValueError,match='paused'):original(tmp_path)
    client.paused=False;client.save(tmp_path)
    selected.step+=1;selected.save(tmp_path)
    with pytest.raises(ValueError,match='conflicts'):original(tmp_path)


def test_retained_public_evidence_counts_toward_work_quota(tmp_path,monkeypatch):
    from pathlib import Path
    from src.network.own_storage import usage
    monkeypatch.setattr(Path,'home',classmethod(lambda cls:tmp_path))
    root=tmp_path/'.local/state';node=root/'lixyswarm';public=node/'public';public.mkdir(parents=True)
    (public/'selected').write_bytes(b'own')
    saved=root/'lixyswarm-public-proof';saved.mkdir();(saved/'previous').write_bytes(b'prior public state')
    atomic_json(node/'own-storage-reservations.json',{'version':1,'directories':[str(saved)]})
    assert usage(node)==len(b'ownprior public state')
    atomic_json(node/'own-storage-reservations.json',{'version':1,'directories':[str(tmp_path/'private')]})
    with pytest.raises(ValueError,match='separate local public state'):usage(node)
