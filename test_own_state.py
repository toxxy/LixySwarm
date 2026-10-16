"""Signed state adoption is atomic, replay safe and preserves local consent."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from src.learning.public_math import canonical
from src.learning.swarm_distributed import policy_id,state_id,WORKERS
from src.network import own_state
from src.network.own_swarm import OwnSwarmSettings
from src.network.client_state import ClientSettings
from src.network.work_protocol import WorkResult,ResultReceipt
from src.network.artifact_store import ArtifactStore
from test_swarm_distributed import fixture_records


def transition_fixture(tmp_path):
    records,workers,evaluator,mid,parent=fixture_records()
    store=ArtifactStore(tmp_path/'public/own/artifacts',max_artifact_bytes=16*1024**2,max_total_bytes=256*1024**2)
    model=tmp_path/'model';model.write_bytes(b'new own model fixture')
    optimizer=tmp_path/'optimizer';optimizer.write_bytes(b'continued own optimizer fixture')
    model=store.import_file(model,kind='model');optimizer=store.import_file(optimizer,kind='other')
    next_state=state_id(model.artifact_id,optimizer.artifact_id,1025)
    before={'nll':2.,'byte_accuracy':.4,'predicted_bytes':57123,'documents':128}
    after={**before,'nll':1.99,'byte_accuracy':.41}
    shared={'accepted':True,'training_policy_id':policy_id(),'state_id':parent,'model_id':mid,
        'optimizer_id':'b'*64,'before':before,'after':after,'previous_optimizer_step':1024,
        'candidate_optimizer_step':1025,'prediction_bytes':32768}
    raw=canonical({**shared,'candidate_state_id':next_state,'contributions':records})
    report=tmp_path/'evaluation';report.write_bytes(raw);evaluation=store.import_file(report,kind='evaluation')
    output={**shared,'candidate_model_id':model.artifact_id,'candidate_model_bytes':model.size,
        'candidate_optimizer_id':optimizer.artifact_id,'candidate_optimizer_bytes':optimizer.size,
        'evaluation_id':evaluation.artifact_id,'evaluation_bytes':evaluation.size}
    result=WorkResult(job_id='f'*64,status='ok',output=output)
    result=replace(result,receipt=ResultReceipt.create(result,evaluator,evaluator.node_id_hex).to_dict())
    audit={'success':True,'state_id':next_state,'model_id':model.artifact_id,'optimizer_id':optimizer.artifact_id,
           'training_policy_id':policy_id(),'saved_state_unchanged':True,'gradient_replay_verified':True}
    command=tmp_path/'installed-engine';command.write_text('local fixture')
    OwnSwarmSettings(enabled=True,command=[str(command)],model_id=mid,optimizer_id='b'*64,
        step=1024,requesters=list(WORKERS)).save(tmp_path)
    ClientSettings().save(tmp_path)
    return result.to_dict(),raw,[v.node_id_hex for v in workers],audit


def test_signed_transition_binds_both_physical_contributions_and_optimizer_ancestry(tmp_path):
    result,raw,workers,audit=transition_fixture(tmp_path)
    verified=own_state.verify_transition(result,raw,workers=workers)
    assert verified['state_id']==audit['state_id'] and verified['step']==1025
    with pytest.raises(ValueError,match='evaluation differs'):
        own_state.verify_transition(result,raw+b'\n',workers=workers)
    with pytest.raises(ValueError,match='known physical'):
        own_state.verify_transition(result,raw,workers=WORKERS)


def test_adoption_recovers_idempotently_and_keeps_resource_and_engine_permissions(tmp_path,monkeypatch):
    result,raw,workers,audit=transition_fixture(tmp_path)
    original=own_state.verify_transition
    monkeypatch.setattr(own_state,'verify_transition',lambda r,b:original(r,b,workers=workers))
    controls=(tmp_path/'controls.json').read_bytes();old=OwnSwarmSettings.load(tmp_path)
    adopted=own_state.activate(tmp_path,result,audit)
    selected=OwnSwarmSettings.load(tmp_path)
    assert adopted['success'] and not adopted['already_selected'] and selected.step==1025
    assert selected.command==old.command and selected.requesters==old.requesters and selected.enabled==old.enabled
    assert (tmp_path/'controls.json').read_bytes()==controls
    again=own_state.activate(tmp_path,result,audit)
    assert again['already_selected'] and OwnSwarmSettings.load(tmp_path).step==1025
    assert (tmp_path/'public/own/history'/(audit['state_id']+'.json')).is_file()


def test_adoption_rejects_missing_replay_proof_and_local_state_changes(tmp_path,monkeypatch):
    result,raw,workers,audit=transition_fixture(tmp_path)
    original=own_state.verify_transition
    monkeypatch.setattr(own_state,'verify_transition',lambda r,b:original(r,b,workers=workers))
    with pytest.raises(ValueError,match='independent local'):
        own_state.activate(tmp_path,result,{**audit,'gradient_replay_verified':False})
    selected=OwnSwarmSettings.load(tmp_path);selected.step=1026;selected.save(tmp_path)
    with pytest.raises(ValueError,match='does not extend'):
        own_state.activate(tmp_path,result,audit)
