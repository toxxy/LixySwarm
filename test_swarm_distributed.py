"""Own training rejects stale, duplicate, overlapping or unauthenticated work."""
from dataclasses import replace
import hashlib

import pytest

from src.learning.swarm_distributed import (definition,policy_id,state_id,validate_payload,
    verify_contributions,verdict)
from src.learning.public_math import ARTIFACTS
from src.network.lsp import LSPIdentity
from src.network.work_protocol import WorkResult,ResultReceipt
from src.network.own_swarm import OwnSwarmSettings


def fixture_records():
    requester=LSPIdentity.generate();workers=[LSPIdentity.generate(),LSPIdentity.generate()]
    model='a'*64;state=state_id(model,'b'*64,1024);records=[]
    for index,worker in enumerate(workers):
        output={'training_policy_id':policy_id(),'state_id':state,'model_id':model,
            'dataset_id':ARTIFACTS['train'][0],'start':index*128,'count':128,
            'tokens':128*128,'loss':2.,'gradient_id':str(index+1)*64,'gradient_bytes':1024}
        result=WorkResult(job_id=hashlib.sha256(str(index).encode()).hexdigest(),status='ok',output=output)
        result=replace(result,receipt=ResultReceipt.create(result,worker,requester.node_id_hex).to_dict())
        records.append({'worker_id':worker.node_id_hex,'result':result.to_dict()})
    return records,workers,requester,model,state


def test_signed_productive_shards_are_distinct_and_bound_to_the_whole_training_state():
    records,workers,requester,model,state=fixture_records()
    options={'model_id':model,'training_state_id':state,'requester_id':requester.node_id_hex,
             'workers':[v.node_id_hex for v in workers]}
    assert len(verify_contributions(records,**options))==2
    with pytest.raises(ValueError,match='another model'):
        verify_contributions(records,**{**options,'training_state_id':state_id(model,'c'*64,1024)})
    with pytest.raises(ValueError,match='signature'):
        verify_contributions(records,**{**options,'requester_id':'d'*64})
    with pytest.raises(ValueError,match='signature'):
        verify_contributions([records[0],records[0]],**options)
    records[1]['result']['output']['start']=64
    old=WorkResult.from_dict(records[1]['result'])
    old=replace(old,receipt=ResultReceipt.create(old,workers[1],requester.node_id_hex).to_dict())
    records[1]['result']=old.to_dict()
    with pytest.raises(ValueError,match='overlap'):verify_contributions(records,**options)


def test_owner_can_select_only_typed_public_work_and_explicit_training_state():
    payload={'state_id':state_id('a'*64,'b'*64,1024),'model_id':'a'*64,
        'dataset_id':ARTIFACTS['train'][0],'training_policy_id':policy_id(),'start':0,'count':128}
    validate_payload(payload)
    for change in [{'program':'arbitrary.py'},{'dataset_id':'b'*64},{'count':257},{'start':4096},{'count':True}]:
        with pytest.raises(ValueError):validate_payload({**payload,**change})
    assert state_id('a'*64,'b'*64,1024)!=state_id('a'*64,'b'*64,1025)
    for command in [[],['relative-command']]:
        with pytest.raises(ValueError):OwnSwarmSettings(enabled=True,command=command,
            model_id='a'*64,optimizer_id='b'*64,step=1024).validate()
    assert not OwnSwarmSettings().enabled


def test_candidate_gate_requires_nll_gain_and_preserves_aggregate_byte_accuracy():
    before={'nll':2.,'byte_accuracy':.4,'predicted_bytes':57123,'documents':128}
    assert verdict(before,{**before,'nll':1.99,'byte_accuracy':.41})
    assert not verdict(before,{**before,'nll':1.99999})
    assert not verdict(before,{**before,'nll':1.99,'byte_accuracy':.399})
    with pytest.raises(ValueError):verdict(before,{**before,'nll':float('nan')})
    with pytest.raises(ValueError):verdict(before,{**before,'predicted_bytes':1})


