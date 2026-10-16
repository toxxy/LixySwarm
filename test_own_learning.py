"""Durable recovery across the two-sided selection boundary and bounded rejection."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from src.learning import swarm_distributed
from src.network import own_learning,own_learning_settings,own_state
from src.network.own_learning_settings import OwnLearningSettings
from src.network.own_replication import HEAD_OPERATION,ADOPT_OPERATION
from src.network.artifact_store import ArtifactStore
from test_own_state import transition_fixture


def setup(tmp_path,monkeypatch):
    result,raw,workers,audit=transition_fixture(tmp_path)
    output=result['output'];owner=workers[0]
    monkeypatch.setattr(own_learning,'WORKERS',workers)
    monkeypatch.setattr(own_learning_settings,'WORKERS',workers)
    original=swarm_distributed.verify_contributions
    monkeypatch.setattr(own_learning,'verify_contributions',lambda records,**kw:original(records,**kw,workers=workers))
    original_transition=own_state.verify_transition
    monkeypatch.setattr(own_learning,'verify_transition',lambda r,b:original_transition(r,b,workers=workers))
    selected={'state_id':output['state_id'],'model_id':output['model_id'],'ready':True}
    service=SimpleNamespace(directory=tmp_path,root=tmp_path/'public/own',ready=True,
        knowledge=SimpleNamespace(allowed=True,node=SimpleNamespace(identity=SimpleNamespace(node_id_hex=owner))),
        store=ArtifactStore(tmp_path/'public/own/artifacts'),snapshot=lambda:deepcopy(selected))
    contributions=json.loads(raw)['contributions']
    # The fixture protocol uses 0:128 and 128:256. These signed ranges are
    # deliberately bound to the same cursor by the controller under test.
    settings=OwnLearningSettings(enabled=True,follow_enabled=True,owner_id=owner,
        initial_state_id=selected['state_id'],initial_cursor=0,maximum_rounds=1)
    settings.save(tmp_path)
    class Work:
        def __init__(self):self.jobs={};self.started=[];self.cancelled=[]
        def start(self,value):
            self.started.append(value)
            response=result if value['operation']==swarm_distributed.CANDIDATE_OPERATION else next(
                row['result'] for row in contributions if row['worker_id']==value['worker_id'])
            self.jobs[value['request_id']]={'state':'complete','result':deepcopy(response)}
        def status(self,key):return self.jobs[key]
        def cancel(self,key):self.cancelled.append(key)
    work=Work();coordinator=own_learning.OwnLearningCoordinator(service,work);coordinator.settings=settings
    coordinator.ledger={'version':1,'contract_id':settings.contract(),'training_policy_id':swarm_distributed.policy_id(),
        'initial_state_id':selected['state_id'],'state_id':selected['state_id'],'round':0,'cursor':0,'epoch':0,
        'accepted':0,'rejections':0,'pending':None,'history':[],'status':'ready','error':None,'retry_at':0}
    coordinator._save()
    # These are transport/compute boundaries; protocol signatures and replay
    # have dedicated tests. Here each side can fail independently of the other.
    replicas={workers[1]:output['state_id']};failures={'remote':False};rpc_calls=[]
    def rpc(worker,operation,payload,contract):
        rpc_calls.append((worker,operation))
        if operation==HEAD_OPERATION:
            return {'output':{'ready':True,'state_id':replicas[worker],'training_policy_id':swarm_distributed.policy_id()}}
        assert operation==ADOPT_OPERATION and payload['candidate_result']==result
        if worker==owner:
            selected.update(state_id=audit['state_id'],model_id=audit['model_id'])
        else:
            if failures['remote']:raise ConnectionError('fixture replica temporarily unavailable')
            replicas[worker]=audit['state_id']
        return {'output':{'success':True,'state_id':audit['state_id']}}
    coordinator._rpc=rpc
    return coordinator,work,result,selected,replicas,failures,rpc


def test_restart_after_local_selection_resumes_replication_without_retraining_or_cursor_reset(tmp_path,monkeypatch):
    c,work,result,selected,replicas,failures,rpc=setup(tmp_path,monkeypatch)
    failures['remote']=True;c._run(c.settings.contract())
    assert c.ledger['status']=='waiting' and c.ledger['round']==0 and c.ledger['cursor']==0
    assert c.ledger['pending']['candidate']['result']==result
    assert selected['state_id']!=c.ledger['state_id'] and len(work.started)==3
    failures['remote']=False
    resumed=own_learning.OwnLearningCoordinator(c.service,work);resumed.settings=c.settings;resumed._rpc=rpc
    resumed._run(c.settings.contract())
    assert resumed.ledger['round']==1 and resumed.ledger['cursor']==256 and resumed.ledger['accepted']==1
    assert resumed.ledger['pending'] is None and len(work.started)==3
    assert set(replicas.values())=={selected['state_id']}
    resumed.update();assert resumed.ledger['status']=='complete'


def test_rejected_cycle_records_cost_and_advances_data_without_changing_weights(tmp_path,monkeypatch):
    c,work,result,selected,*_=setup(tmp_path,monkeypatch);initial=selected['state_id']
    # A trusted local worker returns a genuine rejection at its compute boundary.
    result['output']['accepted']=False;result['output']['after']=deepcopy(result['output']['before'])
    c._run(c.settings.contract())
    assert c.ledger['round']==1 and c.ledger['accepted']==0 and c.ledger['rejections']==1
    assert c.ledger['cursor']==256 and selected['state_id']==initial
    record=json.loads((c.root/'rounds/0000.json').read_text())
    assert not record['accepted'] and record['finished_at']>=record['started_at'] and record['autonomous']


def test_pause_cancels_incomplete_work_and_preserves_the_same_pending_ranges(tmp_path,monkeypatch):
    c,work,*_=setup(tmp_path,monkeypatch)
    def pause(_key):c.cancel.set();return {'state':'running'}
    work.status=pause;c._run(c.settings.contract())
    assert c.ledger['status']=='paused' and c.ledger['round']==0 and c.ledger['cursor']==0
    assert len(work.cancelled)==2 and c.ledger['pending']['assignments'][1]['start']==128


def test_changing_program_limits_does_not_reset_or_replace_durable_history(tmp_path,monkeypatch):
    c,*_=setup(tmp_path,monkeypatch);before=c.path.read_bytes()
    c.settings.maximum_rounds=2;c.settings.save(tmp_path);c.update()
    assert 'program changed' in c.error and c.path.read_bytes()==before


def test_saved_contribution_from_another_range_is_rejected_before_candidate(tmp_path,monkeypatch):
    c,work,*_=setup(tmp_path,monkeypatch)
    c.ledger['cursor']=128;c._save()
    c._run(c.settings.contract())
    assert c.ledger['round']==0 and 'curriculum assignment' in c.ledger['error']
    assert len(work.started)==2


def test_three_quality_rejections_stop_the_program_without_erasing_its_history(tmp_path,monkeypatch):
    c,work,*_=setup(tmp_path,monkeypatch)
    c.settings.maximum_rounds=10;c.settings.save(tmp_path)
    c.ledger.update(contract_id=c.settings.contract(),round=3,rejections=3,cursor=768,
        history=[{'round':i,'accepted':False} for i in range(3)])
    c._save();c.update()
    assert c.ledger['status']=='no_improvement' and c.ledger['round']==3 and c.ledger['cursor']==768
    assert work.started==[] and c.ledger['rejections']==3
