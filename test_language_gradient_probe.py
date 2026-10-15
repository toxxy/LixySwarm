"""Comparable work plans and one live connection per public requester."""
import pytest
import json
from copy import deepcopy
from types import SimpleNamespace

from experiments.language_gradient_probe import requester_lock,shard_plan,load_checkpoint
from src.network.work_protocol import WorkResult
from test_language_versions import fixture


def test_one_and_two_pc_plans_cover_identical_work_without_duplicate_shards():
    one,two='a'*64,'b'*64
    baseline=shard_plan([one],8,2)
    distributed=shard_plan([one,two],8)
    assert [index for index,_worker in baseline]==[index for index,_worker in distributed]==[0,1]
    assert [worker for _index,worker in baseline]==[one,one]
    assert [worker for _index,worker in distributed]==[one,two]


@pytest.mark.parametrize('workers,count,shards',[(['a'*64]*2,8,2),(['a'*64],9,2),(['a'*64,'b'*64],8,1),(['a'*64],8,17),(['bad'],8,2)])
def test_invalid_comparison_work_is_rejected(workers,count,shards):
    with pytest.raises(ValueError):shard_plan(workers,count,shards)


def test_requester_cannot_be_reused_concurrently_and_unlocks_after_failure(tmp_path):
    with pytest.raises(ValueError):
        with requester_lock('a'*64,directory=tmp_path):
            with pytest.raises(RuntimeError,match='identity is already in use'):
                with requester_lock('a'*64,timeout_s=0,directory=tmp_path):pytest.fail('duplicate requester')
            raise ValueError('fixture interrupted')
    with requester_lock('a'*64,timeout_s=0,directory=tmp_path):pass


@pytest.mark.parametrize('failure',[None,'signature','worker','requester','head','shard','timing'])
def test_recovery_verifies_signed_work_and_its_original_bindings(monkeypatch,tmp_path,failure):
    _store,head,update,requester=fixture(monkeypatch,tmp_path/'source')
    _next,evaluation=update(head);result=evaluation['contributions'][0]
    worker=update.workers[0].node_id_hex
    saved={'result':deepcopy(result),'worker_id':worker,'base_name':'qwen3-0.6b',
        'release_id':head['release']['release_id'],'compute_seconds':1,'request_started_at':100}
    payload={key:result['output'][key] for key in ['adapter_id','dataset_id','start','count']}
    requester_id=requester.node_id_hex
    if failure=='signature':saved['result']['output']['tokens']+=1
    elif failure=='worker':saved['worker_id']='a'*64
    elif failure=='requester':requester_id='b'*64
    elif failure=='head':saved['release_id']='c'*64
    elif failure=='shard':payload['start']+=1
    elif failure=='timing':saved['compute_seconds']=float('nan')
    path=tmp_path/'checkpoint.json';path.write_text(json.dumps(saved))
    def recover():return load_checkpoint(path,worker=worker,requester=requester_id,
        base_name='qwen3-0.6b',release_id=head['release']['release_id'],payload=payload)
    if failure:
        with pytest.raises(ValueError):recover()
    else:assert recover()['result']==result


def test_interrupted_transfer_recovers_without_another_gradient_request(monkeypatch,tmp_path):
    from experiments import language_gradient_probe as probe
    _store,head,update,requester=fixture(monkeypatch,tmp_path/'source')
    _next,evaluation=update(head);result=WorkResult.from_dict(evaluation['contributions'][0])
    worker=update.workers[0].node_id_hex;submissions=[];transfers=[]
    class Node:
        def __init__(self,*_a,**_kw):self.connected_node_ids=[worker]
        def start(self):pass
        def stop(self):pass
        def peers(self):return [{'node_id':worker,'encrypted':True,'host':'8.8.8.8','transport':'relay'}]
    class Work:
        def __init__(self,*_a,**_kw):pass
        def close(self):pass
        def submit(self,*a,**kw):submissions.append((a,kw));return result
    class Artifacts:
        def __init__(self,*_a,**_kw):pass
        def fetch(self,*a,**kw):
            transfers.append((a,kw))
            if len(transfers)<=3:raise TimeoutError('injected transfer interruption')
            return tmp_path/'gradient'
    monkeypatch.setattr(probe,'LSPNodeV3',Node);monkeypatch.setattr(probe,'WorkCoordinator',Work)
    monkeypatch.setattr(probe,'ArtifactService',Artifacts)
    monkeypatch.setattr(probe,'fetch_history',lambda *_a:[head])
    monkeypatch.setattr(probe,'base_hash',lambda _name:head['descriptor']['base_hash'])
    args=SimpleNamespace(worker=[worker],count=4,shards=1,parallel_jobs=1,directory=tmp_path/'trial',
        resume=False,base_name='qwen3-0.6b',legacy=False,candidate_worker=None)
    with pytest.raises(TimeoutError):probe.run(args,requester)
    original=json.loads((args.directory/'report.json').read_text())
    assert not original['success'] and (args.directory/'signed-contribution-0.json').is_file()
    with pytest.raises(ValueError,match='requires --resume'):probe.run(args,requester)
    args.resume=True;probe.run(args,requester)
    recovered=json.loads((args.directory/'report.json').read_text())
    assert recovered['success'] and recovered['reused_contributions']==1
    assert len(submissions)==1 and len(transfers)==4
    assert recovered['signed_results']==[result.to_dict()]
    assert recovered['contributions'][0]['reused_signed_contribution']
    assert not recovered['eligible_for_uncontended_scaling_comparison']
    assert json.loads((args.directory/'report-before-recovery-1.json').read_text())==original
    # Even a later valid head cannot make old gradients eligible. Preserve the
    # original evidence and reject before any calculation or artifact request.
    monkeypatch.setattr(probe,'fetch_history',lambda *_a:[head,_next])
    with pytest.raises(ValueError,match='head changed'):probe.run(args,requester)
    assert len(submissions)==1 and len(transfers)==4
    assert json.loads((args.directory/'report.json').read_text())['head']==head
