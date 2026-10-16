"""The own learner uses the native node identity and revocable owner leases."""
from dataclasses import replace
import hashlib
import time
from types import SimpleNamespace

import pytest

from src.learning.swarm_distributed import policy_id,state_id,GRADIENT_OPERATION
from src.learning.public_math import ARTIFACTS
from src.network import local_own_work as native
from src.network.artifact_store import ArtifactStore
from src.network.lsp import LSPIdentity
from src.network.work_protocol import WorkResult,ResultReceipt


def fixture(tmp_path,monkeypatch):
    own=LSPIdentity.generate();calls=[]
    monkeypatch.setattr(native,'WORKERS',(own.node_id_hex,))
    store=ArtifactStore(tmp_path/'artifacts')
    path=tmp_path/'gradient';path.write_bytes(b'public synthetic own gradient')
    item=store.import_file(path,kind='gradient')
    mid='a'*64;sid=state_id(mid,'b'*64,1024)
    payload={'training_policy_id':policy_id(),'state_id':sid,'model_id':mid,
        'dataset_id':ARTIFACTS['train'][0],'start':0,'count':128}
    node=SimpleNamespace(identity=own)
    def execute(operation,supplied,*args,**kwargs):
        calls.append(operation);assert supplied==payload
        result=WorkResult(job_id=hashlib.sha256(b'own-owner-test').hexdigest(),status='ok',
            output={**payload,'gradient_id':item.artifact_id,'gradient_bytes':item.size,'tokens':128,'loss':2.})
        return replace(result,receipt=ResultReceipt.create(result,own,own.node_id_hex).to_dict())
    knowledge=SimpleNamespace(node=node,work=SimpleNamespace(execute_local=execute))
    service=SimpleNamespace(ready=True,knowledge=knowledge,store=store,directory=tmp_path,
        snapshot=lambda:{'state_id':sid,'model_id':mid},
        transfer=SimpleNamespace(fetch=lambda *_a,**_kw:store.object_path(item.artifact_id)))
    manager=native.LocalOwnWork(service)
    value={'request_id':'a'*32,'worker_id':own.node_id_hex,'operation':GRADIENT_OPERATION,'payload':payload}
    return manager,value,calls


def done(manager,request):
    deadline=time.monotonic()+3
    while time.monotonic()<deadline:
        value=manager.status(request)
        if value['state']!='running':return value
        time.sleep(.01)
    raise AssertionError('Synthetic own work did not finish')


def test_own_owner_work_has_no_shadow_identity_and_fetches_only_the_signed_artifact(tmp_path,monkeypatch):
    manager,value,calls=fixture(tmp_path,monkeypatch)
    try:
        manager.start(value);result=done(manager,value['request_id'])
        assert result['state']=='complete' and result['artifact_initially_local']
        assert result['result']['receipt']['requester_node_id']==value['worker_id']
        assert result['route']['transport']=='local' and result['encoded_data_bytes']==0
        assert calls==[GRADIENT_OPERATION]
    finally:manager.close()


@pytest.mark.parametrize('change',['unknown_worker','private_path','stale_state'])
def test_own_owner_rejects_invalid_jobs_before_dispatch(tmp_path,monkeypatch,change):
    manager,value,calls=fixture(tmp_path,monkeypatch)
    try:
        if change=='unknown_worker':value['worker_id']='f'*64
        if change=='private_path':value['payload']['private_file']='/private'
        if change=='stale_state':value['payload']['state_id']='f'*64
        with pytest.raises(ValueError):manager.start(value)
        assert calls==[]
    finally:manager.close()
