"""Genuine own-model backward, serialized gradients and optimizer continuation."""
from dataclasses import replace
import hashlib
from pathlib import Path
import pytest
from test_swarm_distributed import fixture_records
from src.learning.swarm_distributed import policy_id,state_id
from src.learning.public_math import ARTIFACTS
from src.network.work_protocol import WorkResult,ResultReceipt


@pytest.mark.parametrize('n',[None,1,3,8,'food','proper'])
def test_tensor_own_shards_aggregate_with_resumed_adamw_and_verify_file_bindings(tmp_path,monkeypatch,n):
    import time
    import torch
    torch.set_num_threads(2)
    from safetensors.torch import save_file
    from src.learning import swarm_distributed_job as job
    from src.learning.swarm_foundation import make_foundation_model,predict
    from src.learning.swarm_foundation_protocol import protocol_id as init_id
    from src.network.artifact_store import digest_file
    records,workers,requester,_,_=fixture_records()
    selected_policy=policy_id()
    proper=n=='proper'
    food=n in ['food','proper']
    if food:n=3
    if n is not None:
        from src.network.lsp import LSPIdentity
        from src.learning.swarm_policy import network_policy_id
        workers=[LSPIdentity.generate() for _ in range(n)];requester=workers[0]
        selected_policy=network_policy_id()
    if food:
        from src.learning.own_food_catalog import PROGRAMS,PROPER_SCORE_PROGRAMS,policy_identifier
        from src.learning.swarm_policy import definition
        selected_policy=policy_identifier((PROPER_SCORE_PROGRAMS if proper else PROGRAMS)[0]['program_id'])
    # Same own architecture and genuine optimizer state; only data and the
    # quality scorer are scoped fixtures. This does not assert physical quality.
    model=make_foundation_model().train();optimizer=torch.optim.AdamW(model.parameters(),lr=.0006,weight_decay=.01)
    x=torch.arange(8).repeat(2,1);predict(model,x,x)[1].backward();optimizer.step()
    weights=tmp_path/'weights.safetensors'
    save_file({k:v.detach().clone().contiguous() for k,v in model.state_dict().items()},str(weights),
        metadata={'format':'lixyswarm-own-causal-byte-v1','protocol_id':init_id()})
    opt=tmp_path/'optimizer.pt'
    torch.save({'protocol_id':init_id(),'step':1,'optimizer':optimizer.state_dict(),
                'sampler_state':torch.Generator().get_state()},opt)
    model_id=digest_file(weights)[0];optimizer_id=digest_file(opt)[0];sid=state_id(model_id,optimizer_id,1)
    train=tmp_path/'train.json';train.write_text('fixture')
    evaluation=tmp_path/'evaluation.json';evaluation.write_text('fixture')
    snapshot={'rows':[{'question':f'Public example {i}?','answer':'A bounded public answer.'} for i in range(2*len(workers))]}
    monkeypatch.setattr(job,'load_snapshot',lambda *args:snapshot)
    if food:
        from src.learning import own_food
        from src.learning.swarm_foundation_job import documents
        monkeypatch.setattr(own_food,'load',lambda path,p,split:documents(snapshot))
    base={'training_policy_id':selected_policy,'state_id':sid,'model_id':model_id,'optimizer_id':optimizer_id,
        'step':1,'dataset_id':ARTIFACTS['train'][0],'model_file':str(weights),'training_file':str(train),
        'device':'cpu','cpu_threads':2,'deadline':time.time()+120}
    if food:base['dataset_id']=definition(selected_policy)['training_dataset_id']
    contributions=[];files={}
    for index,worker in enumerate(workers):
        out=job.run({**base,'action':'own-gradient','start':index*2,'count':2,
                     'output_directory':str(tmp_path/f'gradient-{index}')})
        files[out['gradient_id']]=out.pop('gradient_file')
        result=WorkResult(job_id=hashlib.sha256(f'own-test-{index}'.encode()).hexdigest(),status='ok',output=out)
        result=replace(result,receipt=ResultReceipt.create(result,worker,requester.node_id_hex).to_dict())
        contributions.append({'worker_id':worker.node_id_hex,'result':result.to_dict()})
    original=job.verify_contributions
    monkeypatch.setattr(job,'verify_contributions',lambda records,**kw:original(records,
        **kw,workers=[v.node_id_hex for v in workers]))
    metrics=iter([{'nll':2.,'byte_accuracy':.4,'predicted_bytes':57123,'documents':128},
                  {'nll':1.99,'byte_accuracy':.41,'predicted_bytes':57123,'documents':128}])
    if food:
        p=definition(selected_policy)
        old={'nll':2.,'byte_accuracy':.4,'predicted_bytes':57123,'documents':128}
        new={**old,'predicted_bytes':p['evaluation_prediction_bytes']}
        metrics=iter([new,old,{**new,'nll':1.99},{**old,'nll':1.999}])
    monkeypatch.setattr(job,'evaluate',lambda *args:next(metrics))
    candidate={**base,'action':'own-candidate','optimizer_file':str(opt),'evaluation_file':str(evaluation),
        'requester_id':requester.node_id_hex,'contributions':contributions,'gradient_files':files,
        'output_directory':str(tmp_path/'candidate')}
    if food:candidate['retention_file']=str(evaluation)
    result=job.run(candidate)
    saved=torch.load(result['candidate_optimizer_file'],weights_only=True)
    assert result['accepted'] and result['candidate_optimizer_step']==2 and saved['step']==2
    assert {int(v['step']) for v in saved['optimizer']['state'].values()}=={2}
    assert digest_file(weights)[0]==model_id and digest_file(opt)[0]==optimizer_id
    # The same signed records cannot authorize changed tensor bytes.
    first=Path(next(iter(files.values())))
    first.write_bytes(first.read_bytes()+b'x')
    with pytest.raises(ValueError,match='content does not match'):
        job.run({**candidate,'output_directory':str(tmp_path/'tampered')})
