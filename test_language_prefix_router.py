from copy import deepcopy
import hashlib
import time

import pytest

from src.learning import language_prefix_router as protocol,language_prefix_router_job as job
from src.learning.public_math import canonical


def fixture():
    p=protocol.definition();names=['dolphin','zero','shuffled','public']
    snapshot={'split':'train','rows':[{'case':f'{i:064x}','question':str(i),'answer':'#### 1','final_number':'1'} for i in range(256)]}
    predictions={name:[{'index':i,'choice':0,'probabilities':[.75,.25]} for i in range(128,256)] for name in names}
    commitment={'protocol_id':protocol.protocol_id(),'feature_space_id':protocol.feature_id(),'expert_ids':p['expert_ids'],
        'learned_artifacts':{name:{'sha256':'1'*64,'bytes':100} for name in [*(n+'.safetensors' for n in names),'optimizer-state.pt']},
        'predictions':predictions}
    fit={'support_indices':list(range(0,128,4)),'query_indices':[i for i in range(128) if i%4],
        'phase1':{'steps':256},'phase2':{n:{'steps':256} for n in p['memory_modes']},
        'prediction_rows_used_for_fit':False,'memory_scope':'public-training-only'}
    outcomes={name:[{'case':row['case'],'answer':'#### 1' if i%mod==0 else '#### 2',
        'output_tokens':3,'generation_seconds':.1} for i,row in enumerate(snapshot['rows'][128:])]
        for name,mod in [('accepted-parent',2),('research-specialist',3)]}
    report={'protocol_id':protocol.protocol_id(),'commitment':commitment,
        'commitment_id':hashlib.sha256(canonical(commitment)).hexdigest(),'original_evaluation_opened':False,
        'native_promotion_authorized':False,'language_experts_updated':False,'committed_before_fresh_generation':True,
        'feature_binding':{'feature_space_id':protocol.feature_id(),'training_artifact_id':p['training_artifact_id'],
            'case_ids':[r['case'] for r in snapshot['rows']],'answer_tokens_used':False},
        'fit':fit,'outcomes':outcomes,'generation_expert_order':[[0,1] for _ in range(128)]}
    return snapshot,report


def test_prefix_protocol_reserves_disjoint_new_train_queries_without_changing_publication_gate():
    p=protocol.definition();assert set(p['fit_indices']).isdisjoint(p['prediction_indices'])
    assert max(p['prediction_indices'])<1024
    assert p['original_evaluation_opened'] is False and p['native_promotion_authorized'] is False
    snapshot,report=fixture();checked=protocol.verify_report(report,snapshot)
    assert checked['parent_correct']==64 and checked['specialist_correct']==43
    assert checked['variants']['public']['correct']==64
    assert checked['primary_actual_generations']==128 and checked['other_arms_are_counterfactual']


@pytest.mark.parametrize('change',[
    lambda r:r.update(original_evaluation_opened=True),lambda r:r.update(native_promotion_authorized=True),
    lambda r:r['fit'].update(prediction_rows_used_for_fit=True),
    lambda r:r['fit']['phase2']['zero'].update(steps=1),
    lambda r:r['generation_expert_order'].__setitem__(0,[1,0]),
    lambda r:r['outcomes']['accepted-parent'][0].update(case='f'*64),
    lambda r:r['feature_binding'].update(answer_tokens_used=True),
    lambda r:r['commitment']['predictions']['public'][0].update(choice=1),
    lambda r:r['commitment']['learned_artifacts']['public.safetensors'].update(sha256='private'),
])
def test_prefix_verifier_rejects_changed_fit_prediction_execution_scope_or_evidence(change):
    snapshot,report=fixture();change(report)
    with pytest.raises(ValueError):protocol.verify_report(report,snapshot)


def test_prefix_request_cannot_open_eval_and_cannot_overwrite_public_inputs(tmp_path):
    p=protocol.definition()
    request={'action':p['action'],'protocol_id':protocol.protocol_id(),'compute_dtype':'float32','streamed_base':True,
        'model_id':p['parent_model_id'],'parent_release_id':p['parent_release_id'],'base_directory':str(tmp_path/'base'),
        'adapter_file':str(tmp_path/'adapter'),'training_file':str(tmp_path/'train'),'specialist_file':str(tmp_path/'specialist'),
        'outcome_files':[str(tmp_path/'even'),str(tmp_path/'odd')],'output_directory':str(tmp_path/'output'),
        'device':'cpu','cpu_threads':2,'gpu_fraction':1.,'deadline':time.time()+10}
    job.validate_request(request)
    with pytest.raises(ValueError):job.validate_request({**request,'evaluation_file':str(tmp_path/'eval')})
    with pytest.raises(ValueError):job.validate_request({**request,'output_directory':str(tmp_path/'base/output')})
    with pytest.raises(ValueError):job.validate_request({**request,'outcome_files':['private-file']})
