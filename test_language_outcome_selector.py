from copy import deepcopy
import hashlib

import pytest

from src.learning import language_outcome_selector as selector
from src.learning.language_outcomes import partition,protocol_id as outcomes_id
from src.learning.public_math import canonical


def fixture(monkeypatch):
    snapshot={'split':'train','rows':[{'case':f'{i:064x}','question':'public fixture','answer':'#### 1','final_number':'1'} for i in range(128)]}
    outcomes=[]
    for shard in [0,1]:
        selected,indices=partition(snapshot,shard);experts={}
        for name,correct_classes in [('accepted-parent',{0,3}),('research-specialist',{1,3})]:
            answers=[{'case':r['case'],'answer':'#### '+('1' if i%4 in correct_classes else '2')} for i,r in zip(indices,selected['rows'])]
            experts[name]={'answers':answers,'output_tokens':[4]*64,'generation_seconds':[.25]*64,'nll':.5,'tokens':256,
                'correct':sum(i%4 in correct_classes for i in indices),'total':64,'at_output_limit':0}
        outcomes.append({'protocol_id':outcomes_id(),'expert_ids':[selector.PARENT_MODEL,selector.SPECIALIST_ID],
            'shard':shard,'training_indices':indices,'experts':experts,'evaluation_data_used':False,'optimizer_updates':False})
    monkeypatch.setattr(selector,'OUTCOME_IDS',tuple(hashlib.sha256(canonical(v)).hexdigest() for v in outcomes))
    return snapshot,outcomes


def test_labels_use_strict_outcomes_and_remain_neutral_without_expert_preference(monkeypatch):
    snapshot,outcomes=fixture(monkeypatch);rows=selector.dataset(outcomes,snapshot)
    assert [r['target'] for r in rows[:4]]==[[1.,0.],[0.,1.],[.5,.5],[.5,.5]]
    assert sum(r['correct'][0] for r in rows)==64
    changed=deepcopy(outcomes);changed[0]['experts']['accepted-parent']['answers'][0]['answer']='#### 7'
    with pytest.raises(ValueError,match='exact independently verified'):selector.dataset(changed,snapshot)
    with pytest.raises(ValueError):selector.dataset(outcomes[::-1],snapshot)
    with pytest.raises(ValueError):selector.dataset(outcomes,{**snapshot,'split':'evaluation'})


def test_every_prediction_is_outside_its_training_and_memory_queries():
    predicted=[]
    for i in range(8):
        train,test=selector.fold_indices(i);support=train[::4]
        assert len(train)==112 and len(test)==16 and not set(train)&set(test) and not set(support)&set(test)
        predicted.extend(test)
    assert sorted(predicted)==list(range(128))
    with pytest.raises(ValueError):selector.fold_indices(True)


def test_verifier_reconstructs_predictions_and_rejects_fold_leakage(monkeypatch):
    snapshot,outcomes=fixture(monkeypatch);rows=selector.dataset(outcomes,snapshot);p=selector.definition()
    folds=[]
    for i in range(8):
        train,test=selector.fold_indices(i);support=list(range(0,112,4))
        folds.append({'training_indices':train,'prediction_indices':test,'fit':{'support_indices':support,
            'query_indices':[j for j in range(112) if j not in support],'phase1':{'steps':p['phase_steps']},
            'phase2':{n:{'steps':p['phase_steps']} for n in p['memory_modes']},'memory_scope':'public-training-only',
            'held_out_used_for_training_or_selection':False}})
    report={'protocol_id':selector.protocol_id(),'original_evaluation_opened':False,'native_promotion_authorized':False,
        'language_experts_updated':False,'folds':folds,'predictions':{n:[{'index':i,'choice':0,'probabilities':[.5,.5]} for i in range(128)]
            for n in ['dolphin',*p['memory_modes']]}}
    proof=selector.verify_report(report,rows)
    assert all(v['correct']==64 and v['regressions']==0 for v in proof['variants'].values())
    bad=deepcopy(report);bad['folds'][0]['training_indices'].append(0)
    with pytest.raises(ValueError,match='mixed prediction queries'):selector.verify_report(bad,rows)
    bad=deepcopy(report);bad['predictions']['public'][0]['choice']=1
    with pytest.raises(ValueError,match='frozen learned'):selector.verify_report(bad,rows)
