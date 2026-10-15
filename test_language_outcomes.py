from copy import deepcopy
from types import SimpleNamespace

import pytest

from experiments.research_artifact import authorize,initialize
from src.learning.language_outcomes import (PARENT_MODEL,SPECIALIST_ID,definition,partition,protocol_id,
    score_training_answers,verify_outcomes)


def fixture():
    snapshot={'split':'train','rows':[{'case':f'{i:064x}','answer':'#### 1','question':'public fixture','final_number':'1'} for i in range(128)]}
    selected,indices=partition(snapshot,1)
    result={'answers':[{'case':r['case'],'answer':'#### 1'} for r in selected['rows']],
        'output_tokens':[4]*64,'generation_seconds':[.25]*64,'nll':.5,'tokens':256,'correct':64,'total':64,'at_output_limit':0}
    value={'protocol_id':protocol_id(),'expert_ids':[PARENT_MODEL,SPECIALIST_ID],'shard':1,'training_indices':indices,
        'experts':{'accepted-parent':deepcopy(result),'research-specialist':deepcopy(result)},'evaluation_data_used':False,'optimizer_updates':False}
    return snapshot,value


def test_train_partitions_are_disjoint_complete_and_not_specialist_updates():
    snapshot,value=fixture();a,ai=partition(snapshot,0);b,bi=partition(snapshot,1)
    assert set(ai).isdisjoint(bi) and sorted(ai+bi)==list(range(128))
    assert max(ai+bi)<1024 and definition()['optimizer_updates'] is False
    report=verify_outcomes(value,snapshot)
    assert report['experts']['accepted-parent']['correct']==64
    assert report['experts']['research-specialist']['output_tokens']==256
    assert not report['native_publication_authorized']
    with pytest.raises(ValueError):partition({**snapshot,'split':'evaluation'},0)
    with pytest.raises(ValueError):partition(snapshot,True)


@pytest.mark.parametrize('mutate',[
    lambda v:v.update(evaluation_data_used=True),lambda v:v.update(expert_ids=[SPECIALIST_ID,PARENT_MODEL]),
    lambda v:v['training_indices'].__setitem__(0,1024),
    lambda v:v['experts']['accepted-parent']['answers'].reverse(),
    lambda v:v['experts']['accepted-parent']['answers'][0].update(answer='the answer is 1'),
    lambda v:v['experts']['accepted-parent']['generation_seconds'].__setitem__(0,float('nan')),
    lambda v:v['experts']['accepted-parent']['output_tokens'].__setitem__(0,385)])
def test_outcome_verifier_rejects_rebinding_label_cost_or_partition_changes(mutate):
    snapshot,value=fixture();mutate(value)
    with pytest.raises(ValueError):verify_outcomes(value,snapshot)


def test_research_handler_authorizes_exact_signed_origin_artifact_and_consent():
    execution=SimpleNamespace(origin_node_id='receiver',raise_if_cancelled=lambda:None)
    authorize({'artifact_id':'model'},execution,'receiver','model',lambda:True)
    for peer,artifact,allowed in [('stranger','model',True),('receiver','private',True),('receiver','model',False)]:
        with pytest.raises(ValueError):authorize({'artifact_id':artifact},execution,peer,'model',lambda:allowed)


def test_transport_identity_is_fresh_and_cannot_replace_existing_research(tmp_path):
    directory=tmp_path/'transport';identifier=initialize(directory)
    assert len(identifier)==64 and (directory/'helper.key').is_file()
    with pytest.raises(ValueError):initialize(directory)
