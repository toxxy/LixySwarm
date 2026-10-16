import time

import pytest

from src.learning.language_response_router import definition,select_secondary,response_messages
from src.learning.language_response_router_job import validate_request,encode_response
from src.learning.language_retention import PARENT_MODEL,PARENT_RELEASE
from src.learning.language_response_router import protocol_id


def test_specialist_needs_predicted_gain_after_cost_and_a_remaining_budget():
    assert select_secondary([.1,.9,.05,.5],0)=='specialist'
    assert select_secondary([.1,.9,.05,.5],8)=='repair'
    assert select_secondary([.2,.25,.05,.9],0)=='repair'
    assert select_secondary([.8,.8,.1,.1],0)=='repair'
    p=definition()
    assert p['fresh_evaluation_performed_by_fit'] is False and p['language_experts_updated'] is False
    assert p['future_evaluation']['minimum_new_correct_over_parent']==4
    assert p['future_evaluation']['maximum_regressions']==0
    assert p['future_evaluation']['maximum_time_ratio_to_parent_repair']==1.1


@pytest.mark.parametrize('prediction,spent',[([True,.9,.1,.1],0),([.1,float('nan'),.1,.2],0),
    ([.1,.5,-.1,.1],0),([.1,.9,.1,2.],0),([.1,.9],0),([.1,.9,.1,.1],True),([.1,.9,.1,.1],9)])
def test_invalid_probabilities_or_costs_cannot_trigger_a_route(prediction,spent):
    with pytest.raises(ValueError):select_secondary(prediction,spent)


def test_encoder_uses_only_the_public_question_and_parent_draft():
    row={'question':'public question','draft':'generated draft','target':'reference-only-canary','answer':'another-reference'}
    messages=response_messages(row)
    assert len(messages)==2 and messages[-1]=={'role':'assistant','content':'generated draft'}
    assert 'canary' not in str(messages) and 'another-reference' not in str(messages)


def request(tmp_path):
    return {'action':definition()['action'],'protocol_id':protocol_id(),'model_id':PARENT_MODEL,
        'parent_release_id':PARENT_RELEASE,'base_directory':str(tmp_path/'base'),'adapter_file':str(tmp_path/'adapter'),
        'training_file':str(tmp_path/'train'),'outcome_files':[str(tmp_path/'a'),str(tmp_path/'b')],
        'repair_files':[str(tmp_path/'c'),str(tmp_path/'d')],'output_directory':str(tmp_path/'output'),
        'device':'cpu','cpu_threads':2,'gpu_fraction':1.,'deadline':time.time()+100}


@pytest.mark.parametrize('field,value',[('evaluation_file','/never/evaluation'),('snapshot_file','/never/future-test'),
    ('gpu_fraction',True),('gpu_fraction',float('nan')),('cpu_threads',0),('deadline',float('inf')),
    ('model_id','0'*64),('repair_files',[])])
def test_fit_cannot_accept_test_data_or_change_parent_or_resource_bounds(tmp_path,field,value):
    r=request(tmp_path);r[field]=value
    with pytest.raises(ValueError):validate_request(r)


def test_fit_preserves_inputs_and_old_evidence(tmp_path):
    r=request(tmp_path);validate_request(r)
    (tmp_path/'output').mkdir()
    with pytest.raises(ValueError):validate_request(r)
    r['output_directory']=str(tmp_path/'base'/'nested')
    with pytest.raises(ValueError):validate_request(r)


def test_context_cannot_be_silently_truncated():
    class Tokenizer:
        def apply_chat_template(self,*_a,**_k):return 'prefix'
        def encode(self,*_a,**_k):return [1]*1025
    with pytest.raises(ValueError):encode_response(Tokenizer(),{'question':'public','draft':'draft','draft_output_tokens':10})
