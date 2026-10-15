"""Precommitted format routing must not use correctness or invent saved cost."""
import copy
import hashlib
import json
import time

import pytest

from src.learning import language_repair_comparison as protocol
from src.learning.language_repair_comparison_job import collect, validate_request


def rows():
    return [{'case':hashlib.sha256(str(i).encode()).hexdigest(),'question':'question '+str(i),
        'answer':'reference-only-canary #### 0','final_number':'0'} for i in range(128)]


def result(answer):
    return {'answer':answer,'input_tokens':30,'output_tokens':20,'generation_seconds':1.}


def report(records):
    return {'protocol_id':protocol.protocol_id(),'snapshot_id':protocol.SNAPSHOT_ID,'records':records,
        'training_performed':False,'model_promoted':False}


def test_component_must_beat_same_parent_repair_and_counts_only_its_own_calls(tmp_path):
    output=tmp_path/'run';output.mkdir();calls=[];completed=[]
    def generate(messages,expert):
        assert (output/'prospective-commitment.json').exists()
        assert 'reference-only-canary' not in json.dumps(messages)
        index=int(messages[0]['content'].split()[-1]);kind='repair' if len(messages)==3 else 'specialist' if expert else 'parent'
        if kind!='parent':
            decision=json.loads((output/f'decision-{index:03}.json').read_text())
            assert decision['second_requested'] and decision['case']==rows()[index]['case']
        calls.append((index,kind))
        return result('#### 0' if kind=='repair' or kind=='parent' and index%2==0 else '#### 1' if expert else 'unfinished work')
    records=collect(rows(),generate,output,lambda d,t:completed.append(d))
    verification=protocol.verify_report(report(records),{'rows':rows()})
    assert verification['policies']['parent']['correct']==64
    assert verification['policies']['parent-repair']['correct']==128
    assert verification['policies']['specialist-fallback']['correct']==64
    assert verification['actual_fresh_generation_calls']==len(calls)==256
    assert verification['policies']['specialist-fallback']['calls']==192
    assert not verification['component_passed'] and completed[-1]==128
    assert [kind for i,kind in calls if i==1]==['parent','specialist','repair']


def test_successful_component_is_still_read_only_and_not_a_neural_or_global_gate(tmp_path):
    output=tmp_path/'run';output.mkdir()
    def generate(messages,expert):
        return result('#### 0' if expert else '#### 1' if len(messages)==3 else 'unfinished work')
    records=collect(rows(),generate,output,lambda *_:None)
    verification=protocol.verify_report(report(records),{'rows':rows()})
    assert verification['component_passed'] and verification['policies']['specialist-fallback']['new_correct']==128
    assert verification['policies']['specialist-fallback']['generation_seconds']==256
    assert verification['actual_fresh_generation_calls']==384
    assert not verification['model_promoted'] and not verification['neural_routing_evaluated']
    assert protocol.definition()['native_publication_authorized'] is False


def evidence():
    return [{'case':row['case'],'parent':result('#### 0'),'second_requested':False,
        'second_order':['repair','specialist'] if i%2==0 else ['specialist','repair'],'responses':{}}
        for i,row in enumerate(rows())]


@pytest.mark.parametrize('change', ['target_route','order','tokens','nonfinite','incomplete','promotion'])
def test_forged_evidence_cannot_create_a_better_score_or_budget(change):
    r=report(evidence())
    if change=='target_route':r['records'][0]['second_requested']=True
    elif change=='order':r['records'][0]['second_order'].reverse()
    elif change=='tokens':r['records'][0]['parent']['output_tokens']=385
    elif change=='nonfinite':r['records'][0]['parent']['generation_seconds']=float('nan')
    elif change=='incomplete':r['records'].pop()
    else:r['model_promoted']=True
    with pytest.raises(ValueError):protocol.verify_report(r,{'rows':rows()})


@pytest.mark.parametrize('answer,expected',[('#### 12',False),('wrong explanation\n#### -2.0',False),
    ('12',True),('#### 12\n#### 12',True),('#### 12\nextra explanation',True)])
def test_route_depends_on_format_even_when_a_valid_number_is_incorrect(answer,expected):
    assert protocol.needs_second_call(answer) is expected


def request(tmp_path):
    return {'action':protocol.definition()['action'],'protocol_id':protocol.protocol_id(),
        'compute_dtype':'float32','streamed_base':True,'model_id':protocol.PARENT_MODEL,
        'parent_release_id':protocol.PARENT_RELEASE,'base_directory':str(tmp_path/'base'),
        'adapter_file':str(tmp_path/'parent'),'specialist_file':str(tmp_path/'specialist'),
        'snapshot_file':str(tmp_path/'snapshot'),'output_directory':str(tmp_path/'output'),
        'device':'cpu','cpu_threads':2,'gpu_fraction':1.,'deadline':time.time()+100}


@pytest.mark.parametrize('field,value',[('gpu_fraction',True),('gpu_fraction',float('nan')),
    ('cpu_threads',0),('deadline',float('inf')),('compute_dtype','bfloat16'),('model_id','0'*64)])
def test_numerical_request_bounds_cannot_change_the_experiment(tmp_path,field,value):
    r=request(tmp_path);r[field]=value
    with pytest.raises(ValueError):validate_request(r)


def test_request_rejects_reused_or_overlapping_evidence_and_unpublished_fields(tmp_path):
    r=request(tmp_path);validate_request(r)
    (tmp_path/'output').mkdir()
    with pytest.raises(ValueError):validate_request(r)
    r['output_directory']=str(tmp_path/'base'/'overlap')
    with pytest.raises(ValueError):validate_request(r)
    r=request(tmp_path);r['reference_answer']='must never be an engine request'
    with pytest.raises(ValueError):validate_request(r)
