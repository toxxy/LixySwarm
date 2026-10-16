"""Response-aware labels must preserve TRAIN provenance and generation bounds."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import pytest

from src.learning import language_repair_calibration as protocol
from src.learning.language_outcomes import protocol_id as prior_protocol_id
from src.learning.language_repair_calibration_job import collect,validate_request
from src.learning.public_math import canonical,final_number


@pytest.fixture
def inputs(monkeypatch):
    snapshot={'split':'train','rows':[{'case':hashlib.sha256(str(i).encode()).hexdigest(),
        'question':'public question '+str(i),'answer':'reference-only-canary #### 0','final_number':'0'}
        for i in range(128)]}
    values=[]
    for shard in (0,1):
        indices=list(range(shard,128,2));experts={}
        for name in ['accepted-parent','research-specialist']:
            answers=[{'case':snapshot['rows'][i]['case'],
                      'answer':('unfinished work' if i%4<2 else '#### 0') if name=='accepted-parent'
                        else '#### 0' if i%3==0 else '#### 1'} for i in indices]
            experts[name]={'answers':answers,'correct':sum(final_number(v['answer'])=='0' for v in answers),
                'output_tokens':[10]*64,'generation_seconds':[1.]*64,'nll':1.,'tokens':640,'total':64,'at_output_limit':0}
        values.append({'version':1,'protocol_id':prior_protocol_id(),
            'expert_ids':[protocol.PARENT_MODEL,protocol.SPECIALIST_ID],'shard':shard,
            'training_indices':indices,'experts':experts,'evaluation_data_used':False,'optimizer_updates':False})
    monkeypatch.setattr(protocol,'OUTCOME_IDS',tuple(hashlib.sha256(canonical(v)).hexdigest() for v in values))
    return snapshot,values


def collected(tmp_path,inputs,shard=0):
    snapshot,values=inputs;tmp_path.mkdir(exist_ok=True);calls=[]
    def generate(messages):
        frozen=json.loads((tmp_path/'prospective-commitment.json').read_text())
        assert len(frozen['cases'])==64 and frozen['protocol_id']==protocol.protocol_id()
        assert len(messages)==3 and messages[1]=={'role':'assistant','content':'unfinished work'}
        assert 'reference-only-canary' not in json.dumps(messages)
        calls.append(messages)
        index=int(messages[0]['content'].split()[-1])
        return {'answer':'#### 0' if index%8<4 else '#### 1',
                'output_tokens':5,'input_tokens':50,'generation_seconds':.5}
    value=collect(snapshot,values,shard,generate,tmp_path,lambda *_:None)
    return value,calls


@pytest.mark.parametrize('shard',[0,1])
def test_only_invalid_cached_parent_receives_one_new_call_with_no_reference(tmp_path,inputs,shard):
    value,calls=collected(tmp_path,inputs,shard)
    verified=protocol.verify_calibration(value,*inputs)
    assert len(calls)==verified['new_generation_calls']==32
    assert verified['parent_correct']==32 and verified['repair_fallback_correct']==48
    assert verified['new_repair_generation_seconds']==16 and verified['new_repair_output_tokens']==160
    assert not verified['neural_routing_evaluated'] and not verified['model_promoted']
    assert verified['public_training_only'] and 'not a latency comparison' in verified['cost_scope']
    assert [v['index'] for v in value['records']]==list(range(shard,128,2))


@pytest.mark.parametrize('change',['reference_route','valid_extra_call','missing','order','draft','nonfinite','token_bound','evaluation','optimizer'])
def test_forged_or_incomplete_labels_are_rejected(tmp_path,inputs,change):
    value,_=collected(tmp_path,inputs)
    if change=='reference_route':value['records'][0]['repair_requested']=False
    elif change=='valid_extra_call':value['records'][1]['repair']=deepcopy(value['records'][0]['repair'])
    elif change=='missing':value['records'].pop()
    elif change=='order':value['records'].reverse()
    elif change=='draft':value['records'][0]['parent_answer_id']='0'*64
    elif change=='nonfinite':value['records'][0]['repair']['generation_seconds']=float('nan')
    elif change=='token_bound':value['records'][0]['repair']['output_tokens']=385
    elif change=='evaluation':value['evaluation_data_used']=True
    else:value['optimizer_updates']=True
    with pytest.raises(ValueError):protocol.verify_calibration(value,*inputs)


def test_cached_data_is_bound_to_exact_bytes_shard_and_train_split(tmp_path,inputs):
    snapshot,values=inputs;paths=[]
    for shard,value in enumerate(values):
        path=tmp_path/f'cached-{shard}.json';path.write_bytes(canonical(value));paths.append(str(path))
    assert protocol.load_outcomes(paths,snapshot)==values
    with pytest.raises(ValueError):protocol.load_outcomes(paths[::-1],snapshot)
    changed=deepcopy(snapshot);changed['split']='evaluation'
    with pytest.raises(ValueError):protocol.load_outcomes(paths,changed)
    Path(paths[0]).write_bytes(Path(paths[0]).read_bytes()+b'\n')
    with pytest.raises(ValueError):protocol.load_outcomes(paths,snapshot)


def test_changing_prior_answers_or_supplying_evaluation_cannot_create_labels(tmp_path,inputs):
    value,_=collected(tmp_path,inputs);snapshot,values=deepcopy(inputs)
    values[0]['experts']['accepted-parent']['answers'][0]['answer']='#### 0'
    with pytest.raises(ValueError):protocol.verify_calibration(value,snapshot,values)
    with pytest.raises(ValueError):protocol.selected_rows(inputs[0],inputs[1],True)


def request(tmp_path):
    return {'action':protocol.definition()['action'],'protocol_id':protocol.protocol_id(),
        'compute_dtype':'float32','streamed_base':True,'model_id':protocol.PARENT_MODEL,
        'parent_release_id':protocol.PARENT_RELEASE,'base_directory':str(tmp_path/'base'),
        'adapter_file':str(tmp_path/'adapter'),'training_file':str(tmp_path/'train'),
        'outcome_files':[str(tmp_path/'prior-0'),str(tmp_path/'prior-1')],'shard':0,
        'output_directory':str(tmp_path/'output'),'device':'cpu','cpu_threads':2,
        'gpu_fraction':1.,'deadline':time.time()+100}


@pytest.mark.parametrize('field,value',[('evaluation_file','/never/evaluation'),('shard',True),
    ('gpu_fraction',True),('gpu_fraction',float('nan')),('cpu_threads',0),('deadline',float('inf')),
    ('compute_dtype','bfloat16'),('model_id','0'*64),('outcome_files',[])])
def test_request_cannot_inject_evaluation_or_change_numerical_scope(tmp_path,field,value):
    r=request(tmp_path);r[field]=value
    with pytest.raises(ValueError):validate_request(r)


def test_input_overlap_or_reused_evidence_is_rejected(tmp_path):
    r=request(tmp_path);validate_request(r)
    (tmp_path/'output').mkdir()
    with pytest.raises(ValueError):validate_request(r)
    r['output_directory']=str(tmp_path/'base'/'overlap')
    with pytest.raises(ValueError):validate_request(r)
