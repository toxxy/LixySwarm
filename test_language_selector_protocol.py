"""Prospective public input bindings without importing tensor runtimes."""
import time
from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.learning.language_selector_protocol import definition,protocol_id
from src.learning.language_selector_job import validate_request
from src.learning.language_retention import PARENT_MODEL,PARENT_RELEASE


def request(tmp_path):
    return dict(action=definition()['action'],protocol_id=protocol_id(),compute_dtype='float32',streamed_base=True,
        base_directory=str(tmp_path/'base'),adapter_file=str(tmp_path/'adapter.safetensors'),model_id=PARENT_MODEL,
        parent_release_id=PARENT_RELEASE,training_file=str(tmp_path/'train'),evaluation_file=str(tmp_path/'eval'),
        output_directory=str(tmp_path/'new'),device='cuda',cpu_threads=2,gpu_fraction=.5,deadline=time.time()+60)


def test_protocol_is_train_only_same_budget_and_not_native_admission(tmp_path):
    value=definition();validate_request(request(tmp_path))
    assert value['historical_training_examples']==200 and value['math_training_examples']==192
    assert value['phase_steps']==256 and value['memory_modes']==['zero','shuffled','public']
    assert not value['held_out_used_for_training_or_selection'] and not value['new_model_publication_authorized']
    assert value['public_scope']=='public-training-only' and 'no generation' in value['evaluation']


def test_prospective_fixture_exactly_binds_code_and_inputs():
    value=json.loads((Path(__file__).parent/'experiments/reports/language_selector_protocol_20261015.json').read_text())
    assert value=={'protocol_id':protocol_id(),'definition':definition()}
    assert protocol_id()=='df6b22c1b9d7a17cb5bd037c39e5f2b453a94c4f131824b402374feae308f8f0'


@pytest.mark.parametrize('key,value',[('model_id','a'*64),('compute_dtype','bfloat16'),('streamed_base',False),
    ('cpu_threads',True),('gpu_fraction',float('nan')),('training_file','relative'),('private_memory','secret')])
def test_requests_cannot_substitute_scope_parent_precision_or_bounds(tmp_path,key,value):
    changed=deepcopy(request(tmp_path));changed[key]=value
    with pytest.raises(ValueError):validate_request(changed)
