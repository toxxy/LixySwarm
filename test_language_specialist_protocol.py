import json
from pathlib import Path
import time

import pytest

from src.learning.language_specialist_protocol import definition,protocol_id,necessary_loss_gate
from src.learning.language_specialist_job import validate_request


def request(tmp_path):
    p=definition()
    return dict(action=p['action'],protocol_id=protocol_id(),compute_dtype='float32',streamed_base=True,
        base_directory=str(tmp_path/'base'),adapter_file=str(tmp_path/'parent'),model_id=p['parent_model_id'],
        parent_release_id=p['parent_release_id'],training_file=str(tmp_path/'train'),evaluation_file=str(tmp_path/'eval'),
        output_directory=str(tmp_path/'new'),device='cuda',cpu_threads=4,gpu_fraction=.5,deadline=time.time()+60)


def test_specialist_uses_unused_math_partition_and_unchanged_global_gates(tmp_path):
    p=definition();validate_request(request(tmp_path))
    assert p['specialist_training_start']==p['math_training_examples']==1024
    assert p['specialist_training_examples']==3072 and p['specialist_epochs']==1
    assert p['specialist_training_examples']//p['specialist_batch_size']==48
    assert p['primary_selector']=='public' and p['primary_selected_before_heldout']
    assert p['memory_feature_space']=='lixyswarm-public-language-adapter-selection-v2'
    assert p['new_remote_contributions']==0 and not p['native_publication_authorized']


def test_loss_screen_is_original_strict_necessary_condition_not_promotion():
    before={'math':{'nll':1.},'historical':{'nll':1.}}
    equal={'math':{'nll':.99},'historical':{'nll':.999}}
    assert not necessary_loss_gate(before,equal)['necessary_loss_gates_passed']
    better={'math':{'nll':.98},'historical':{'nll':.998}}
    verdict=necessary_loss_gate(before,better)
    assert verdict['necessary_loss_gates_passed'] and not verdict['native_publication_authorized']


@pytest.mark.parametrize('key,value',[('action','public-language-selector-v1'),('model_id','a'*64),
    ('compute_dtype','bfloat16'),('streamed_base',False),('cpu_threads',True),('training_file','relative'),('private_memory','secret')])
def test_specialist_request_cannot_substitute_inputs_or_scope(tmp_path,key,value):
    changed=request(tmp_path);changed[key]=value
    with pytest.raises(ValueError):validate_request(changed)


def test_specialist_protocol_matches_its_prospective_fixture():
    path=Path(__file__).parent/'experiments/reports/language_specialist_protocol_20261015.json'
    assert json.loads(path.read_text())=={'protocol_id':protocol_id(),'definition':definition()}
