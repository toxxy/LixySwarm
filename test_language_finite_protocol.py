"""Public training-only scope; fixed schedule and a lightweight supervisor."""
from copy import deepcopy
import subprocess
import sys

import pytest

from src.learning.language_finite_protocol import definition,protocol_id,KINDS
from src.learning.language_finite_step_job import validate_request
from test_language_retention import request


def test_finite_protocol_fixed_before_training_and_has_no_publication_authority():
    value=definition()
    assert value['rates']==[.001/(2**i) for i in range(12)]
    assert set(value['protected'])=={'old_prefix64','old_full200',*['old_kind_'+kind for kind in KINDS]}
    assert value['target']=='math_train192' and value['math_range']==[0,192]
    assert value['model_compute_dtype']=='bfloat16' and value['ram_gb']==6
    assert not value['held_out_data_used_in_compute'] and not value['new_model_publication_authorized']
    assert definition('float32')['ram_gb']==10 and protocol_id('float32')!=protocol_id('bfloat16')
    subprocess.run([sys.executable,'-c',"import sys; import experiments.diagnose_language_retention; import src.learning.language_finite_step_job; assert 'torch' not in sys.modules"],check=True)


def test_group_protocol_is_distinct_and_preserves_original_finite_policy():
    assert protocol_id('bfloat16')=='5d2639b43923dee48e2d44282261c68eb1b5b79701be990da5bed190afba0246'
    assert protocol_id('float32')=='5a4365e24a26c8207dcb1599019bd1b80e8a5f04081ea125ba08ce51fff4dafa'
    projected=definition(group_direction=True)
    assert projected['version']==2 and projected['maximum_projection_sweeps']==512
    assert projected['rates']==definition()['rates'] and projected['protected']==definition()['protected']
    assert protocol_id(group_direction=True)!=protocol_id()


@pytest.mark.parametrize('change',['heldout','schedule','action','protocol','precision','direction'])
def test_finite_request_cannot_change_pinned_scope(tmp_path,change):
    value=request(tmp_path);value.update(action='public-finite-step-v1',protocol_id=protocol_id('float32'))
    validate_request(value)
    value=deepcopy(value)
    if change=='heldout':value['evaluation_file']=str(tmp_path/'heldout')
    elif change=='schedule':value['rates']=[.1]
    elif change=='action':value['action']='train'
    elif change=='protocol':value['protocol_id']='0'*64
    elif change=='precision':value['compute_dtype']='bfloat16'
    else:value['contributions'][0]['count']=63
    with pytest.raises(ValueError):validate_request(value)


def test_group_request_requires_its_own_protocol(tmp_path):
    value=request(tmp_path);value.update(action='public-group-finite-step-v2',protocol_id=protocol_id('float32',group_direction=True))
    validate_request(value)
    value['protocol_id']=protocol_id('float32')
    with pytest.raises(ValueError):validate_request(value)
