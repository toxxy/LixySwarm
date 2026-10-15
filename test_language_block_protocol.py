"""Fixed public training method and unchanged held-out acceptance scope."""
import subprocess,sys,time
import pytest
from src.learning.language_block_protocol import definition,protocol_id
from src.learning.language_block_quality_job import validate_request
from src.learning.public_math import gate_id
from test_language_retention import request


def quality_request(tmp_path):
    value=request(tmp_path);value.update(action='public-block-quality-v1',protocol_id=protocol_id(),streamed_base=True,
        evaluation_file=str(tmp_path/'eval'))
    return value


def test_block_protocol_fixed_and_no_publication_or_native_cycle_authority():
    value=definition()
    assert value['maximum_steps']==4 and value['projection_relative_margin']==.01
    assert value['evaluation_gate_id']==gate_id()=='5e085b48da4be5c0e2b3176bb387466405005a0ec61c44b2372f521dbb8e4507'
    assert value['maximum_seconds']==3600 and value['ram_gb']==6
    assert value['held_out_data_used_in_compute'] and value['generation_performed']
    assert not value['new_model_publication_authorized'] and not value['model_files_written']
    subprocess.run([sys.executable,'-c',"import experiments.diagnose_language_retention,src.learning.language_block_quality_job,sys; assert 'torch' not in sys.modules"],check=True)


@pytest.mark.parametrize('change',['steps','margin','scope','precision','expired','late','relative_eval','parent'])
def test_quality_request_cannot_choose_hyperparameters_or_another_parent(tmp_path,change):
    value=quality_request(tmp_path);value['deadline']=time.time()+3500;validate_request(value)
    if change=='steps':value['maximum_steps']=1
    elif change=='margin':value['projection_relative_margin']=0.
    elif change=='scope':value['streamed_base']=False
    elif change=='precision':value['compute_dtype']='bfloat16'
    elif change=='expired':value['deadline']=time.time()-1
    elif change=='late':value['deadline']=time.time()+3601
    elif change=='relative_eval':value['evaluation_file']='evaluation.json'
    else:value['parent_release_id']='0'*64
    with pytest.raises(ValueError):validate_request(value)
