"""Scope and provenance of a retrospective training-only diagnostic."""
from copy import deepcopy
import subprocess
import sys
import time

import pytest

from src.learning import language_retention as protocol
from src.learning.language_retention_job import validate_request,training_sets
from src.learning.language_policy import dataset_id
from src.learning.public_bases import base_hash


def request(tmp_path):
    rows=[dict(zip(['gradient_id','gradient_bytes','start','count','tokens'],row),
        path=str(tmp_path/(str(row[2])+'.safetensors')),base_hash=base_hash('qwen3-1.7b'),
        adapter_id=protocol.PARENT_MODEL,dataset_id=dataset_id(4),quantization_relative_l2_error=.0002) for row in protocol.GRADIENTS]
    return {'action':'public-retention-diagnostic-v1','protocol_id':protocol.protocol_id(),
        'base_directory':str(tmp_path/'base'),'adapter_file':str(tmp_path/'adapter'),
        'model_id':protocol.PARENT_MODEL,'parent_release_id':protocol.PARENT_RELEASE,
        'training_file':str(tmp_path/'train'),'contributions':rows,'output_directory':str(tmp_path/'output'),
        'device':'cpu','cpu_threads':1,'gpu_fraction':.5,'deadline':time.time()+60}


def test_diagnostic_protocol_has_no_promotion_or_held_out_compute():
    definition=protocol.definition()
    assert definition['new_model_publication_authorized'] is False
    assert definition['held_out_data_used_in_compute'] is False and definition['learning_rate']==.001
    assert definition['math_range']==[0,192] and len(protocol.protocol_id())==64
    subprocess.run([sys.executable,'-c',"import sys; import experiments.diagnose_language_retention; assert 'torch' not in sys.modules"],check=True)


@pytest.mark.parametrize('change',['heldout','model','direction','bool_offset','precision','overwrite','expired'])
def test_request_cannot_substitute_scope_model_or_existing_output(tmp_path,change):
    value=request(tmp_path);validate_request(value)
    if change=='heldout':value['evaluation_file']=str(tmp_path/'heldout')
    elif change=='model':value['model_id']='a'*64
    elif change=='direction':value['contributions'][0]['gradient_id']='b'*64
    elif change=='bool_offset':value['contributions'][0]['start']=False
    elif change=='precision':value['contributions'][0]['quantization_relative_l2_error']=.01
    elif change=='overwrite':(tmp_path/'output').mkdir()
    else:value['deadline']=time.time()-1
    with pytest.raises(ValueError):validate_request(value)


def test_only_training_rows_enter_the_numeric_diagnostic(monkeypatch):
    from src.learning import public_conversation
    class NeverRead:
        def __getitem__(self,_key):raise AssertionError('Held-out data was accessed')
        def __iter__(self):raise AssertionError('Held-out data was accessed')
    old=[{'prompt':'Old public '+str(i),'answer':str(i)} for i in range(200)]
    monkeypatch.setattr(public_conversation,'public_corpus',lambda:{'train':old,'evaluation':NeverRead()})
    data={'rows':[{'question':'Math public '+str(i),'answer':str(i)} for i in range(4096)]}
    sets=training_sets(data)
    assert sets['old_prefix64']==old[:64] and sets['old_full200']==old
    assert len(sets['math_train192'])==192 and sets['math_train192'][-1]['answer']=='191'
