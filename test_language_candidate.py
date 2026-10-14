from copy import deepcopy
import time

import pytest
import torch
from torch import nn

from src.learning.language_adapter import install_adapter,adapter_parameters
from src.learning.language_gradient import save_gradient
from src.learning.language_candidate_job import validate_request,apply_update
from src.learning.public_bases import base_hash
from src.learning.public_conversation import public_corpus,corpus_digest


def request(tmp_path):
    return {'action':'candidate','base_name':'qwen3-1.7b','base_directory':str(tmp_path/'base'),
        'adapter_file':str(tmp_path/'parent.safetensors'),'adapter_id':'a'*64,'dataset_id':corpus_digest(public_corpus()),
        'parent_release_id':'b'*64,'learning_rate':.001,'output_file':str(tmp_path/'candidate.safetensors'),
        'cpu_threads':2,'gpu_fraction':.5,'device':'cuda','deadline':time.time()+60,
        'contributions':[{'path':str(tmp_path/'gradient.safetensors'),'gradient_id':'c'*64,'gradient_bytes':100,
            'base_hash':base_hash('qwen3-1.7b'),'adapter_id':'a'*64,'dataset_id':corpus_digest(public_corpus()),
            'start':0,'count':4,'tokens':16}]}


@pytest.mark.parametrize('failure',['stale','foreign_data','overlap','duplicate','range','personal_input','nan','replace_parent'])
def test_candidate_rejects_invalid_data_before_loading_models(tmp_path,failure):
    value=request(tmp_path)
    assert validate_request(value)['version']==public_corpus()['version']
    if failure=='stale':value['contributions'][0]['adapter_id']='d'*64
    elif failure=='foreign_data':value['dataset_id']='d'*64
    elif failure=='overlap':value['contributions'].append({**value['contributions'][0],'gradient_id':'d'*64,'start':2})
    elif failure=='duplicate':value['contributions'].append({**value['contributions'][0],'start':4})
    elif failure=='range':value['contributions'][0]['count']=9
    elif failure=='personal_input':value['personal_path']=str(tmp_path/'personal')
    elif failure=='nan':value['learning_rate']=float('nan')
    else:
        parent=tmp_path/'parent.safetensors';parent.write_bytes(b'preserve parent');value['output_file']=str(parent)
    with pytest.raises(ValueError):validate_request(value)


def test_candidate_step_uses_token_weights_and_preserves_all_base_parameters(tmp_path):
    model=nn.Module();model.q_proj=nn.Linear(3,4);model.v_proj=nn.Linear(3,4)
    parameters=install_adapter(model);before={name:p.detach().clone() for name,p in model.named_parameters()}
    contributions=[]
    for index,(magnitude,tokens) in enumerate([(.01,4),(.02,8)]):
        for parameter in parameters.values():parameter.grad=torch.full_like(parameter,magnitude)
        fields={'base_hash':'a'*64,'adapter_id':'b'*64,'dataset_id':'c'*64,'start':index,'count':1,'tokens':tokens}
        path=tmp_path/f'{index}.safetensors';save_gradient(model,path,**fields)
        contributions.append({'path':str(path),**fields})
    apply_update(model,contributions,.001)
    for name,parameter in model.named_parameters():
        expected=before[name]-.001*(.01*4+.02*8)/12 if parameter.requires_grad else before[name]
        assert torch.equal(parameter,expected) if not parameter.requires_grad else torch.allclose(parameter,expected)


def test_incompatible_later_gradient_cannot_partially_modify_adapter(tmp_path):
    model=nn.Module();model.q_proj=nn.Linear(3,4);model.v_proj=nn.Linear(3,4)
    parameters=install_adapter(model)
    for parameter in parameters.values():parameter.grad=torch.full_like(parameter,.01)
    fields={'base_hash':'a'*64,'adapter_id':'b'*64,'dataset_id':'c'*64,'start':0,'count':1,'tokens':4}
    path=tmp_path/'gradient.safetensors';save_gradient(model,path,**fields)
    before={name:p.detach().clone() for name,p in model.named_parameters()}
    incompatible={**fields,'start':1,'path':str(path)}
    with pytest.raises(ValueError,match='does not match'):
        apply_update(model,[{'path':str(path),**fields},incompatible],.001)
    assert all(torch.equal(before[name],p) for name,p in model.named_parameters())
