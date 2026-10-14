from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.learning.language_adapter import install_adapter,adapter_parameters
from src.learning.language_gradient import save_gradient,load_gradient,combine_gradients


def model():
    value=nn.Module();value.q_proj=nn.Linear(3,4);value.v_proj=nn.Linear(3,4);install_adapter(value)
    return value


def contribution(value,path,start,magnitude):
    for p in adapter_parameters(value).values():p.grad=torch.full_like(p,magnitude)
    args={'base_hash':'a'*64,'adapter_id':'b'*64,'dataset_id':'c'*64,'start':start,'count':1,'tokens':4 if start==0 else 8}
    save_gradient(value,path,**args)
    return {'path':path,**args}


def test_compatible_disjoint_language_gradients_are_token_weighted(tmp_path):
    value=model();a=contribution(value,tmp_path/'a.safetensors',0,.01);b=contribution(value,tmp_path/'b.safetensors',1,.02)
    result=combine_gradients(value,[a,b])
    assert all(torch.allclose(t,torch.full_like(t,.05/3)) for t in result.values())


def test_version_overlap_and_excessive_gradients_are_rejected_before_candidate_update(tmp_path):
    value=model();a=contribution(value,tmp_path/'a.safetensors',0,.01);b=contribution(value,tmp_path/'b.safetensors',0,.02)
    with pytest.raises(ValueError,match='overlap'):combine_gradients(value,[a,b])
    with pytest.raises(ValueError,match='version'):load_gradient(value,a['path'],**{**{k:a[k] for k in ['base_hash','adapter_id','dataset_id','start','count','tokens']},'adapter_id':'d'*64})
    c=contribution(value,tmp_path/'c.safetensors',2,10)
    with pytest.raises(ValueError,match='clipping'):load_gradient(value,c['path'],**{k:c[k] for k in ['base_hash','adapter_id','dataset_id','start','count','tokens']})
