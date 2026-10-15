"""Actual typed float16 gradient serialization and adapter-only updates."""
import pytest
torch=pytest.importorskip('torch')
pytest.importorskip('safetensors')
from safetensors import safe_open
from safetensors.torch import load_file,save_file
from src.learning.language_adapter import install_adapter,adapter_parameters
from src.learning.language_gradient import save_gradient,load_gradient,combine_gradients
from src.learning.language_candidate_job import apply_update


def fixture():
    torch.manual_seed(24)
    model=torch.nn.Module();model.q_proj=torch.nn.Linear(7,13,bias=False);model.v_proj=torch.nn.Linear(7,13,bias=False)
    install_adapter(model)
    x=torch.randn(8,7);loss=(model.q_proj(x)+model.v_proj(x)).square().mean();loss.backward()
    torch.nn.utils.clip_grad_norm_(adapter_parameters(model).values(),1)
    fields={'base_hash':'a'*64,'adapter_id':'b'*64,'dataset_id':'c'*64,'start':0,'count':8,'tokens':16}
    return model,fields


def test_actual_half_file_preserves_clipping_metadata_and_adapter_only_update(tmp_path):
    model,fields=fixture();before={k:p.detach().clone() for k,p in model.named_parameters()}
    source={k:p.grad.clone() for k,p in adapter_parameters(model).items()}
    original=tmp_path/'fp32';half=tmp_path/'half'
    save_gradient(model,original,**fields);precision=save_gradient(model,half,**fields,policy_version=3)
    decoded=load_gradient(model,half,**fields,policy_version=3,**precision)
    assert half.stat().st_size<original.stat().st_size
    assert all(t.dtype==torch.float16 for t in load_file(half).values())
    assert all(t.dtype==torch.float32 for t in decoded.values())
    assert sum(float(t.double().square().sum()) for t in decoded.values())<=1.00002
    assert precision['quantization_relative_l2_error']<=.001
    assert all(torch.equal(source[k],p.grad) for k,p in adapter_parameters(model).items())
    with pytest.raises(ValueError):load_gradient(model,half,**fields)
    with pytest.raises(ValueError):load_gradient(model,original,**fields,policy_version=3,**precision)
    apply_update(model,[{'path':str(half),**fields,**precision}],.001,policy_version=3)
    assert any(not torch.equal(before[k],p) for k,p in model.named_parameters() if p.requires_grad)
    assert all(torch.equal(before[k],p) for k,p in model.named_parameters() if not p.requires_grad)


@pytest.mark.parametrize('problem',['nan','type','norm','header'])
def test_invalid_half_content_is_rejected_before_applying_parameters(tmp_path,problem):
    model,fields=fixture();path=tmp_path/'half';precision=save_gradient(model,path,**fields,policy_version=3)
    with safe_open(path,framework='pt') as opened:header=opened.metadata()
    tensors=load_file(path)
    if problem=='nan':next(iter(tensors.values())).flatten()[0]=float('nan')
    elif problem=='type':tensors={k:v.float() for k,v in tensors.items()}
    elif problem=='norm':tensors={k:torch.ones_like(v) for k,v in tensors.items()}
    else:header['training_policy_id']='f'*64
    save_file(tensors,path,metadata=header)
    with pytest.raises(ValueError):load_gradient(model,path,**fields,policy_version=3,**precision)


def test_half_contributions_are_weighted_by_tokens_and_overlap_is_rejected(tmp_path):
    model,fields=fixture();rows=[];decoded=[]
    for index,tokens in enumerate([4,12]):
        for p in adapter_parameters(model).values():p.grad.mul_(.75)
        f={**fields,'start':index*4,'count':4,'tokens':tokens};path=tmp_path/str(index)
        precision=save_gradient(model,path,**f,policy_version=3)
        rows.append({'path':str(path),**f,**precision});decoded.append(load_gradient(model,path,**f,policy_version=3,**precision))
    actual=combine_gradients(model,rows,policy_version=3)
    for name,t in actual.items():assert torch.allclose(t,decoded[0][name]*.25+decoded[1][name]*.75,atol=1e-8)
    rows[1]['start']=0
    with pytest.raises(ValueError,match='overlap'):combine_gradients(model,rows,policy_version=3)


def test_underflow_cannot_silently_erase_an_entire_gradient(tmp_path):
    model,fields=fixture()
    for p in adapter_parameters(model).values():p.grad.fill_(1e-10)
    path=tmp_path/'underflow'
    with pytest.raises(ValueError,match='reconstruction error'):save_gradient(model,path,**fields,policy_version=3)
    assert not path.exists()
