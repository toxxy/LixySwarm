"""Actual replay projection, backward pass and policy-4 file boundaries on CPU."""
import time
import pytest
torch=pytest.importorskip('torch')
from src.learning.language_math_gradient import project_with_replay,backward
from src.learning.language_gradient import save_gradient,load_gradient
from src.learning.language_adapter import install_adapter,adapter_parameters
from src.learning.language_policy import dataset_id


def test_conflicting_replay_is_projected_before_clipping():
    new={'p':torch.tensor([-3.,2.])};replay={'p':torch.tensor([1.,0.])}
    value=project_with_replay(new,replay)
    assert torch.equal(value['p'],torch.tensor([1.,2.]))
    assert float((value['p']*replay['p']).sum())>=0
    assert torch.equal(new['p'],torch.tensor([-3.,2.]))
    assert torch.equal(project_with_replay(new,{'p':torch.zeros(2)})['p'],new['p'])
    assert torch.isfinite(project_with_replay(new,{'p':torch.tensor([1e-40,0.])})['p']).all()
    with pytest.raises(ValueError):project_with_replay(new,{'p':torch.ones(3)})
    with pytest.raises(ValueError):project_with_replay(new,{'p':torch.tensor([float('nan'),0.])})


def test_actual_common_training_replay_and_math_gradient_format(tmp_path):
    from transformers.models.qwen3 import Qwen3Config,Qwen3ForCausalLM
    torch.set_num_threads(1);torch.manual_seed(18)
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=1,head_dim=8)).train();install_adapter(model)
    before={name:p.detach().clone() for name,p in model.named_parameters()}
    class Tokenizer:
        eos_token_id=9
        def apply_chat_template(self,messages,**_kw):return 'PREFIX'+messages[0]['content']
        def encode(self,text,**_kw):return [1,2,3] if text.startswith('PREFIX') else [5,6]
    loss,replay_loss,tokens,replay_tokens=backward(model,Tokenizer(),[{'prompt':'Public math fixture','answer':'#### 1'}]*2,'cpu',time.time()+60)
    assert loss>0 and replay_loss>0 and tokens==6 and replay_tokens==64*3
    parameters=adapter_parameters(model)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters.values())
    assert sum(float(p.grad.double().square().sum()) for p in parameters.values())<=1.00001
    assert all(torch.equal(before[name],p) for name,p in model.named_parameters())
    fields={'base_hash':'a'*64,'adapter_id':'b'*64,'dataset_id':dataset_id(4),'start':0,'count':2,'tokens':tokens}
    path=tmp_path/'math.safetensors';precision=save_gradient(model,path,**fields,policy_version=4)
    assert load_gradient(model,path,**fields,policy_version=4,**precision)
    with pytest.raises(ValueError):load_gradient(model,path,**fields,policy_version=3,**precision)
