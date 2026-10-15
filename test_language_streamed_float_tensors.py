"""FP32 outputs/adapter gradients and BF16 storage checked on real tensors."""
import copy
import pytest
torch=pytest.importorskip('torch')
from src.learning.language_streamed_float import install_streamed_float32,FrozenFloatLinear
from src.learning.language_adapter import install_adapter,adapter_parameters


def test_frozen_backward_saves_original_bf16_weight_not_converted_copy():
    source=torch.nn.Linear(5,7,bias=True).bfloat16().requires_grad_(False)
    model=torch.nn.Sequential(source);install_streamed_float32(model)
    x=torch.randn(2,5,requires_grad=True);reference=x.detach().clone().requires_grad_();saved=[]
    def capture(tensor):saved.append((tensor.dtype,tensor.shape,tensor.data_ptr()));return tensor
    with torch.autograd.graph.saved_tensors_hooks(capture,lambda t:t):out=model(x)
    expected=torch.nn.functional.linear(reference,source.weight.float(),source.bias.float())
    assert torch.equal(out,expected) and saved==[(torch.bfloat16,source.weight.shape,source.weight.data_ptr())]
    out.sum().backward();expected.sum().backward()
    assert torch.equal(x.grad,reference.grad) and source.weight.grad is None


@pytest.mark.parametrize('checkpoint',[False,True])
def test_tiny_real_qwen_matches_full_fp32_logits_loss_and_adapter_gradients(checkpoint):
    from transformers import Qwen3Config,Qwen3ForCausalLM
    config=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=128,
        attention_dropout=0.,tie_word_embeddings=True)
    torch.manual_seed(7);stored=Qwen3ForCausalLM(config).bfloat16();full=copy.deepcopy(stored).float()
    torch.manual_seed(9);install_adapter(stored)
    torch.manual_seed(9);install_adapter(full)
    report=install_streamed_float32(stored)
    assert stored.model.embed_tokens.weight is stored.lm_head.weight
    assert set(stored.state_dict())==set(full.state_dict())
    assert report['stored_parameter_bytes']<sum(p.numel()*p.element_size() for p in full.parameters())*.7
    if checkpoint:
        for model in [stored,full]:model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    # Nonzero adapters exercise both A and B derivative paths and base backward.
    with torch.no_grad():
        for name,p in adapter_parameters(full).items():
            if name.endswith('lora_b'):p.uniform_(-.02,.02)
        for name,p in adapter_parameters(stored).items():p.copy_(adapter_parameters(full)[name])
    tokens=torch.randint(0,64,(2,12));labels=tokens.clone();labels[:,:3]=-100
    actual=stored(input_ids=tokens,labels=labels,use_cache=False)
    expected=full(input_ids=tokens,labels=labels,use_cache=False)
    torch.testing.assert_close(actual.logits,expected.logits,rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(actual.loss,expected.loss,rtol=1e-6,atol=1e-7)
    actual.loss.backward();expected.loss.backward()
    for name,p in adapter_parameters(stored).items():
        torch.testing.assert_close(p.grad,adapter_parameters(full)[name].grad,rtol=2e-5,atol=1e-7)
    assert all(p.dtype==torch.bfloat16 for module in stored.modules() if isinstance(module,FrozenFloatLinear) for p in module.parameters())


def test_streaming_rejects_trainable_base_and_unsupported_autocast():
    model=torch.nn.Sequential(torch.nn.Linear(3,4).bfloat16())
    with pytest.raises(ValueError):install_streamed_float32(model)
    assert isinstance(model[0],torch.nn.Linear)
    model.requires_grad_(False);install_streamed_float32(model)
    with pytest.raises(ValueError):install_streamed_float32(model)
    with pytest.raises(ValueError):model(torch.ones(1,3,dtype=torch.bfloat16))
    with torch.autocast('cpu',dtype=torch.bfloat16):
        with pytest.raises(ValueError):model(torch.ones(1,3))
