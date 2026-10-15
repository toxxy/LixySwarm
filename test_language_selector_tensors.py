"""Real learned routing, bounded public state and actual single-expert execution."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM

from src.agents.dolphin_agent import DolphinConfig,Echolocation
from src.learning.language_adapter import install_adapter,adapters_enabled,AdapterLinear
from src.learning.language_selector import (LanguageExpertSelector,train_selectors,pad_queries,
    tensor_digest,save_selector,load_selector,RoutedFrozenExperts)
from src.learning.language_selector_protocol import definition


def small_protocol():
    return {**definition(),'phase_steps':8,'batch_size':8}


@pytest.fixture
def trained():
    queries=[[1+(i%3),4,5+(i//3)%8] for i in range(24)]
    losses=torch.tensor([[.1,1.] if i%3==0 else [1.,.1] for i in range(24)])
    variants,report,optimizers=train_selectors(queries,losses,32,small_protocol())
    return queries,variants,report,optimizers


def test_padded_five_pings_match_individual_queries_and_receive_gradients():
    torch.manual_seed(51)
    model=Echolocation(DolphinConfig(vocab_size=32,n_embd=32,echo_dim=8,feromon_dim=8,dropout=0)).eval()
    queries=[[1,2,3],[4,5]]
    tokens,mask=pad_queries(queries,'cpu')
    together=model(tokens,mask)[0]
    individual=torch.cat([model(torch.tensor([q]))[0] for q in queries])
    assert torch.allclose(together,individual,atol=1e-6)
    together.square().sum().backward()
    for name in ['topic','intent','need','context','emotion']:
        ping=getattr(model,'ping_'+name)
        assert ping.embed.weight.grad.abs().sum()>0
        assert ping.embed.weight.grad[0].abs().sum()==0
    with pytest.raises(ValueError,match='at least one'):
        model(tokens,torch.zeros_like(mask))


def test_memory_variants_have_equal_budget_frozen_space_and_no_query_writes(trained):
    queries,variants,report,optimizers=trained
    assert set(variants)=={'zero','shuffled','public','dolphin'}
    assert set(report['support_indices']).isdisjoint(report['query_indices'])
    assert sorted(report['support_indices']+report['query_indices'])==list(range(len(queries)))
    assert {row['steps'] for row in report['phase2'].values()}=={8}
    for name,model in variants.items():
        assert tensor_digest(model.dolphin.state_dict())==report['dolphin_state_id']
        before=tensor_digest(model.state_dict())
        choices,probabilities=model.choose(queries,use_memory=name!='dolphin')
        assert choices.shape==(len(queries),) and torch.allclose(probabilities.sum(-1),torch.ones(len(queries)))
        assert tensor_digest(model.state_dict())==before
    public=variants['public'];shuffled=variants['shuffled']
    assert public.public_memory.shape==(6,16)
    assert torch.equal(public.public_memory[:,:8],shuffled.public_memory[:,:8])
    assert not torch.equal(public.public_memory[:,8:10],shuffled.public_memory[:,8:10])
    assert variants['zero'].public_memory.count_nonzero()==0
    assert tensor_digest(public.memory.state_dict())!=tensor_digest(variants['dolphin'].memory.state_dict())
    assert optimizers['dolphin']['state'] and all(s['state'] for s in optimizers['memory'].values())


def binding(model):
    return dict(protocol_id='a'*64,base_hash='b'*64,tokenizer_id='c'*64,
        dolphin_state_id=tensor_digest(model.dolphin.state_dict()),
        feature_space='lixyswarm-public-language-expert-echo-v1',memory_scope='public-training-only',variant='public')


def test_safe_state_reload_exact_and_rejects_foreign_space_before_copy(tmp_path,trained):
    queries,variants,report,optimizers=trained
    model=variants['public'];metadata=binding(model);path=tmp_path/'router.safetensors'
    save_selector(model,path,metadata)
    restored=LanguageExpertSelector(32,support_count=6).eval()
    load_selector(restored,path,metadata)
    assert tensor_digest(restored.state_dict())==tensor_digest(model.state_dict())
    assert all(torch.equal(a,b) for a,b in zip(restored.choose(queries),model.choose(queries)))
    before=tensor_digest(restored.state_dict())
    for changed in [{'tokenizer_id':'d'*64},{'memory_scope':'personal'},{'dolphin_state_id':'e'*64}]:
        with pytest.raises(ValueError,match='binding|scope|space'):
            load_selector(restored,path,{**metadata,**changed})
        assert tensor_digest(restored.state_dict())==before
    optimizer_path=tmp_path/'optimizer-state.pt';torch.save(optimizers,optimizer_path)
    assert torch.load(optimizer_path,weights_only=True)['dolphin']['state'].keys()==optimizers['dolphin']['state'].keys()


@pytest.mark.parametrize('expert',[0,1])
def test_qwen_executes_only_selected_expert_and_never_routes_on_reference_answer(expert):
    torch.manual_seed(52)
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=64,
        bos_token_id=1,eos_token_id=31,pad_token_id=0)).eval()
    parameters=install_adapter(model)
    with torch.no_grad():
        for name,p in parameters.items():
            if name.endswith('lora_b'):p.normal_(std=.03)
    before=tensor_digest(model.state_dict());prefixes=[]
    class FixedSelector:
        def choose(self,queries,**kwargs):
            prefixes.extend(queries);return torch.tensor([expert]),torch.tensor([[1.-expert,expert]])
        def eval(self):return self
    routed=RoutedFrozenExperts(model,FixedSelector()).eval()
    inputs=torch.tensor([[1,2,3,4,5]]);labels=torch.tensor([[-100,-100,-100,4,5]])
    with torch.inference_mode():
        with adapters_enabled(model,expert==0):expected=model(input_ids=inputs,labels=labels,use_cache=False).loss
        count=[];hook=model.register_forward_hook(lambda *_:count.append(1))
        actual=routed(input_ids=inputs,labels=labels,use_cache=False).loss
        hook.remove()
        assert len(count)==1 and torch.equal(actual,expected)
        assert prefixes==[[1,2,3]]
        generated=routed.generate(input_ids=inputs[:,:3],max_new_tokens=2,do_sample=False,use_cache=True)
        with adapters_enabled(model,expert==0):expected_generation=model.generate(input_ids=inputs[:,:3],max_new_tokens=2,do_sample=False,use_cache=True)
        assert torch.equal(generated,expected_generation)
    assert tensor_digest(model.state_dict())==before
    assert all(m.enabled for m in model.modules() if isinstance(m,AdapterLinear))
    with pytest.raises(ValueError,match='nonempty masked'):
        routed(input_ids=inputs,labels=torch.ones_like(inputs))


def test_failure_restores_adapter_execution_mode():
    module=AdapterLinear(torch.nn.Linear(3,3),__import__('src.learning.language_adapter',fromlist=['AdapterSpec']).AdapterSpec())
    class Broken(torch.nn.Module):
        def __init__(self):super().__init__();self.layer=module
        def forward(self,**kwargs):
            assert not self.layer.enabled
            raise RuntimeError('fixture failure')
    class Selector:
        def choose(self,*_,**__):return torch.tensor([1]),torch.tensor([[0.,1.]])
    with pytest.raises(RuntimeError,match='fixture failure'):
        RoutedFrozenExperts(Broken(),Selector())(input_ids=torch.tensor([[1,2]]))
    assert module.enabled


def test_invalid_teacher_inputs_cannot_start_training():
    with pytest.raises(ValueError,match='TRAIN expert losses'):
        train_selectors([[1]]*8,torch.full((8,2),float('nan')),32,small_protocol())
    with pytest.raises(ValueError,match='query prefixes'):
        pad_queries([[]],'cpu')
