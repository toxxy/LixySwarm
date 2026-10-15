from copy import deepcopy

import pytest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM

from src.learning.language_adapter import install_adapter
from src.learning.language_specialist import fit_specialist,FrozenAdapterBank,RoutedAdapterExperts
from src.learning.language_specialist_protocol import definition
from src.learning.language_selector import LanguageExpertSelector,tensor_digest,save_selector,load_selector


def tiny():
    torch.manual_seed(91)
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=64,
        bos_token_id=1,eos_token_id=31,pad_token_id=0)).eval()
    return model,install_adapter(model)


def test_fixed_epoch_covers_math_once_replays_public_and_preserves_base():
    model,params=tiny();before={n:p.detach().clone() for n,p in model.named_parameters()}
    p={**definition(),'specialist_training_examples':8,'specialist_batch_size':4,'specialist_replay_examples':2}
    trace,optimizer=fit_specialist(model,params,[([1,2],[3+i,31]) for i in range(8)],
        [([1,3],[5,31]),([1,4],[6,31]),([1,5],[7,31])],p,device='cpu')
    assert trace['steps']==2 and sorted(trace['math_order'])==list(range(8))
    assert [r['replay_indices'] for r in trace['steps_trace']]==[[0,1],[2,0]]
    assert all(int(v['step'])==2 for v in optimizer['state'].values())
    assert any(not torch.equal(v,before[n]) for n,v in params.items())
    assert all(torch.equal(v,before[n]) for n,v in model.named_parameters() if n not in params)


def test_training_cancellation_restores_parent_after_an_update():
    model,params=tiny();before={n:p.detach().clone() for n,p in params.items()}
    p={**definition(),'specialist_training_examples':8,'specialist_batch_size':4,'specialist_replay_examples':2}
    def progress(*args):raise RuntimeError('cancel after first step')
    with pytest.raises(RuntimeError,match='cancel after'):
        fit_specialist(model,params,[([1,2],[3,31])]*8,[([1,3],[4,31])]*3,p,device='cpu',progress=progress)
    assert all(torch.equal(v,before[n]) for n,v in params.items())


def test_transactional_adapter_selection_and_nested_failure_restore_every_tensor():
    model,params=tiny();parent={n:p.detach().clone() for n,p in params.items()}
    specialist={n:t+.01 for n,t in parent.items()}
    for p in params.values():p.requires_grad_(False)
    bank=FrozenAdapterBank(model,params,[parent,specialist]);seen=[]
    class Selector:
        def choose(self,queries,**kwargs):seen.extend(queries);return torch.tensor([1]),torch.tensor([[0.,1.]])
        def eval(self):return self
    inputs=torch.tensor([[1,2,3,4]]);labels=torch.tensor([[-100,-100,3,4]])
    with torch.inference_mode():
        with bank.activate(1):expected=model(input_ids=inputs,labels=labels).loss
        actual=RoutedAdapterExperts(bank,Selector())(input_ids=inputs,labels=labels).loss
    assert seen==[[1,2]] and torch.equal(actual,expected)
    with pytest.raises(RuntimeError,match='nested failure'):
        with bank.activate(1):
            with bank.activate(0):
                assert all(torch.equal(v,parent[n]) for n,v in params.items())
            assert all(torch.equal(v,specialist[n]) for n,v in params.items())
            raise RuntimeError('nested failure')
    assert all(torch.equal(v,parent[n]) for n,v in params.items())
    corrupt=deepcopy(specialist);next(iter(corrupt.values())).fill_(float('nan'))
    with pytest.raises(ValueError,match='mismatch'):FrozenAdapterBank(model,params,[parent,corrupt])
    assert all(torch.equal(v,parent[n]) for n,v in params.items())


def test_selector_artifact_binds_both_experts_in_order(tmp_path):
    model=LanguageExpertSelector(32,support_count=2).eval()
    metadata=dict(protocol_id='a'*64,base_hash='b'*64,tokenizer_id='c'*64,
        dolphin_state_id=tensor_digest(model.dolphin.state_dict()),feature_space='lixyswarm-public-language-adapter-selection-v2',
        memory_scope='public-training-only',variant='public',expert_ids=['d'*64,'e'*64])
    path=tmp_path/'bound.safetensors';save_selector(model,path,metadata)
    target=LanguageExpertSelector(32,support_count=2).eval();load_selector(target,path,metadata)
    before=tensor_digest(target.state_dict())
    with pytest.raises(ValueError,match='space'):
        load_selector(target,path,{**metadata,'expert_ids':list(reversed(metadata['expert_ids']))})
    assert tensor_digest(target.state_dict())==before
