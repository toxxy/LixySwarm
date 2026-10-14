from pathlib import Path

import pytest
import torch
from torch import nn

from src.learning.language_adapter import (AdapterSpec,adapter_parameters,install_adapter,load_adapter,save_adapter,adapters_enabled)
from src.learning.public_conversation import corpus_digest,public_corpus


class Expert(nn.Module):
    def __init__(self):
        super().__init__();self.q_proj=nn.Linear(4,6);self.v_proj=nn.Linear(6,4)

    def forward(self,x):return self.v_proj(torch.tanh(self.q_proj(x)))


def test_adapter_starts_as_the_base_and_training_changes_only_the_update(tmp_path):
    torch.manual_seed(4);model=Expert();x=torch.randn(8,4);expected=model(x).detach()
    original={name:p.detach().clone() for name,p in model.named_parameters()}
    parameters=install_adapter(model,AdapterSpec(rank=2,alpha=4))
    assert torch.equal(model(x).detach(),expected)
    optimizer=torch.optim.AdamW(parameters.values(),lr=0.02)
    for _ in range(5):
        optimizer.zero_grad();model(x).square().mean().backward();optimizer.step()
    assert not torch.equal(model(x).detach(),expected)
    with adapters_enabled(model,False):assert torch.equal(model(x).detach(),expected)
    assert not torch.equal(model(x).detach(),expected)
    for name,value in original.items():
        module,field=name.rsplit(".",1)
        assert torch.equal(getattr(model.get_submodule(module).base,field),value)
    path=tmp_path/"delta.safetensors";save_adapter(model,path,"a"*64,AdapterSpec(rank=2,alpha=4))
    torch.manual_seed(4);replica=Expert();install_adapter(replica,AdapterSpec(rank=2,alpha=4))
    load_adapter(replica,path,"a"*64,AdapterSpec(rank=2,alpha=4))
    assert torch.equal(model(x),replica(x))
    with pytest.raises(ValueError,match="incompatible"):
        load_adapter(replica,path,"b"*64,AdapterSpec(rank=2,alpha=4))


def test_invalid_adapter_values_are_not_exported_and_base_parameters_cannot_train(tmp_path):
    model=Expert();parameters=install_adapter(model)
    with torch.no_grad():next(iter(parameters.values())).fill_(float("nan"))
    with pytest.raises(ValueError,match="nonfinite"):save_adapter(model,tmp_path/"bad","a"*64)
    model.q_proj.base.weight.requires_grad_(True)
    with pytest.raises(ValueError,match="Only adapter"):adapter_parameters(model)


def test_public_corpus_is_reproducible_and_has_no_overlapping_gate_or_eval_prompts():
    from experiments.evaluate_conversation_seed import CASES
    corpus=public_corpus()
    assert corpus_digest(corpus)==corpus_digest(public_corpus())
    train={row["prompt"] for row in corpus["train"]}
    assert len(train)==len(corpus["train"])
    assert not train&{row["prompt"] for row in corpus["evaluation"]}
    assert not train&{case[1] for case in CASES}
    assert corpus_digest(public_corpus("lixyswarm-public-instructions-v1"))=="5d37d9c79de8cebeea0d4698f0d24c4f734ad7e3feb9745815c060b0484934c6"
