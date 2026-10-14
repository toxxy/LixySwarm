"""Small, weights-only updates to an immutable pretrained language expert.

Imported only by tensor engines, never by the lightweight node or app. Each
adapter is tied to the exact public base and carries no executable code.
"""
from __future__ import annotations
import math
import contextlib
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from safetensors import safe_open
from safetensors.torch import load_file, save_file

MAX_ADAPTER_BYTES=32*1024**2


@dataclass(frozen=True)
class AdapterSpec:
    rank:int=8
    alpha:int=16
    targets:tuple[str,...]=("q_proj","v_proj")

    def validate(self):
        if type(self.rank) is not int or not 1<=self.rank<=64 or type(self.alpha) is not int or not 1<=self.alpha<=128 or self.targets!=("q_proj","v_proj"):
            raise ValueError("Unsupported bounded language adapter")


class AdapterLinear(nn.Module):
    def __init__(self,base:nn.Linear,spec:AdapterSpec):
        super().__init__();self.base=base;self.scale=spec.alpha/spec.rank;self.enabled=True
        self.lora_a=nn.Parameter(torch.empty(spec.rank,base.in_features,device=base.weight.device,dtype=torch.float32))
        self.lora_b=nn.Parameter(torch.zeros(base.out_features,spec.rank,device=base.weight.device,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a,a=math.sqrt(5))

    def forward(self,x):
        original=self.base(x)
        if not self.enabled:return original
        delta=(x.float()@self.lora_a.t())@self.lora_b.t()
        return original+(delta*self.scale).to(original.dtype)


@contextlib.contextmanager
def adapters_enabled(model,enabled):
    modules=[module for module in model.modules() if isinstance(module,AdapterLinear)]
    previous=[module.enabled for module in modules]
    try:
        for module in modules:module.enabled=enabled
        yield
    finally:
        for module,value in zip(modules,previous):module.enabled=value


def install_adapter(model,spec=AdapterSpec()):
    spec.validate()
    for parameter in model.parameters():parameter.requires_grad_(False)
    targets=[(name,module) for name,module in model.named_modules()
             if isinstance(module,nn.Linear) and name.rsplit(".",1)[-1] in spec.targets]
    if not 1<=len(targets)<=128:raise ValueError("Model has no bounded adapter targets")
    for name,module in targets:
        parent_name,_,attribute=name.rpartition(".")
        parent=model.get_submodule(parent_name) if parent_name else model
        setattr(parent,attribute,AdapterLinear(module,spec))
    return adapter_parameters(model)


def adapter_parameters(model):
    parameters={name:p for name,p in model.named_parameters() if p.requires_grad}
    if not parameters or any(not name.endswith((".lora_a",".lora_b","lora_a","lora_b")) for name in parameters):
        raise ValueError("Only adapter parameters may be trainable")
    return parameters


def save_adapter(model,path,base_model_hash,spec=AdapterSpec()):
    spec.validate()
    if len(base_model_hash)!=64 or any(c not in "0123456789abcdef" for c in base_model_hash):raise ValueError("Invalid public base hash")
    tensors={name:p.detach().float().cpu().contiguous() for name,p in adapter_parameters(model).items()}
    if any(not torch.isfinite(tensor).all() for tensor in tensors.values()):raise ValueError("Adapter has nonfinite parameters")
    if sum(t.numel()*t.element_size() for t in tensors.values())>MAX_ADAPTER_BYTES:raise ValueError("Adapter exceeds its bound")
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    save_file(tensors,path,metadata={"format":"lixyswarm-lora-v1","base_model_hash":base_model_hash,
              "rank":str(spec.rank),"alpha":str(spec.alpha),"targets":",".join(spec.targets)})


def load_adapter(model,path,base_model_hash,spec=AdapterSpec()):
    path=Path(path)
    if path.stat().st_size>MAX_ADAPTER_BYTES:raise ValueError("Adapter exceeds its bound")
    expected={"format":"lixyswarm-lora-v1","base_model_hash":base_model_hash,"rank":str(spec.rank),
              "alpha":str(spec.alpha),"targets":",".join(spec.targets)}
    with safe_open(path,framework="pt",device="cpu") as source:
        if source.metadata()!=expected:raise ValueError("Adapter is incompatible with the pinned base or architecture")
    parameters=adapter_parameters(model);tensors=load_file(path,device="cpu")
    if set(tensors)!=set(parameters):raise ValueError("Adapter tensor names do not match")
    for name,tensor in tensors.items():
        if tensor.shape!=parameters[name].shape or tensor.dtype!=torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError("Adapter tensor shapes, types or values are invalid")
    with torch.no_grad():
        for name,tensor in tensors.items():parameters[name].copy_(tensor.to(parameters[name].device))
