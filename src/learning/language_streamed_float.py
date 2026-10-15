"""FP32 computation from frozen BF16 base weights without persistent FP32 copies.

Only first-order adapter training is supported. Every matrix conversion is
temporary; backward saves the original frozen weight and converts it again.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd.function import once_differentiable


def _full_precision(device):
    if torch.is_autocast_enabled(device.type):raise ValueError('Streamed FP32 does not support autocast')


class _FrozenLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,inputs,weight,bias):
        if inputs.dtype!=torch.float32:raise ValueError('Streamed linear requires FP32 activations')
        _full_precision(inputs.device)
        if ctx.needs_input_grad[0]:ctx.save_for_backward(weight)
        return F.linear(inputs,weight.float(),None if bias is None else bias.float())

    @staticmethod
    @once_differentiable
    def backward(ctx,gradient):
        _full_precision(gradient.device)
        weight,=ctx.saved_tensors
        return gradient.float()@weight.float(),None,None


class FrozenFloatLinear(nn.Module):
    def __init__(self,source):
        super().__init__();self.weight=source.weight;self.bias=source.bias
        self.in_features=source.in_features;self.out_features=source.out_features

    def forward(self,inputs):
        return _FrozenLinearFunction.apply(inputs,self.weight,self.bias)


class FrozenFloatEmbedding(nn.Module):
    def __init__(self,source):
        super().__init__();self.weight=source.weight;self.padding_idx=source.padding_idx
        self.num_embeddings=source.num_embeddings;self.embedding_dim=source.embedding_dim

    def forward(self,inputs):
        _full_precision(inputs.device)
        return F.embedding(inputs,self.weight,padding_idx=self.padding_idx).float()


def install_streamed_float32(model):
    """Keep tied storage and adapter names; reject unsupported bases before editing."""
    if any(isinstance(m,(FrozenFloatLinear,FrozenFloatEmbedding)) for m in model.modules()):
        raise ValueError('Streamed base conversion is one-shot')
    selected=[(name,module) for name,module in model.named_modules() if isinstance(module,(nn.Linear,nn.Embedding))]
    if not selected:raise ValueError('Streamed base needs frozen linear or embedding weights')
    excluded=set()
    for name,module in selected:
        if not name or any(p.requires_grad or p.dtype!=torch.bfloat16 for p in module.parameters(recurse=False)):
            raise ValueError('Streamed base matrices must be frozen original BF16 parameters')
        if isinstance(module,nn.Embedding) and module.max_norm is not None:
            raise ValueError('Streamed embeddings cannot mutate frozen weights')
        excluded.update(id(p) for p in module.parameters(recurse=False))
    if any(p.requires_grad and (p.dtype!=torch.float32 or not name.endswith(('lora_a','lora_b')))
           for name,p in model.named_parameters()):raise ValueError('Only FP32 adapters may be trained with a streamed base')
    if len({p.device for p in model.parameters()})!=1:raise ValueError('Streamed base requires one locally selected device')
    promoted=0
    for p in model.parameters():
        if id(p) not in excluded and not p.requires_grad and p.is_floating_point() and p.dtype!=torch.float32:
            p.data=p.data.float();promoted+=1
    for name,module in selected:
        parent_name,_,attribute=name.rpartition('.')
        setattr(model.get_submodule(parent_name) if parent_name else model,attribute,
                FrozenFloatLinear(module) if isinstance(module,nn.Linear) else FrozenFloatEmbedding(module))
    return {'scheme':'frozen-bf16-storage-fp32-linear-recompute-v1','linear_compute_dtype':'float32',
        'base_matrix_storage_dtype':'bfloat16','linear_modules':sum(isinstance(m,nn.Linear) for _,m in selected),
        'embedding_modules':sum(isinstance(m,nn.Embedding) for _,m in selected),
        'small_frozen_parameters_promoted':promoted,
        'stored_parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
        'converted_base_matrices_saved_for_backward':False,'higher_order_derivatives_supported':False,
        'reference':'https://docs.pytorch.org/docs/2.12/notes/extending.html#extending-torch-autograd'}
