"""Bounded, base/version/data-specific safetensors LoRA contributions."""
import math
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file,load_file
from .language_adapter import adapter_parameters,MAX_ADAPTER_BYTES


def metadata(base_hash,adapter_id,dataset_id,start,count,tokens):
    return {'format':'lixyswarm-language-gradient-v1','base_hash':base_hash,'adapter_id':adapter_id,
            'dataset_id':dataset_id,'start':str(start),'count':str(count),'tokens':str(tokens)}


def save_gradient(model,path,*,base_hash,adapter_id,dataset_id,start,count,tokens):
    tensors={name:parameter.grad.detach().float().cpu().contiguous() for name,parameter in adapter_parameters(model).items()}
    if any(not torch.isfinite(t).all() for t in tensors.values()):raise ValueError('Nonfinite language gradient')
    if sum(t.numel()*t.element_size() for t in tensors.values())>MAX_ADAPTER_BYTES:raise ValueError('Language gradient exceeds its bound')
    save_file(tensors,path,metadata=metadata(base_hash,adapter_id,dataset_id,start,count,tokens))


def load_gradient(model,path,*,base_hash,adapter_id,dataset_id,start,count,tokens):
    if Path(path).stat().st_size>MAX_ADAPTER_BYTES:raise ValueError('Language gradient exceeds its bound')
    with safe_open(path,framework='pt',device='cpu') as source:
        if source.metadata()!=metadata(base_hash,adapter_id,dataset_id,start,count,tokens):raise ValueError('Gradient version, data or shard does not match')
    tensors=load_file(path,device='cpu');parameters=adapter_parameters(model)
    if set(tensors)!=set(parameters):raise ValueError('Language gradient names do not match')
    for name,tensor in tensors.items():
        if tensor.dtype!=torch.float32 or tensor.shape!=parameters[name].shape or not torch.isfinite(tensor).all():
            raise ValueError('Language gradient shape, type or values are invalid')
    norm=math.sqrt(sum(float(t.double().square().sum()) for t in tensors.values()))
    if not math.isfinite(norm) or norm>1.00001:raise ValueError('Language gradient exceeds its declared clipping bound')
    return tensors


def combine_gradients(model,contributions):
    """Token-weighted compatible shards; overlap never counts as new data."""
    if not contributions:raise ValueError('No language contributions')
    ranges=sorted((row['start'],row['start']+row['count']) for row in contributions)
    if any(a[1]>b[0] for a,b in zip(ranges,ranges[1:])):raise ValueError('Language training shards overlap')
    signature={(row['base_hash'],row['adapter_id'],row['dataset_id']) for row in contributions}
    if len(signature)!=1 or any(type(row['tokens']) is not int or row['tokens']<=0 for row in contributions):
        raise ValueError('Incompatible language versions or token counts')
    parameters=adapter_parameters(model);total=sum(row['tokens'] for row in contributions)
    combined={name:torch.zeros_like(p,device='cpu',dtype=torch.float32) for name,p in parameters.items()}
    for row in contributions:
        tensors=load_gradient(model,row['path'],**{k:row[k] for k in ['base_hash','adapter_id','dataset_id','start','count','tokens']})
        for name,tensor in tensors.items():combined[name].add_(tensor,alpha=row['tokens']/total)
    return combined
