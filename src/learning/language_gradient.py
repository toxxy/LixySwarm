"""Bounded, base/version/data-specific safetensors LoRA contributions."""
import math
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file,load_file
from .language_adapter import adapter_parameters,MAX_ADAPTER_BYTES


def metadata(base_hash,adapter_id,dataset_id,start,count,tokens,*,policy_version=1,quantization_relative_l2_error=None):
    from .language_policy import limits,HALF_POLICY,HALF_POLICY_ID
    limits(policy_version)
    value={'format':'lixyswarm-language-gradient-v1','base_hash':base_hash,'adapter_id':adapter_id,
            'dataset_id':dataset_id,'start':str(start),'count':str(count),'tokens':str(tokens)}
    if policy_version==3:
        error=quantization_relative_l2_error
        if type(error) not in (int,float) or not math.isfinite(error) or not 0<=error<=HALF_POLICY['maximum_relative_l2_error']:
            raise ValueError('Invalid half-precision reconstruction error')
        value.update(format='lixyswarm-language-gradient-v2',training_policy_id=HALF_POLICY_ID,
            quantization_relative_l2_error=repr(float(error)))
    elif quantization_relative_l2_error is not None:raise ValueError('Legacy gradients cannot claim half precision')
    return value


def _norm(tensors):return math.sqrt(sum(float(t.double().square().sum()) for t in tensors.values()))


def save_gradient(model,path,*,base_hash,adapter_id,dataset_id,start,count,tokens,policy_version=1):
    tensors={name:parameter.grad.detach().float().cpu().contiguous() for name,parameter in adapter_parameters(model).items()}
    if any(not torch.isfinite(t).all() for t in tensors.values()):raise ValueError('Nonfinite language gradient')
    if sum(t.numel()*t.element_size() for t in tensors.values())>MAX_ADAPTER_BYTES:raise ValueError('Language gradient exceeds its bound')
    metrics={}
    if policy_version==3:
        source_norm=_norm(tensors)
        if not math.isfinite(source_norm) or source_norm>1.00001:raise ValueError('Source gradient exceeds its clipping bound')
        packed={name:t.half() for name,t in tensors.items()}
        decoded={name:t.float() for name,t in packed.items()};decoded_norm=_norm(decoded)
        if decoded_norm>1:decoded={name:t/decoded_norm for name,t in decoded.items()}
        error=_norm({name:tensors[name]-t for name,t in decoded.items()})/source_norm if source_norm else 0.
        metrics={'quantization_relative_l2_error':error};tensors=packed
    save_file(tensors,path,metadata=metadata(base_hash,adapter_id,dataset_id,start,count,tokens,policy_version=policy_version,**metrics))
    return metrics


def load_gradient(model,path,*,base_hash,adapter_id,dataset_id,start,count,tokens,policy_version=1,quantization_relative_l2_error=None):
    if Path(path).stat().st_size>MAX_ADAPTER_BYTES:raise ValueError('Language gradient exceeds its bound')
    with safe_open(path,framework='pt',device='cpu') as source:
        if source.metadata()!=metadata(base_hash,adapter_id,dataset_id,start,count,tokens,policy_version=policy_version,
                quantization_relative_l2_error=quantization_relative_l2_error):raise ValueError('Gradient version, data or shard does not match')
    tensors=load_file(path,device='cpu');parameters=adapter_parameters(model)
    if set(tensors)!=set(parameters):raise ValueError('Language gradient names do not match')
    for name,tensor in tensors.items():
        if tensor.dtype!=(torch.float16 if policy_version==3 else torch.float32) or tensor.shape!=parameters[name].shape or not torch.isfinite(tensor).all():
            raise ValueError('Language gradient shape, type or values are invalid')
    if policy_version==3:tensors={name:t.float() for name,t in tensors.items()}
    norm=math.sqrt(sum(float(t.double().square().sum()) for t in tensors.values()))
    if not math.isfinite(norm) or norm>(1.005 if policy_version==3 else 1.00001):raise ValueError('Language gradient exceeds its declared clipping bound')
    if policy_version==3 and norm>1:tensors={name:t/norm for name,t in tensors.items()}
    return tensors


def combine_gradients(model,contributions,*,policy_version=1):
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
        fields={k:row[k] for k in ['base_hash','adapter_id','dataset_id','start','count','tokens']}
        if policy_version==3:fields['quantization_relative_l2_error']=row['quantization_relative_l2_error']
        tensors=load_gradient(model,row['path'],policy_version=policy_version,**fields)
        for name,tensor in tensors.items():combined[name].add_(tensor,alpha=row['tokens']/total)
    return combined
