"""Bounded public LoRA safetensors inspection without a tensor runtime."""
import array
import json
import math
from pathlib import Path
import struct
import sys

from .manifest import ReleaseError


def _unique_object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ReleaseError('Duplicate adapter header field')
        result[key]=value
    return result


def validate_adapter(path,config,base_hash):
    path=Path(path);size=path.stat().st_size
    if path.is_symlink() or not 8<size<=32*1024**2:
        raise ReleaseError('Invalid bounded adapter file')
    with path.open('rb') as source:
        header_size=struct.unpack('<Q',source.read(8))[0]
        if not 2<=header_size<=64*1024 or header_size+8>=size:
            raise ReleaseError('Invalid safetensors adapter header')
        header=json.loads(source.read(header_size),object_pairs_hook=_unique_object)
        metadata={'format':'lixyswarm-lora-v1','base_model_hash':base_hash,'rank':'8','alpha':'16','targets':'q_proj,v_proj'}
        if not isinstance(header,dict) or header.pop('__metadata__',None)!=metadata:
            raise ReleaseError('Adapter metadata differs from the public base')
        layers,hidden,heads,kv,dimension=(config[k] for k in ('num_hidden_layers','hidden_size','num_attention_heads','num_key_value_heads','head_dim'))
        if any(type(v) is not int or not 1<=v<=8192 for v in (layers,hidden,heads,kv,dimension)) or layers>128:
            raise ReleaseError('Invalid pinned adapter architecture')
        expected={}
        for layer in range(layers):
            for projection,width in [('q_proj',heads*dimension),('v_proj',kv*dimension)]:
                prefix=f'model.layers.{layer}.self_attn.{projection}'
                expected[prefix+'.lora_a']=[8,hidden]
                expected[prefix+'.lora_b']=[width,8]
        if set(header)!=set(expected):
            raise ReleaseError('Adapter parameter names do not match the public architecture')
        ranges=[]
        for name,row in header.items():
            if (not isinstance(row,dict) or set(row)!={'dtype','shape','data_offsets'}
                    or row['dtype']!='F32' or row['shape']!=expected[name]
                    or any(type(v) is not int for v in row['shape'])
                    or not isinstance(row['data_offsets'],list) or len(row['data_offsets'])!=2
                    or any(type(v) is not int or v<0 for v in row['data_offsets'])):
                raise ReleaseError('Invalid adapter tensor shape or type')
            start,end=row['data_offsets']
            if end-start!=4*math.prod(expected[name]):
                raise ReleaseError('Adapter tensor byte ranges do not match their shapes')
            ranges.append((start,end))
        ranges.sort()
        if ranges[0][0]!=0 or any(a[1]!=b[0] for a,b in zip(ranges,ranges[1:])) or ranges[-1][1]!=size-header_size-8:
            raise ReleaseError('Adapter byte ranges overlap, contain gaps or exceed the file')
        while block:=source.read(64*1024):
            values=array.array('f');values.frombytes(block)
            if sys.byteorder!='little':values.byteswap()
            if not all(math.isfinite(v) for v in values):
                raise ReleaseError('Adapter contains nonfinite weights')
    return {'parameters':sum(math.prod(v) for v in expected.values()),'bytes':size}
