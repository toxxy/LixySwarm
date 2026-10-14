import json
import math
import struct

import pytest

from src.release.adapter_format import validate_adapter
from src.release.manifest import ReleaseError


CONFIG={'num_hidden_layers':1,'hidden_size':2,'num_attention_heads':1,'num_key_value_heads':1,'head_dim':2}


def fixture(path,mutate=lambda _:None,nonfinite=False):
    header={'__metadata__':{'format':'lixyswarm-lora-v1','base_model_hash':'a'*64,'rank':'8','alpha':'16','targets':'q_proj,v_proj'}}
    values=[]
    for projection in ['q_proj','v_proj']:
        for suffix,shape in [('lora_a',[8,2]),('lora_b',[2,8])]:
            start=4*len(values);values.extend([.1]*math.prod(shape))
            header[f'model.layers.0.self_attn.{projection}.{suffix}']={'dtype':'F32','shape':shape,'data_offsets':[start,4*len(values)]}
    mutate(header)
    if nonfinite:values[-1]=float('nan')
    encoded=json.dumps(header).encode();encoded+=b' '*((-len(encoded))%8)
    path.write_bytes(struct.pack('<Q',len(encoded))+encoded+struct.pack('<'+'f'*len(values),*values))


def test_public_adapter_inspection_is_weights_only_and_needs_no_tensor_runtime(tmp_path):
    path=tmp_path/'adapter.safetensors';fixture(path)
    assert validate_adapter(path,CONFIG,'a'*64)['parameters']==64


@pytest.mark.parametrize('failure',['base','name','dtype','offset','nonfinite'])
def test_incompatible_headers_and_nonfinite_weights_are_rejected(tmp_path,failure):
    key='model.layers.0.self_attn.q_proj.lora_a'
    def mutate(header):
        if failure=='base':header['__metadata__']['base_model_hash']='b'*64
        elif failure=='name':header['../personal.enc']=header.pop(key)
        elif failure=='dtype':header[key]['dtype']='F16'
        elif failure=='offset':header[key]['data_offsets']=[0,1]
    path=tmp_path/'adapter.safetensors';fixture(path,mutate,nonfinite=failure=='nonfinite')
    with pytest.raises(ReleaseError):validate_adapter(path,CONFIG,'a'*64)
