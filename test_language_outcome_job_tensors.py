from copy import deepcopy
import hashlib
import json
import time

import pytest
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,Qwen3Config,Qwen3ForCausalLM

import src.learning.language_outcome_job as job
import src.learning.language_outcomes as protocol
import src.learning.language_selector_job as common
import src.learning.public_bases as bases
import src.learning.public_math as math
import src.learning.public_math_job as generation
import src.learning.language_adapter as adapter


@pytest.mark.parametrize('mutate',[False,True])
def test_outcome_job_loads_two_frozen_adapters_without_eval_or_optimizer(tmp_path,monkeypatch,mutate):
    base=tmp_path/'base';base.mkdir();(base/'tokenizer.json').write_bytes(b'public tokenizer fixture')
    monkeypatch.setitem(bases.BASES,'qwen3-1.7b',{'tokenizer.json':hashlib.sha256((base/'tokenizer.json').read_bytes()).hexdigest()})
    parent=tmp_path/'parent';parent.write_bytes(b'public parent');specialist=tmp_path/'specialist';specialist.write_bytes(b'public specialist')
    pid=hashlib.sha256(parent.read_bytes()).hexdigest();sid=hashlib.sha256(specialist.read_bytes()).hexdigest()
    for module in [job,protocol,common]:monkeypatch.setattr(module,'PARENT_MODEL',pid)
    for module in [job,protocol]:monkeypatch.setattr(module,'SPECIALIST_ID',sid)
    snapshot={'split':'train','rows':[{'case':f'{i:064x}','question':'public','answer':'#### 1','final_number':'1'} for i in range(128)]}
    opened=[]
    def load(path,split):
        opened.append(split);assert split=='train';return snapshot
    monkeypatch.setattr(math,'load_snapshot',load)
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=64)).to(torch.bfloat16)
    monkeypatch.setattr(AutoTokenizer,'from_pretrained',lambda *a,**kw:object())
    monkeypatch.setattr(AutoModelForCausalLM,'from_pretrained',lambda *a,**kw:model)
    def load_adapter(model,path,base_hash):
        with torch.no_grad():
            for p in adapter.adapter_parameters(model).values():p.fill_(.1 if Path(path)==parent else .2)
    from pathlib import Path
    monkeypatch.setattr(adapter,'load_adapter',load_adapter)
    seen=[]
    def frozen_parameters():return {n:p for n,p in model.named_parameters() if n.endswith(('.lora_a','.lora_b'))}
    def evaluate(active,tokenizer,selected,device,deadline,**kwargs):
        assert kwargs['record_generation_seconds'];assert selected['split']=='train'
        seen.append(float(next(iter(frozen_parameters().values())).flatten()[0]))
        if mutate:
            # The transactional expert bank must restore even a faulty generation callback.
            with torch.no_grad():next(iter(frozen_parameters().values())).add_(1.)
        if mutate:raise RuntimeError('generation failure')
        answers=[{'case':r['case'],'answer':'#### 1'} for r in selected['rows']]
        assert all(kwargs['score'](answers,selected).values())
        return {'answers':answers,'output_tokens':[4]*64,'generation_seconds':[.1]*64,
            'correct':64,'total':64,'nll':.5,'tokens':256,'at_output_limit':0}
    monkeypatch.setattr(generation,'evaluate_math',evaluate)
    monkeypatch.setattr(torch.optim,'AdamW',lambda *a,**kw:pytest.fail('Outcome generation cannot train'))
    request=dict(action=job.definition()['action'],protocol_id=job.protocol_id(),compute_dtype='float32',streamed_base=True,
        base_directory=str(base),adapter_file=str(parent),specialist_file=str(specialist),model_id=pid,parent_release_id=job.PARENT_RELEASE,
        training_file=str(tmp_path/'train'),output_directory=str(tmp_path/'out'),device='cpu',cpu_threads=2,gpu_fraction=.5,deadline=time.time()+60,shard=0)
    bad={**request,'evaluation_file':str(tmp_path/'eval')}
    with pytest.raises(ValueError,match='evaluation snapshot'):job.validate_request(bad)
    if mutate:
        with pytest.raises(RuntimeError,match='generation failure'):job.run(request)
    else:
        response=job.run(request);report=json.loads((tmp_path/'out/diagnostic.json').read_text())
        assert response['success'] and report['public_training_only'] and not report['updated_model_files_written']
        assert seen==pytest.approx([.1,.2])
        assert not list((tmp_path/'out').glob('*.safetensors'))
    assert opened==['train']
    assert all(torch.allclose(p,torch.full_like(p,.1)) for p in frozen_parameters().values())
