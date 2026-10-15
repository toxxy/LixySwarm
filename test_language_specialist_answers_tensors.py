from copy import deepcopy
import hashlib
import io
import json
import sys
import time

import pytest
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,Qwen3Config,Qwen3ForCausalLM

import src.learning.language_specialist_answers as job
import src.learning.language_selector_job as selector_job
import src.learning.public_bases as bases
import src.learning.public_math as math
import src.learning.public_math_job as math_job
import src.learning.language_adapter as adapter


@pytest.mark.parametrize('mutate',[False,True])
def test_read_only_answers_keep_rejection_and_bind_raw_trace(tmp_path,monkeypatch,capsys,mutate):
    base=tmp_path/'base';base.mkdir();(base/'tokenizer.json').write_bytes(b'public token fixture')
    monkeypatch.setitem(bases.BASES,'qwen3-1.7b',{'tokenizer.json':hashlib.sha256((base/'tokenizer.json').read_bytes()).hexdigest()})
    parent=tmp_path/'parent';parent.write_bytes(b'public parent')
    specialist=tmp_path/'specialist';specialist.write_bytes(b'public specialist')
    parent_id=hashlib.sha256(parent.read_bytes()).hexdigest();specialist_id=hashlib.sha256(specialist.read_bytes()).hexdigest()
    monkeypatch.setattr(job,'PARENT_MODEL',parent_id);monkeypatch.setattr(selector_job,'PARENT_MODEL',parent_id)
    monkeypatch.setattr(job,'SPECIALIST_ID',specialist_id)
    evaluation={'split':'evaluation','rows':[{'case':hashlib.sha256(str(i).encode()).hexdigest(),
        'question':'public fixture','answer':'#### 1','final_number':'1'} for i in range(128)]}
    monkeypatch.setitem(math.ARTIFACTS,'evaluation',(math.digest(evaluation),len(math.canonical(evaluation))))
    before={'nll':1.,'tokens':128,'answers':[{'case':r['case'],'answer':'#### 0'} for r in evaluation['rows']],
        'correct':0,'total':128,'output_tokens':[1]*128,'at_output_limit':0}
    cached={'full_public_math_trace':{'before':before,'evaluation_artifact_id':math.ARTIFACTS['evaluation'][0]},
        'research_report':{'diagnostic':{'model_id':parent_id,'parent_release_id':job.PARENT_RELEASE}}}
    cache=tmp_path/'cached.json';cache.write_bytes(math.canonical(cached))
    monkeypatch.setattr(job,'BASELINE_REPORT_ID',hashlib.sha256(cache.read_bytes()).hexdigest())
    monkeypatch.setattr(math,'load_snapshot',lambda *_:evaluation)
    class Tokenizer:pass
    monkeypatch.setattr(AutoTokenizer,'from_pretrained',lambda *_,**__:Tokenizer())
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=64)).to(torch.bfloat16)
    monkeypatch.setattr(AutoModelForCausalLM,'from_pretrained',lambda *_,**__:model)
    monkeypatch.setattr(adapter,'load_adapter',lambda *_:None)
    def evaluate(*args,**kwargs):
        if mutate:
            with torch.no_grad():next(iter(adapter.adapter_parameters(model).values())).add_(1.)
        after=deepcopy(before);after['nll']=.8;after['correct']=5
        for row in after['answers'][:5]:row['answer']='#### 1'
        return after
    monkeypatch.setattr(math_job,'evaluate_math',evaluate)
    request=dict(action=job.definition()['action'],protocol_id=job.protocol_id(),compute_dtype='float32',streamed_base=True,
        base_directory=str(base),adapter_file=str(parent),specialist_file=str(specialist),cached_baseline_file=str(cache),
        model_id=parent_id,parent_release_id=job.PARENT_RELEASE,training_file=str(tmp_path/'unused-train'),evaluation_file=str(tmp_path/'eval'),
        output_directory=str(tmp_path/'output'),device='cpu',cpu_threads=2,gpu_fraction=.5,deadline=time.time()+60)
    monkeypatch.setattr(sys,'stdin',type('Input',(),{'buffer':io.BytesIO(json.dumps(request).encode())})())
    if mutate:
        with pytest.raises(ValueError,match='changed the specialist'):job.main()
        return
    job.main();response=json.loads(capsys.readouterr().out);raw=(tmp_path/'output/diagnostic.json').read_bytes();report=json.loads(raw)
    assert response['report_id']==hashlib.sha256(raw).hexdigest()
    assert report['correct_before']==0 and report['correct_after']==5 and report['accuracy_gain']==5
    assert all(report['math_only_conditions'].values()) and not report['native_publication_authorized']
    assert not report['training_performed'] and report['original_combined_system_rejection_retained']
    trace=json.loads((tmp_path/'output/math-traces.json').read_text())
    assert math.compact_evaluation(trace['after'])==report['math_after']
