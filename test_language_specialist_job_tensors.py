"""Full child boundaries, actual tiny training/routing and original trace gates."""
from copy import deepcopy
import hashlib
import io
import json
import sys
import time

import pytest
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,Qwen3Config,Qwen3ForCausalLM

import src.learning.language_specialist_job as job
import src.learning.language_specialist_protocol as protocol_module
import src.learning.language_selector_job as selector_job
import src.learning.language_selector_protocol as selector_protocol
import src.learning.public_bases as bases
import src.learning.public_conversation as conversation
import src.learning.public_math as public_math
import src.learning.public_math_job as math_job
import src.learning.language_candidate_job as candidate_job
import src.learning.language_adapter as adapters


@pytest.mark.parametrize('mode',['screen-reject','full-quality','interrupt-heldout'])
def test_specialist_main_runs_owned_training_and_all_gate_branches(tmp_path,monkeypatch,capsys,mode):
    base=tmp_path/'base';base.mkdir();(base/'tokenizer.json').write_bytes(b'public fixture tokenizer')
    monkeypatch.setitem(bases.BASES,'qwen3-1.7b',{'tokenizer.json':hashlib.sha256((base/'tokenizer.json').read_bytes()).hexdigest()})
    adapter=tmp_path/'adapter';adapter.write_bytes(b'public frozen parent fixture')
    parent_id=hashlib.sha256(adapter.read_bytes()).hexdigest()
    for module in [job,selector_job,selector_protocol]:monkeypatch.setattr(module,'PARENT_MODEL',parent_id)
    corpus={'train':[{'prompt':'old train '+str(i),'answer':'4','kind':'arithmetic'} for i in range(200)],
        'evaluation':[{'prompt':'old evaluation '+str(i),'answer':'5','kind':'arithmetic'} for i in range(72)]}
    monkeypatch.setattr(conversation,'public_corpus',lambda:corpus);monkeypatch.setattr(selector_protocol,'public_corpus',lambda:corpus)
    original=protocol_module.definition
    short=lambda:{**original(),'specialist_training_start':16,'specialist_training_examples':16,
        'math_training_examples':16,'specialist_batch_size':8,'specialist_replay_examples':2,'phase_steps':2}
    monkeypatch.setattr(protocol_module,'definition',short);monkeypatch.setattr(job,'definition',short)
    output=tmp_path/'engine';opened=[]
    evaluation={'version':1,'split':'evaluation','private_inputs':False,'rows':[{'case':hashlib.sha256(('eval'+str(i)).encode()).hexdigest(),
        'question':'evaluation problem '+str(i),'answer':'#### 1','final_number':'1'} for i in range(128)]}
    monkeypatch.setitem(public_math.ARTIFACTS,'evaluation',(public_math.digest(evaluation),len(public_math.canonical(evaluation))))
    def snapshot(path,kind):
        opened.append(kind)
        if kind=='evaluation':
            assert len(list(output.glob('*.safetensors')))==5 and (output/'training.json').exists()
            if mode=='interrupt-heldout':raise RuntimeError('fixture heldout interruption')
            return evaluation
        return {'rows':[{'question':'training problem '+str(i),'answer':'#### 1'} for i in range(32)]}
    monkeypatch.setattr(public_math,'load_snapshot',snapshot)
    class Tokenizer:
        eos_token_id=31
        def apply_chat_template(self,messages,**kwargs):return 'prefix:'+messages[0]['content']
        def encode(self,text,**kwargs):return [1,2,3] if text.startswith('prefix:') else [4]
    tokenizer=Tokenizer();monkeypatch.setattr(AutoTokenizer,'from_pretrained',lambda *_,**__:tokenizer)
    config=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=64,
        bos_token_id=1,eos_token_id=31,pad_token_id=0)
    torch.manual_seed(73);base_state=deepcopy(Qwen3ForCausalLM(config).state_dict());models=[];parent={}
    def factory(*args,**kwargs):
        model=Qwen3ForCausalLM(config);model.load_state_dict(base_state);model.to(torch.bfloat16)
        models.append(model);return model
    monkeypatch.setattr(AutoModelForCausalLM,'from_pretrained',factory)
    def load(model,*args,**kwargs):
        with torch.no_grad():
            for name,p in adapters.adapter_parameters(model).items():
                if not parent:break
                p.copy_(parent[name])
            if not parent:
                for name,p in adapters.adapter_parameters(model).items():
                    if name.endswith('lora_b'):p.normal_(std=.01)
                    parent[name]=p.detach().clone()
    monkeypatch.setattr(adapters,'load_adapter',load)
    # Both branches must work independently of whether this tiny fixture has a
    # useful candidate. Full answer verdicts still use the original real scorer.
    screen=mode=='full-quality'
    forced_screen=lambda *_:{'necessary_loss_gates_passed':screen,
        'math_nll_gate_passed':screen,'historical_nll_gate_passed':screen,'native_publication_authorized':False}
    monkeypatch.setattr(job,'necessary_loss_gate',forced_screen);monkeypatch.setattr(protocol_module,'necessary_loss_gate',forced_screen)
    from test_public_math import paired_results
    paired=paired_results(evaluation);assessment=[]
    def nll(model):
        with torch.inference_mode():return float(model(input_ids=torch.tensor([[1,2,3,4,31]]),
            labels=torch.tensor([[-100,-100,-100,4,31]]),use_cache=False).loss)
    def historical(model,*args,**kwargs):
        from src.learning.language_specialist import RoutedAdapterExperts
        side=int(isinstance(model,RoutedAdapterExperts));assessment.append(('historical',side))
        return {**deepcopy(paired[side]['legacy']),'nll':nll(model),'tokens':144}
    def mathematical(model,*args,**kwargs):
        from src.learning.language_specialist import RoutedAdapterExperts
        side=int(isinstance(model,RoutedAdapterExperts));assessment.append(('math',side))
        return {'nll':nll(model),'tokens':256,'answers':deepcopy(paired[side]['answers']),
            'correct':4 if side else 0,'total':128,'output_tokens':[1]*128,'at_output_limit':0}
    monkeypatch.setattr(candidate_job,'evaluate',historical);monkeypatch.setattr(math_job,'evaluate_math',mathematical)
    request=dict(action=short()['action'],protocol_id=job.protocol_id(),compute_dtype='float32',streamed_base=True,
        base_directory=str(base),adapter_file=str(adapter),model_id=parent_id,parent_release_id=job.PARENT_RELEASE,
        training_file=str(tmp_path/'train'),evaluation_file=str(tmp_path/'eval'),output_directory=str(output),
        device='cpu',cpu_threads=2,gpu_fraction=.5,deadline=time.time()+120)
    if mode=='interrupt-heldout':
        with pytest.raises(RuntimeError,match='fixture heldout interruption'):job.run(request)
        assert all(torch.equal(p,parent[n]) for n,p in adapters.adapter_parameters(models[0]).items())
        assert not (output/'diagnostic.json').exists();return
    monkeypatch.setattr(sys,'stdin',type('Input',(),{'buffer':io.BytesIO(json.dumps(request).encode())})())
    job.main();captured=capsys.readouterr();response=json.loads(captured.out)
    raw=(output/'diagnostic.json').read_bytes();report=json.loads(raw)
    assert hashlib.sha256(raw).hexdigest()==response['report_id']
    assert report['specialist_training']['steps']==2 and sorted(report['specialist_training']['math_order'])==list(range(16))
    assert report['binding']['expert_ids']==[parent_id,report['selector_artifacts']['specialist.safetensors']['sha256']]
    assert report['parent_parameters_restored'] and not report['generalist_files_written']
    assert report['updated_model_files_written'] and report['language_expert_files_written']
    assert not report['eligible_for_native_promotion'] and not report['held_out_data_used_for_training_or_selection']
    assert report['generation_performed'] is screen
    assert selector_protocol.verify_loss_report(report,protocol=short(),identifier=job.protocol_id())['losses_reconstructed']
    assert protocol_module.verify_specialist_report(report)['complete_math_training_coverage_verified']
    changed=deepcopy(report);changed['specialist_training']['math_order'][0]=changed['specialist_training']['math_order'][1]
    with pytest.raises(ValueError,match='coverage'):protocol_module.verify_specialist_report(changed)
    assert all(torch.equal(p,parent[n]) for n,p in adapters.adapter_parameters(models[0]).items())
    assert opened==['train','evaluation']
    if screen:
        assert assessment==[('historical',0),('math',0),('historical',1),('math',1),('historical',0),('math',0)]
        assert set(report['quality'])=={'parent_fp32','parent_bf16'} and len(models)==2
        for label in ['fp32','bf16']:
            value=report['quality']['parent_'+label];trace=json.loads((output/('math-traces-'+label+'.json')).read_text())
            assert public_math.verify_math_transition(value,evaluation,trace)==value['verdict']
        assert report['eligible_for_research_promotion']==all(v['verdict']['eligible_for_research_promotion'] for v in report['quality'].values())
    else:
        assert not assessment and not report['quality'] and not report['eligible_for_research_promotion']
