"""Complete selector child orchestration with tiny real Qwen and public IO fixtures."""
import hashlib
import json
import time

import pytest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM,AutoTokenizer,AutoModelForCausalLM

import src.learning.language_selector_job as job
import src.learning.language_selector_protocol as protocol_module
import src.learning.public_bases as bases
import src.learning.public_conversation as conversation
import src.learning.public_math as public_math
import src.learning.language_adapter as adapters


@pytest.mark.parametrize('fail_evaluation',[False,True])
def test_full_child_freezes_before_heldout_saves_reloadable_state_and_runs_real_routes(tmp_path,monkeypatch,fail_evaluation):
    base=tmp_path/'base';base.mkdir();(base/'tokenizer.json').write_bytes(b'public fixture tokenizer')
    tokenizer_hash=hashlib.sha256((base/'tokenizer.json').read_bytes()).hexdigest()
    monkeypatch.setitem(bases.BASES,'qwen3-1.7b',{'tokenizer.json':tokenizer_hash})
    adapter=tmp_path/'adapter';adapter.write_bytes(b'public fixture accepted adapter')
    parent=hashlib.sha256(adapter.read_bytes()).hexdigest()
    monkeypatch.setattr(job,'PARENT_MODEL',parent);monkeypatch.setattr(protocol_module,'PARENT_MODEL',parent)
    corpus={'train':[{'prompt':'old train '+str(i),'answer':'4','kind':'arithmetic'} for i in range(200)],
            'evaluation':[{'prompt':'old heldout '+str(i),'answer':'5','kind':'arithmetic'} for i in range(72)]}
    monkeypatch.setattr(conversation,'public_corpus',lambda:corpus)
    monkeypatch.setattr(protocol_module,'public_corpus',lambda:corpus)
    original_definition=protocol_module.definition
    short=lambda:{**original_definition(),'phase_steps':1}
    monkeypatch.setattr(protocol_module,'definition',short);monkeypatch.setattr(job,'definition',short)
    output=tmp_path/'engine';opened=[]
    def snapshot(path,kind):
        opened.append(kind)
        if kind=='evaluation':
            assert len(list(output.glob('*.safetensors')))==4
            assert (output/'training.json').exists() and (output/'optimizer-state.pt').exists()
            if fail_evaluation:raise RuntimeError('fixture evaluation failure')
        return {'rows':[{'question':kind+' problem '+str(i),'answer':'6','case':str(i)}
                        for i in range(192 if kind=='train' else 128)]}
    monkeypatch.setattr(public_math,'load_snapshot',snapshot)
    class Tokenizer:
        eos_token_id=31
        def apply_chat_template(self,messages,**kwargs):return messages[0]['content']
        def encode(self,text,**kwargs):return [1,2,3] if len(text)>2 else [4]
    monkeypatch.setattr(AutoTokenizer,'from_pretrained',lambda *_,**__:Tokenizer())
    torch.manual_seed(63)
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
        max_position_embeddings=64,bos_token_id=1,eos_token_id=31,pad_token_id=0)).to(torch.bfloat16)
    monkeypatch.setattr(AutoModelForCausalLM,'from_pretrained',lambda *_,**__:model)
    originals={}
    def load(net,*_,**__):
        with torch.no_grad():
            for name,p in adapters.adapter_parameters(net).items():
                if name.endswith('lora_b'):p.normal_(std=.01)
                originals[name]=p.detach().clone()
    monkeypatch.setattr(adapters,'load_adapter',load)
    request=dict(action=short()['action'],protocol_id=job.protocol_id(),compute_dtype='float32',streamed_base=True,
        base_directory=str(base),adapter_file=str(adapter),model_id=parent,parent_release_id=job.PARENT_RELEASE,
        training_file=str(tmp_path/'train'),evaluation_file=str(tmp_path/'eval'),output_directory=str(output),
        device='cpu',cpu_threads=2,gpu_fraction=.5,deadline=time.time()+120)
    if fail_evaluation:
        with pytest.raises(RuntimeError,match='fixture evaluation failure'):job.run(request)
        assert all(torch.equal(p,originals[n]) for n,p in adapters.adapter_parameters(model).items())
        assert not (output/'diagnostic.json').exists()
        return
    response=job.run(request);raw=(output/'diagnostic.json').read_bytes();report=json.loads(raw)
    assert response['report_id']==hashlib.sha256(raw).hexdigest() and response['bytes']==len(raw)
    assert opened==['train','evaluation']
    assert report['parent_parameters_restored'] and not report['generation_performed']
    assert not report['full_generation_gates_evaluated'] and not report['eligible_for_native_promotion']
    assert report['selector_state_files_written'] and not report['language_expert_files_written']
    assert not report['held_out_data_used_for_training_or_selection']
    assert len(report['heldout_expert_nll'])==200
    assert protocol_module.verify_loss_report(report)['losses_reconstructed']
    changed=json.loads(raw);changed['evaluations']['public']['math']['nll']+=.01
    with pytest.raises(ValueError,match='does not reconstruct'):protocol_module.verify_loss_report(changed)
    assert all(torch.equal(p,originals[n]) for n,p in adapters.adapter_parameters(model).items())
    for mode in ['public','zero','shuffled','dolphin']:
        row=report['evaluations'][mode]
        assert row['single_selected_expert_executed_per_request'] and len(row['choices'])==200
        assert row['historical']['examples']==72 and row['math']['examples']==128
