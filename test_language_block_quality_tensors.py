"""Complete quality-job orchestration, real tiny gradients and verified traces."""
import copy,hashlib,json
from types import SimpleNamespace
import pytest
torch=pytest.importorskip('torch')
from test_language_block_protocol import quality_request


@pytest.mark.parametrize('interrupt_after_training',[False,True])
def test_quality_is_observed_only_before_and_after_train_selected_block(tmp_path,monkeypatch,interrupt_after_training):
    import transformers
    from src.learning import language_block_quality_job as job,public_math as math,public_bases,weights_io
    from src.learning import language_adapter,language_gradient,language_finite_step,language_candidate_job,public_math_job
    from test_public_math import paired_results
    evaluation={'version':1,'split':'evaluation','private_inputs':False,'rows':[
        {'case':hashlib.sha256(str(i).encode()).hexdigest(),'question':'Public fixture','answer':'#### 1','final_number':'1'} for i in range(128)]}
    encoded=math.canonical(evaluation);monkeypatch.setitem(math.ARTIFACTS,'evaluation',(math.digest(evaluation),len(encoded)))
    value=quality_request(tmp_path)
    digests={str(tmp_path/'base'/name):(identifier,1) for name,identifier in public_bases.BASES['qwen3-1.7b'].items()}
    digests[value['adapter_file']]=(value['model_id'],1)
    digests.update({r['path']:(r['gradient_id'],r['gradient_bytes']) for r in value['contributions']})
    monkeypatch.setattr(weights_io,'digest_file',lambda path:digests[str(path)])
    monkeypatch.setattr(math,'load_snapshot',lambda _path,split:evaluation if split=='evaluation' else {
        'rows':[{'question':'Public math','answer':'2'} for _ in range(192)]})
    class Tokenizer:
        eos_token_id=9
        def apply_chat_template(self,*_a,**_kw):return 'prefix'
        def encode(self,*_a,**_kw):return [1]
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.lora_a=torch.nn.Parameter(torch.tensor([.5]));self.config=SimpleNamespace(use_cache=True)
            self.base=torch.nn.Linear(1,1,dtype=torch.bfloat16).requires_grad_(False)
        def gradient_checkpointing_enable(self,**_kw):pass
        def forward(self,**_kw):return SimpleNamespace(loss=self.lora_a.square().mean())
    model=Tiny();initial=model.lora_a.detach().clone();events=[]
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *_a,**_kw:Tokenizer())
    monkeypatch.setattr(transformers.AutoModelForCausalLM,'from_pretrained',lambda *_a,**_kw:model)
    monkeypatch.setattr(language_adapter,'install_adapter',lambda _m:{'lora_a':model.lora_a})
    monkeypatch.setattr(language_adapter,'load_adapter',lambda *_a:None)
    monkeypatch.setattr(language_gradient,'combine_gradients',lambda *_a,**_kw:{'lora_a':torch.ones(1)})
    paired=paired_results(evaluation)
    def side():return int(model.lora_a.item()<.4999)
    def historical(*_a,**_kw):
        events.append(('quality-historical',float(model.lora_a.detach())))
        if interrupt_after_training and side():raise RuntimeError('Interrupted final quality assessment')
        return copy.deepcopy(paired[side()]['legacy'])
    def mathematical(*_a,**_kw):
        events.append(('quality-math',float(model.lora_a.detach())))
        raw=copy.deepcopy(paired[side()]);raw.pop('legacy')
        raw.update(tokens=128,total=128,correct=sum(math.score_answers(raw['answers'],evaluation).values()),
                   output_tokens=[1]*128,at_output_limit=0)
        return raw
    def measure(*_a):
        events.append(('train',float(model.lora_a.detach())))
        return {**{name:1. for name in job.definition()['protected']},'math_train192':1+float(model.lora_a.detach())}
    monkeypatch.setattr(language_candidate_job,'evaluate',historical)
    monkeypatch.setattr(public_math_job,'evaluate_math',mathematical)
    monkeypatch.setattr(language_finite_step,'public_training_measurement',measure)
    if interrupt_after_training:
        with pytest.raises(RuntimeError):job.run(value)
        assert torch.equal(model.lora_a,initial);return
    response=job.run(value);raw=(tmp_path/'output/diagnostic.json').read_bytes();result=json.loads(raw)
    assert response['protocol_id']==value['protocol_id']==result['protocol_id']
    assert response['report_id']==hashlib.sha256(raw).hexdigest()
    assert result['block']['accepted_steps']==4 and result['quality']['math_verdict']['eligible_for_research_promotion']
    trace=json.loads((tmp_path/'output/math-traces.json').read_text())
    assert math.verify_math_transition(result['quality'],evaluation,trace)==result['quality']['math_verdict']
    assert result['held_out_data_used_for_training_or_selection'] is False and not result['updated_model_files_written']
    assert result['parent_parameters_restored'] and torch.equal(model.lora_a,initial)
    train_indices=[i for i,(kind,_v) in enumerate(events) if kind=='train']
    assert len(train_indices)==12 and all(events[i][0]=='train' for i in range(min(train_indices),max(train_indices)+1))
    assert [kind for kind,_v in events if kind.startswith('quality-')]==[
        'quality-historical','quality-math','quality-historical','quality-math']
