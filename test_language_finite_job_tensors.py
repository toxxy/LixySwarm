"""Exercise the whole research job with real tiny tensors and pinned I/O mocks."""
import hashlib,json
from types import SimpleNamespace
import pytest
torch=pytest.importorskip('torch')

from src.learning import language_finite_step_job as job
from src.learning.language_finite_protocol import definition,protocol_id
from test_language_retention import request


@pytest.mark.parametrize('group',[False,True,'streamed'])
def test_completed_job_binds_request_protocol_after_base_verification_and_restores(tmp_path,monkeypatch,group):
    from src.learning import public_bases,public_math,weights_io,language_adapter,language_gradient,language_finite_step
    import transformers
    value=request(tmp_path)
    streamed=group=='streamed'
    value.update(action='public-streamed-group-finite-step-v3' if streamed else 'public-group-finite-step-v2' if group else 'public-finite-step-v1',
        protocol_id=protocol_id('float32',group_direction=bool(group),streamed_base=streamed))
    if streamed:value['streamed_base']=True
    digests={str(tmp_path/'base'/name):(identifier,1) for name,identifier in public_bases.BASES['qwen3-1.7b'].items()}
    digests[value['adapter_file']]=(value['model_id'],1)
    digests.update({r['path']:(r['gradient_id'],r['gradient_bytes']) for r in value['contributions']})
    monkeypatch.setattr(weights_io,'digest_file',lambda path:digests[str(path)])
    monkeypatch.setattr(public_math,'load_snapshot',lambda *_a:{'rows':[{'question':'Public math','answer':'2'} for _ in range(192)]})
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
    model=Tiny();before=model.lora_a.detach().clone()
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *_a,**_kw:Tokenizer())
    monkeypatch.setattr(transformers.AutoModelForCausalLM,'from_pretrained',lambda *_a,**_kw:model)
    monkeypatch.setattr(language_adapter,'install_adapter',lambda _m:{'lora_a':model.lora_a})
    monkeypatch.setattr(language_adapter,'load_adapter',lambda *_a:None)
    monkeypatch.setattr(language_gradient,'combine_gradients',lambda *_a,**_kw:{'lora_a':torch.ones(1)})
    def measurement(*_a):
        return {**{name:1. for name in definition()['protected']},'math_train192':1+float(model.lora_a.detach())}
    monkeypatch.setattr(language_finite_step,'public_training_measurement',measurement)
    response=job.run(value)
    raw=(tmp_path/'output/diagnostic.json').read_bytes();result=json.loads(raw)
    assert response['protocol_id']==result['protocol_id']==value['protocol_id']
    assert response['report_id']==hashlib.sha256(raw).hexdigest() and response['bytes']==len(raw)
    assert result['parent_parameters_restored'] and result['guard']['applied']
    assert result['guard']['accepted_rate']==.001 and torch.equal(model.lora_a,before)
    assert not result['updated_model_files_written'] and not result['held_out_data_used_in_compute']
    if streamed:
        assert result['base_execution']['base_matrix_storage_dtype']=='bfloat16'
        assert result['base_execution']['linear_compute_dtype']=='float32'
