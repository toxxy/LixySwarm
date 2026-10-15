"""Real derivatives and parent restoration; no external model downloads."""
import time
from types import SimpleNamespace

import pytest
torch=pytest.importorskip('torch')

from src.learning.language_retention_job import training_statistics,direction_statistics,temporary_step


def test_reference_direction_sign_and_finite_step_can_disagree():
    # Even a correct downhill tangent need not preserve a finite-step loss.
    reference={'p':torch.tensor([.01])};direction={'p':torch.tensor([1.])}
    value=direction_statistics(direction,reference,rate=1.)
    assert value['dot']>0 and value['predicted_first_order_loss_change']<0
    old=torch.nn.Parameter(torch.tensor([.005]));before=float(old.detach().square())
    with temporary_step({'p':old},direction,rate=1.):assert float(old.detach().square())>before
    assert old.item()==pytest.approx(.005)
    assert direction_statistics(direction,{'p':torch.zeros(1)})['cosine'] is None
    with pytest.raises(ValueError):direction_statistics(direction,{'p':torch.ones(2)})
    with pytest.raises(ValueError):direction_statistics({'p':torch.tensor([float('nan')])},reference)


def test_parent_parameters_restored_even_after_interrupted_measurement():
    p=torch.nn.Parameter(torch.tensor([.25]));original=p.detach().clone()
    with pytest.raises(RuntimeError):
        with temporary_step({'p':p},{'p':torch.ones(1)}):
            assert not torch.equal(p,original)
            raise RuntimeError('Interrupted public diagnostic')
    assert torch.equal(p,original)
    with pytest.raises(ValueError):
        with temporary_step({'p':p},{'p':torch.tensor([float('inf')])}):pass
    assert torch.equal(p,original)


def test_training_loss_masks_prompt_and_computes_real_weighted_gradient():
    class Tokenizer:
        eos_token_id=9
        def apply_chat_template(self,messages,**_kwargs):return 'PREFIX'
        def encode(self,text,**_kwargs):return [1,2] if text=='PREFIX' else [5] if text=='short' else [6,7,8]
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.lora_a=torch.nn.Parameter(torch.tensor([1.]));self.calls=[]
        def forward(self,*,input_ids,labels,use_cache):
            assert labels[0,:2].tolist()==[-100,-100] and not use_cache
            count=int((labels!=-100).sum());self.calls.append((count,torch.is_grad_enabled(),self.training))
            return SimpleNamespace(loss=self.lora_a.square().mean()*count)
    model=Model().eval();before=model.lora_a.detach().clone()
    rows=[{'prompt':'public','answer':'short'},{'prompt':'public','answer':'long'}]
    result,gradient=training_statistics(model,Tokenizer(),rows,'cpu',time.time()+10,gradient=True)
    assert result=={'nll':pytest.approx(20/6),'tokens':6,'examples':2}
    assert gradient['lora_a'].item()==pytest.approx(40/6) and not model.training
    assert model.lora_a.grad is None and torch.equal(model.lora_a,before)
    result,_none=training_statistics(model,Tokenizer(),rows,'cpu',time.time()+10)
    assert model.calls[-1][1:] == (False,False)
    with pytest.raises(ValueError):training_statistics(model,Tokenizer(),rows,'cpu',time.time()+10,maximum=3)
    with pytest.raises(TimeoutError):training_statistics(None,None,rows,'cpu',time.time()-1)
