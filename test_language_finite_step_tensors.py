"""Real finite losses, backtracking, group retention and exact rollback."""
import time
from types import SimpleNamespace

import pytest
torch=pytest.importorskip('torch')

from src.learning.language_finite_step import safeguarded_step,public_training_measurement


def test_rejected_step_restored_before_smaller_step_and_only_safe_step_commits():
    p=torch.nn.Parameter(torch.tensor([.5]));seen=[]
    def measure():
        value=p.detach().item();seen.append(value)
        return {'old':(value-.25)**2,'math':(value+1)**2}
    result=safeguarded_step({'p':p},{'p':torch.ones(1)},measure,protected=['old'],target='math',rates=[1.,.5])
    assert seen==[.5,.5,-.5,0.] and p.item()==0
    assert result['applied'] and result['accepted_rate']==.5 and not result['publication_authorized']
    assert result['trials'][0]['protected_regressions']==['old'] and result['trials'][1]['accepted']


def test_overall_gain_cannot_hide_regression_in_one_protected_kind():
    p=torch.nn.Parameter(torch.tensor([.5]));before=p.detach().clone()
    def measure():
        value=p.detach().item()
        return {'old_mean':1+value,'old_kind_a':1-value,'math':1+value}
    result=safeguarded_step({'p':p},{'p':torch.ones(1)},measure,
        protected=['old_mean','old_kind_a'],target='math',rates=[.1,.01])
    assert not result['applied'] and torch.equal(p,before)
    assert all(row['protected_regressions']==['old_kind_a'] for row in result['trials'])


@pytest.mark.parametrize('failure',['measurement','nan','group','cancel','nonfinite_update'])
def test_failure_during_temporary_update_restores_exact_parent(failure):
    p=torch.nn.Parameter(torch.tensor([.5]));before=p.detach().clone();calls=0
    def measure():
        nonlocal calls
        calls+=1
        if calls==3:
            if failure=='measurement':raise RuntimeError('Interrupted loss measurement')
            if failure=='nan':return {'old':float('nan'),'math':1.}
            if failure=='group':return {'other':1.,'math':1.}
        return {'old':float(p.detach().square()),'math':float((p.detach()+1).square())}
    def check():
        if failure=='cancel' and p.item()!=.5:raise RuntimeError('User cancellation')
    direction=torch.tensor([float('inf')]) if failure=='nonfinite_update' else torch.ones(1)
    with pytest.raises((RuntimeError,ValueError)):
        safeguarded_step({'p':p},{'p':direction},measure,protected=['old'],target='math',rates=[.25],check=check)
    assert torch.equal(p,before)


def test_no_update_noise_refuses_search_and_rolls_back():
    p=torch.nn.Parameter(torch.ones(1));calls=0
    def measure():
        nonlocal calls
        calls+=1
        return {'old':1.+.01*calls,'math':2.}
    with pytest.raises(ValueError,match='without parameter updates'):
        safeguarded_step({'p':p},{'p':torch.ones(1)},measure,protected=['old'],target='math',rates=[.1])
    assert calls==2 and p.item()==1


@pytest.mark.parametrize('rates',[[],[.1,.2],[.1,.1],[True],[float('nan')],[0.],[-1.]])
def test_invalid_search_is_rejected_before_measurement(rates):
    with pytest.raises(ValueError):
        safeguarded_step({'p':torch.ones(1)},{'p':torch.ones(1)},lambda:pytest.fail('measured invalid search'),
            protected=['old'],target='math',rates=rates)


def test_training_groups_use_answer_token_weighting_and_identical_mode():
    class Tokenizer:
        eos_token_id=9
        def apply_chat_template(self,*_a,**_kw):return 'prefix'
        def encode(self,text,**_kw):return [1] if text in {'prefix','short'} else [1,2,3]
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.lora_a=torch.nn.Parameter(torch.ones(1));self.calls=0
        def forward(self,*,labels,**_kw):
            assert self.training and torch.is_grad_enabled()
            self.calls+=1
            return SimpleNamespace(loss=self.lora_a.square().mean()*int((labels!=-100).sum()))
    model=Model().eval()
    old=[{'prompt':'public','answer':'short','kind':'a'} for _ in range(64)]+[
        {'prompt':'public','answer':'long','kind':'b'}]
    result=public_training_measurement(model,Tokenizer(),old,[{'prompt':'math','answer':'long'}],'cpu',time.time()+10)
    assert result=={'old_prefix64':2.,'old_full200':pytest.approx((64*4+16)/(64*2+4)),
                    'old_kind_a':2.,'old_kind_b':4.,'math_train192':4.}
    assert not model.training and model.calls==66 and model.lora_a.grad is None
