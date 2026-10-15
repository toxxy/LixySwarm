"""Microstep retention cannot drift past the original parent; errors roll back."""
import pytest
torch=pytest.importorskip('torch')
from src.learning.language_block import protected_block


def options():
    return {'protected':['old'],'target':'math','rates':[1.],
            'absolute_tolerance':1e-7,'relative_tolerance':0.,'minimum_absolute_gain':.1,'minimum_relative_gain':0.}


def test_parent_ceiling_prevents_small_per_step_regressions_accumulating():
    p=torch.nn.Parameter(torch.zeros(1,dtype=torch.float64))
    def measure():
        v=float(p.detach());return {'old':1+v*5e-8,'math':10-v}
    report=protected_block({'p':p},{'p':-torch.ones(1,dtype=torch.float64)},measure,
        lambda source,_i:(source,{'converged':True}),maximum_steps=4,guard_options=options())
    assert report['accepted_steps']==2 and p.item()==2
    assert report['initial_parent_protected_ceiling']=={'old':1.}
    assert report['steps'][-1]['guard']['trials'][0]['protected_regressions']==['old']
    assert report['new_remote_contributions']==report['native_learning_cycles_added']==0


def test_error_after_a_safe_step_restores_the_initial_parent():
    p=torch.nn.Parameter(torch.zeros(1));original=p.detach().clone()
    def prepare(source,index):
        if index==1:raise RuntimeError('Cancelled between microsteps')
        return source,{'converged':True}
    with pytest.raises(RuntimeError):
        protected_block({'p':p},{'p':-torch.ones(1)},lambda:{'old':1.,'math':10-float(p.detach())},prepare,
                        maximum_steps=4,guard_options=options())
    assert torch.equal(p,original)


def test_nonconvergent_projection_preserves_last_safe_partial_block():
    p=torch.nn.Parameter(torch.zeros(1))
    report=protected_block({'p':p},{'p':-torch.ones(1)},lambda:{'old':1.,'math':10-float(p.detach())},
        lambda source,index:(source,{'converged':index==0}),maximum_steps=4,guard_options=options())
    assert report['accepted_steps']==1 and p.item()==1 and report['steps'][-1]['guard'] is None
