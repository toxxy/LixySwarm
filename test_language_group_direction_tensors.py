"""Multi-group projection geometry checked against analytic constrained cases."""
import pytest
torch=pytest.importorskip('torch')

from src.learning.language_group_direction import project_groups


def test_projection_does_not_stop_at_feasible_but_nonoptimal_single_sweep():
    source={'p':torch.tensor([-1.,-1.])}
    references={'a':{'p':torch.tensor([1.,0.])},'b':{'p':torch.tensor([.5,1.])}}
    first,limited=project_groups(source,references,maximum_sweeps=1)
    assert not limited['converged'] and first['p'][0]>0
    projected,report=project_groups(source,references)
    assert report['converged'] and report['sweeps']>1
    assert torch.allclose(projected['p'],torch.zeros(2),atol=1e-6)
    assert torch.equal(source['p'],torch.tensor([-1.,-1.]))
    assert report['dual_kkt_residual']<=report['normalized_dot_tolerance']


def test_group_scaling_does_not_erase_small_well_learned_reference():
    source={'p':torch.tensor([-1.,2.])}
    projected,report=project_groups(source,{'tiny':{'p':torch.tensor([1e-20,0.])},
        'large':{'p':torch.tensor([0.,1e20])},'zero':{'p':torch.zeros(2)}})
    assert report['converged'] and torch.allclose(projected['p'],torch.tensor([0.,2.]))
    assert all(row['normalized_dot_after']>=-report['normalized_dot_tolerance'] for row in report['groups'].values())


def test_opposite_constraints_produce_zero_component_instead_of_relaxing_groups():
    result,report=project_groups({'p':torch.tensor([1.,3.])},
        {'left':{'p':torch.tensor([1.,0.])},'right':{'p':torch.tensor([-1.,0.])}})
    assert report['converged'] and torch.equal(result['p'],torch.tensor([0.,3.]))


@pytest.mark.parametrize('bad',[{}, {'a':{'other':torch.ones(2)}}, {'a':{'p':torch.ones(3)}},
    {'a':{'p':torch.tensor([float('nan'),0.])}}])
def test_invalid_group_constraints_are_rejected(bad):
    with pytest.raises(ValueError):project_groups({'p':torch.ones(2)},bad)
