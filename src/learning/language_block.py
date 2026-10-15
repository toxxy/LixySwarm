"""Bounded local refinement of a fixed shared direction, with a parent ceiling."""


def protected_block(parameters,source,measure,prepare,*,maximum_steps,guard_options,check=lambda:None):
    """Keep the last safe partial block; any exception restores the initial parent.

    This reuses a fixed signed direction. Microsteps are not additional remote
    contributions or autonomous learning cycles. No quality evaluation occurs.
    """
    import torch
    from .language_finite_step import safeguarded_step
    if type(maximum_steps) is not int or not 1<=maximum_steps<=8:raise ValueError('Invalid bounded training block')
    originals={name:p.detach().clone() for name,p in parameters.items()};completed=False;ceiling=None;steps=[]
    try:
        for index in range(maximum_steps):
            check();direction,projection=prepare(source,index);check()
            if not projection['converged']:
                steps.append({'index':index,'projection':projection,'guard':None});break
            guard=safeguarded_step(parameters,direction,measure,check=check,protected_ceiling=ceiling,**guard_options)
            if ceiling is None:ceiling={name:guard['baseline'][name] for name in guard_options['protected']}
            steps.append({'index':index,'projection':projection,'guard':guard})
            if not guard['applied']:break
        check()
        report={'maximum_steps':maximum_steps,'accepted_steps':sum(bool(s['guard'] and s['guard']['applied']) for s in steps),
            'steps':steps,'initial_parent_protected_ceiling':ceiling,'fixed_shared_direction_reused':True,
            'new_remote_contributions':0,'native_learning_cycles_added':0,'publication_authorized':False}
        completed=True;return report
    finally:
        if not completed:
            with torch.no_grad():
                for name,p in parameters.items():p.copy_(originals[name])
            if not all(torch.equal(p,originals[n]) for n,p in parameters.items()):raise ValueError('Training block rollback failed')
