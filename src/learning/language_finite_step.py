"""Bounded finite-loss backtracking; no model publication or dataset selection."""
import math


def safeguarded_step(parameters,direction,measure,*,protected,target,rates,
                     check=lambda:None,absolute_tolerance=1e-7,relative_tolerance=1e-6,
                     minimum_absolute_gain=1e-5,minimum_relative_gain=1e-4):
    """Commit the first measured safe step, or restore exactly on rejection/error.

    Caller supplies public TRAIN losses in one fixed execution mode. This is
    not a guarantee for held-out examples, and never authorizes publication.
    """
    import torch
    if (not parameters or set(parameters)!=set(direction) or not protected or len(set(protected))!=len(protected)
            or target in protected or not 1<=len(rates)<=16
            or any(type(r) not in (int,float) or not math.isfinite(r) or not 0<r<=1 for r in rates)
            or any(a<=b for a,b in zip(rates,rates[1:]))
            or any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in
                [absolute_tolerance,relative_tolerance,minimum_absolute_gain,minimum_relative_gain])):
        raise ValueError('Invalid bounded finite-step search')
    if any(parameters[n].shape!=direction[n].shape or not torch.isfinite(parameters[n]).all()
           or not torch.isfinite(direction[n]).all() for n in parameters):raise ValueError('Invalid finite-step parameter direction')
    original={name:p.detach().clone() for name,p in parameters.items()};committed=False
    def restore():
        with torch.no_grad():
            for name,p in parameters.items():p.copy_(original[name])
        if not all(torch.equal(p,original[n]) for n,p in parameters.items()):raise ValueError('Finite-step rollback failed')
    def tolerance(value):return max(absolute_tolerance,abs(value)*relative_tolerance)
    def measured():
        check();values=measure();check()
        if (not isinstance(values,dict) or set(values)!={target,*protected}
                or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in values.values())):
            raise ValueError('Finite-step measurement has different groups or invalid losses')
        return values
    try:
        baseline=measured();control=measured()
        if (not all(torch.equal(p,original[n]) for n,p in parameters.items())
                or any(abs(control[n]-v)>tolerance(v) for n,v in baseline.items())):
            raise ValueError('Unstable finite-step measurement without parameter updates')
        required_gain=max(minimum_absolute_gain,baseline[target]*minimum_relative_gain)
        trials=[]
        for rate in rates:
            check();restore()
            with torch.no_grad():
                for name,p in parameters.items():
                    p.add_(direction[name].to(p.device,dtype=p.dtype),alpha=-rate)
                    if not torch.isfinite(p).all():raise ValueError('Nonfinite finite-step update')
            values=measured();regressions=[n for n in protected if values[n]>baseline[n]+tolerance(baseline[n])]
            gain=baseline[target]-values[target];eligible=not regressions and gain>=required_gain
            trials.append({'rate':rate,'losses':values,'protected_regressions':regressions,
                           'target_gain':gain,'required_target_gain':required_gain,'accepted':eligible})
            if eligible:
                check();committed=True
                return {'applied':True,'accepted_rate':rate,'baseline':baseline,'no_update_control':control,
                        'trials':trials,'reason':'measured-training-groups-pass','publication_authorized':False}
        return {'applied':False,'accepted_rate':None,'baseline':baseline,'no_update_control':control,
                'trials':trials,'reason':'no-safe-training-step','publication_authorized':False}
    finally:
        if not committed:restore()


def public_training_measurement(model,tokenizer,old_rows,math_rows,device,deadline):
    """One old-data pass supplies prefix/full means and each public task kind."""
    from .language_retention_job import training_statistics
    observed=[]
    full,_=training_statistics(model,tokenizer,old_rows,device,deadline,
        observer=lambda i,loss,tokens:observed.append((i,loss,tokens)))
    def mean(rows):
        count=sum(n for _i,_loss,n in rows)
        if not count:raise ValueError('Empty protected public training group')
        return sum(loss*n for _i,loss,n in rows)/count
    result={'old_full200':full['nll'],'old_prefix64':mean(observed[:64])}
    for kind in sorted({row['kind'] for row in old_rows}):
        result['old_kind_'+kind]=mean([entry for entry in observed if old_rows[entry[0]]['kind']==kind])
    result['math_train192']=training_statistics(model,tokenizer,math_rows,device,deadline)[0]['nll']
    return result
