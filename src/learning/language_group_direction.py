"""Bounded projection against separate public TRAIN gradients; tensor-only."""
import math


def project_groups(direction,references,*,maximum_sweeps=512,relative_tolerance=1e-7,relative_margin=0.):
    """Project in the small dual using normalized constraint coordinate ascent.

    The retained gradient for each group constrains its own first-order loss.
    A finite-loss check is still mandatory; this never authorizes a model.
    """
    import torch
    if (not direction or not references or len(references)>16 or type(maximum_sweeps) is not int
            or not 1<=maximum_sweeps<=1024 or not math.isfinite(relative_tolerance) or not 0<relative_tolerance<=1e-5
            or type(relative_margin) not in (int,float) or not math.isfinite(relative_margin) or not 0<=relative_margin<=.1
            or any(set(row)!=set(direction) for row in references.values())):
        raise ValueError('Invalid bounded group projection')
    keys=sorted(direction)
    def vector(row):
        if any(row[n].shape!=direction[n].shape or not torch.isfinite(row[n]).all() for n in keys):
            raise ValueError('Invalid public group gradient shape or value')
        return torch.cat([row[n].detach().double().cpu().reshape(-1) for n in keys])
    source=vector(direction);norm=float(torch.linalg.vector_norm(source));groups=sorted(references)
    raw=torch.stack([vector(references[name]) for name in groups]);norms=torch.linalg.vector_norm(raw,dim=1)
    if not math.isfinite(norm) or not torch.isfinite(norms).all():raise ValueError('Nonfinite public group gradient norm')
    rows=raw/norms.clamp_min(torch.finfo(torch.float64).tiny).unsqueeze(1)
    gram=rows@rows.t();dots=rows@source;dual=torch.zeros(len(groups),dtype=torch.float64)
    floors=(norms>0).double()*(relative_margin*norm)
    # Hildreth/Dykstra halfspace projection in the dual: keep every coefficient
    # nonnegative. Normalization keeps tiny well-learned groups represented.
    for sweep in range(maximum_sweeps):
        for index in range(len(groups)):
            diagonal=float(gram[index,index])
            if not diagonal:continue
            residual=float(dots[index]-floors[index]+gram[index]@dual)
            dual[index]=max(0.,float(dual[index])-residual/diagonal)
        residuals=dots-floors+gram@dual
        kkt=float((dual-(dual-residuals).clamp_min(0)).abs().max())
        if kkt<=relative_tolerance*max(norm,1e-30):break
    projected=source+rows.t()@dual
    result={};offset=0
    for name in keys:
        count=direction[name].numel()
        result[name]=projected[offset:offset+count].reshape(direction[name].shape).to(direction[name].dtype);offset+=count
    # Check the actual returned representation, not only its float64 solution.
    actual=vector(result);after=rows@actual;threshold=relative_tolerance*max(norm,1e-30)
    converged=bool(torch.isfinite(actual).all() and float((after-floors).min())>=-threshold and kkt<=threshold)
    report={'converged':converged,'sweeps':sweep+1,'maximum_sweeps':maximum_sweeps,
        'source_norm':norm,'projected_norm':float(torch.linalg.vector_norm(actual)),
        'relative_direction_change':float(torch.linalg.vector_norm(actual-source))/norm if norm else 0.,
        'normalized_dot_tolerance':threshold,
        'dual_kkt_residual':kkt,
        'groups':{name:{'reference_norm':float(norms[i]),'normalized_dot_before':float(dots[i]),
                       'normalized_dot_after':float(after[i]),'dual_coefficient':float(dual[i])} for i,name in enumerate(groups)},
        'scope':'Training-gradient geometry only; finite-loss and held-out checks remain mandatory'}
    if relative_margin:
        report['relative_margin']=relative_margin
        for i,name in enumerate(groups):report['groups'][name]['minimum_normalized_dot']=float(floors[i])
    return result,report
