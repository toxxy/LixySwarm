"""Validate a public own-state snapshot without training or importing an identity."""
import contextlib
import json
import math
from pathlib import Path
import sys
import time


def run(request):
    from .swarm_policy import state_id
    from .swarm_foundation_protocol import protocol_id
    fields = {'action', 'model_id', 'optimizer_id', 'step', 'state_id', 'initialization_protocol_id',
        'model_file', 'optimizer_file', 'evaluation_file', 'device', 'cpu_threads', 'deadline'}
    if (not isinstance(request, dict) or set(request) != fields or request['action'] != 'own-validate'
            or request['initialization_protocol_id'] != protocol_id() or request['device'] != 'cpu'
            or state_id(request['model_id'], request['optimizer_id'], request['step']) != request['state_id']
            or type(request['cpu_threads']) is not int or not 1 <= request['cpu_threads'] <= 16
            or type(request['deadline']) not in (int, float) or not math.isfinite(request['deadline'])
            or not time.time() < request['deadline'] <= time.time() + 180):
        raise ValueError('Invalid locally selected own bootstrap validation')
    paths = [Path(request[k]) for k in ['model_file', 'optimizer_file', 'evaluation_file']]
    if any(not p.is_absolute() or p.is_symlink() or not p.is_file() for p in paths):
        raise ValueError('Own validation needs explicit regular public files')
    import torch
    from src.network.artifact_store import digest_file
    from .swarm_distributed_job import load_model
    from .swarm_foundation_job import documents, evaluate, causal_check
    from .public_math import load_snapshot
    torch.set_num_threads(request['cpu_threads']); torch.set_float32_matmul_precision('highest')
    original = [digest_file(p) for p in paths]
    if original[1][0] != request['optimizer_id'] or original[1][1] > 16 * 1024**2:
        raise ValueError('Own bootstrap optimizer differs from the selected snapshot')
    model = load_model(request)
    saved = torch.load(paths[1], map_location='cpu', weights_only=True)
    if saved['protocol_id'] != protocol_id() or saved['step'] != request['step']:
        raise ValueError('Own bootstrap optimizer belongs to another lineage or step')
    optimizer = torch.optim.AdamW(model.parameters()); optimizer.load_state_dict(saved['optimizer'])
    parameters = list(model.parameters()); states = optimizer.state_dict()['state']
    if len(states) != 175 or {int(v['step']) for v in states.values()} != {request['step']}:
        raise ValueError('Own bootstrap lost optimizer moments or progress')
    for index, item in states.items():
        for name in ['exp_avg', 'exp_avg_sq']:
            value = item[name]
            if value.shape != parameters[index].shape or not torch.isfinite(value).all():
                raise ValueError('Own bootstrap has incompatible optimizer tensors')
    sampler = saved['sampler_state']
    if sampler.dtype != torch.uint8 or sampler.ndim != 1 or not 1 <= sampler.numel() <= 16384:
        raise ValueError('Own bootstrap lost sampler continuity')
    causal = causal_check(model, 'cpu')
    if not causal['passed']: raise ValueError('Own bootstrap model is not causal')
    metrics = evaluate(model, documents(load_snapshot(paths[2], 'evaluation')), 'cpu')
    if time.time() >= request['deadline']: raise TimeoutError('Own snapshot validation expired')
    if original != [digest_file(p) for p in paths]: raise ValueError('Own snapshot validation changed its inputs')
    return {'success': True, 'state_id': request['state_id'], 'model_id': request['model_id'],
        'optimizer_id': request['optimizer_id'], 'optimizer_step': request['step'], 'optimizer_states': len(states),
        'initialization_protocol_id': protocol_id(), 'evaluation': metrics, 'causality': causal,
        'saved_state_unchanged': True, 'new_gradient_computed': False, 'cuda_initialized': torch.cuda.is_initialized()}


def run_request(request):
    with contextlib.redirect_stdout(sys.stderr): result = run(request)
    print(json.dumps(result, allow_nan=False), flush=True)
