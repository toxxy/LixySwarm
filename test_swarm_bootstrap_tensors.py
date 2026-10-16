"""Real own architecture and AdamW continuity at the cold-validation boundary."""
import time

import pytest


def test_bootstrap_checks_causality_moments_and_never_updates_received_weights(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import save_file
    from src.learning import swarm_bootstrap_job as job
    from src.learning import public_math
    from src.learning.swarm_foundation import make_foundation_model, predict
    from src.learning.swarm_foundation_protocol import protocol_id
    from src.learning.swarm_policy import state_id
    from src.network.artifact_store import digest_file
    torch.set_num_threads(2)
    model = make_foundation_model().train(); optimizer = torch.optim.AdamW(model.parameters())
    inputs = torch.arange(8).repeat(2, 1); predict(model, inputs, inputs)[1].backward(); optimizer.step()
    weights = tmp_path/'own.safetensors'; opt = tmp_path/'own-optimizer.pt'; evaluation = tmp_path/'evaluation'
    save_file({k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}, str(weights),
        metadata={'format': 'lixyswarm-own-causal-byte-v1', 'protocol_id': protocol_id()})
    saved = {'protocol_id': protocol_id(), 'step': 1, 'optimizer': optimizer.state_dict(),
        'sampler_state': torch.Generator().get_state()}
    torch.save(saved, opt); evaluation.write_text('public test fixture')
    monkeypatch.setattr(public_math, 'load_snapshot', lambda *_a: {'rows': [{'question': 'Public question?', 'answer': 'Public answer.'}]})
    mid, oid = digest_file(weights)[0], digest_file(opt)[0]
    request = {'action': 'own-validate', 'model_id': mid, 'optimizer_id': oid, 'step': 1,
        'state_id': state_id(mid, oid, 1), 'initialization_protocol_id': protocol_id(),
        'model_file': str(weights), 'optimizer_file': str(opt), 'evaluation_file': str(evaluation),
        'device': 'cpu', 'cpu_threads': 2, 'deadline': time.time()+120}
    result = job.run(request)
    assert result['success'] and result['causality']['passed'] and result['optimizer_states'] == 175
    assert not result['new_gradient_computed'] and result['saved_state_unchanged'] and result['optimizer_step'] == 1
    assert digest_file(weights)[0] == mid and digest_file(opt)[0] == oid
    saved['optimizer']['state'].pop(next(iter(saved['optimizer']['state'])))
    torch.save(saved, opt); oid = digest_file(opt)[0]
    with pytest.raises(ValueError, match='moments'):
        job.run({**request, 'optimizer_id': oid, 'state_id': state_id(mid, oid, 1)})
