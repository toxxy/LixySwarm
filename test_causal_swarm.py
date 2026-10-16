"""Future suffixes must never help predict earlier next tokens."""
import copy

import pytest
import torch
import torch.nn.functional as F

from src.agents.agent_base import AgentConfig
from src.agents.dolphin_agent import DolphinConfig
from src.learning.swarm_foundation import predict
from src.swarm.orchestrator import LixySwarm, SwarmConfig


def causal_swarm():
    return LixySwarm(SwarmConfig(n_agents=2, feromon_dim=8, echolocation_dim=8,
        swarm_rounds=2, causal_signals=True,
        agent_configs=[AgentConfig(block_size=16, vocab_size=32, n_layer=1,
            n_head=2, n_embd=16, feromon_dim=8, identity_dim=8,
            agent_id=i, n_agents=2) for i in range(2)],
        dolphin_config=DolphinConfig(vocab_size=32, n_embd=16, feromon_dim=8,
            identity_dim=8, echo_dim=8, sleep_dim=8, dropout=0)),
        load_matriarca=False, load_runtime_state=False).cpu()


@pytest.mark.parametrize('training', [False, True])
@pytest.mark.parametrize('dolphins', [1, 2])
def test_future_suffix_and_other_requests_cannot_change_prefix_predictions(training, dolphins):
    torch.manual_seed(182)
    model = causal_swarm().train(training)
    model.dolphin.device = 'cpu'
    if dolphins == 2:
        model.dolphin.scale_to_network(2)
    x = torch.randint(0, 32, (2, 8))
    changed = x.clone()
    changed[0, 4:] = (changed[0, 4:] + 7) % 32
    changed[1] = (changed[1] + 9) % 32
    before, _, signal = predict(model, x)
    after, _, new_signal = predict(model, changed)
    torch.testing.assert_close(before[0, :4], after[0, :4], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(signal[0, :4], new_signal[0, :4], atol=1e-6, rtol=1e-5)
    assert before.shape == (2, 8, 32)


def test_parallel_teacher_forcing_matches_actual_generation_from_every_prefix():
    torch.manual_seed(183)
    model = causal_swarm().eval()
    x = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        full, _, _ = predict(model, x)
        for length in range(1, 9):
            prefix, _, _ = predict(model, x[:, :length])
            torch.testing.assert_close(full[:, length-1], prefix[:, -1], atol=1e-6, rtol=1e-5)
        labels = torch.randint(0, 32, x.shape)
        labelled, loss, _ = predict(model, x, labels)
        assert torch.equal(full, labelled)
        torch.testing.assert_close(loss, F.cross_entropy(full.flatten(0, 1), labels.flatten()))


def test_collective_learning_reaches_pings_signals_agents_and_selector_and_resumes(tmp_path):
    torch.manual_seed(184)
    model = causal_swarm().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    x = torch.randint(0, 32, (2, 8)); y = torch.randint(0, 32, x.shape)

    def step(net, opt):
        opt.zero_grad(set_to_none=True)
        _, loss, _ = predict(net, x, y)
        loss.backward()
        opt.step()

    step(model, optimizer)
    groups = [model.feromon_pool, *model.agents, *model.confidence_heads]
    echo = model.dolphin.primary.dolphin.echolocation
    groups += [getattr(echo, 'ping_'+name) for name in model.config.dolphin_config.ping_names]
    for module in groups:
        assert sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None) > 0
    path = tmp_path/'own-weights-and-optimizer.pt'
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, path)
    restored = causal_swarm().train()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.001)
    saved = torch.load(path, weights_only=True)
    restored.load_state_dict(saved['model'], strict=True)
    restored_optimizer.load_state_dict(saved['optimizer'])
    step(model, optimizer); step(restored, restored_optimizer)
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())


def test_causal_lineage_rejects_implicit_private_runtime_and_unversioned_memory():
    config = copy.deepcopy(causal_swarm().config)
    with pytest.raises(ValueError, match='clean initialization'):
        LixySwarm(config)
    model = causal_swarm()
    with pytest.raises(ValueError, match='memory and runtime'):
        model(torch.ones((1, 4), dtype=torch.long))
    with pytest.raises(ValueError, match='prefix memory protocol'):
        model.aggregate_agent_logits([torch.randn(1, 4, 32) for _ in model.agents],
                                     torch.randn(1, 64), tokenwise=True)


def test_own_foundation_initialization_is_reproducible_and_preserves_callers_rng():
    from src.learning.swarm_foundation import make_foundation_model
    torch.manual_seed(185)
    before = torch.get_rng_state().clone()
    a = make_foundation_model(seed=37)
    assert torch.equal(before, torch.get_rng_state())
    b = make_foundation_model(seed=37)
    assert a.matriarca is None and a.config.causal_signals
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())


def test_cpu_foundation_does_not_initialize_cuda(monkeypatch):
    from src.learning.swarm_foundation import make_foundation_model
    def forbidden():
        raise AssertionError('CPU foundation initialized CUDA')
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    assert all(p.device.type == 'cpu' for p in make_foundation_model().parameters())


def test_own_training_receipt_and_cpu_optimizer_recovery_end_to_end(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import time
    from src.learning import swarm_foundation_job as job, swarm_foundation_audit as reader
    from src.learning.swarm_foundation_protocol import definition, protocol_id
    # A bounded fixture exercises genuine backward steps and serialization,
    # not the physical experiment's quality or 1024-step acceptance claim.
    protocol = definition()
    protocol.update(steps=2, batch_size=2, block_size=8,
                    evaluation_maximum_bytes_per_document=16)
    monkeypatch.setattr(job, 'definition', lambda: protocol)
    monkeypatch.setattr(reader, 'definition', lambda: protocol)
    def snapshot(_path, split):
        return {'rows': [{'question': 'train question' if split == 'train' else 'unseen question',
                          'answer': 'a public answer'}]}
    monkeypatch.setattr(job, 'load_snapshot', snapshot)
    monkeypatch.setattr(reader, 'load_snapshot', snapshot)
    train = tmp_path/'train.json'; train.write_text('fixture')
    evaluation = tmp_path/'evaluation.json'; evaluation.write_text('fixture')
    output = tmp_path/'engine'
    receipt = job.run({'action': protocol['action'], 'protocol_id': protocol_id(),
        'training_file': str(train), 'evaluation_file': str(evaluation),
        'output_directory': str(output), 'device': 'cpu', 'cpu_threads': 2,
        'gpu_fraction': 1., 'deadline': time.time()+60})
    args = SimpleNamespace(weights=output/'model.safetensors', optimizer=output/'optimizer.pt',
        report=output/'diagnostic.json', report_id=receipt['report_id'],
        training_file=train, evaluation_file=evaluation)
    recovered = reader.audit(args)
    assert recovered['saved_optimizer_step'] == 2
    assert recovered['resumed_optimizer_step'] == 3
    assert recovered['identical_resumed_branches']
    assert recovered['saved_weights_unchanged'] and not recovered['model_promoted']
    args.report_id = '0'*64
    with pytest.raises(ValueError, match='source report changed'):
        reader.audit(args)
