"""The learned memory vote must reach prediction, gradients and generation."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from src.agents.agent_base import AgentConfig
from src.agents.dolphin_agent import DolphinConfig
from src.matriarca.matriarca import Matriarca, MatriarcaConfig
from src.swarm.orchestrator import LixySwarm, SwarmConfig
from src.swarm.runtime_session import RuntimeSession


def public_swarm(tmp_path, *, memory=True):
    """Small real networks; all memory is synthetic and in a fresh directory."""
    cfg = SwarmConfig(
        n_agents=2, swarm_rounds=2, feromon_dim=8, echolocation_dim=8,
        agent_configs=[AgentConfig(block_size=16, vocab_size=32, n_layer=1,
            n_head=2, n_embd=16, feromon_dim=8, identity_dim=8,
            agent_id=i, n_agents=2) for i in range(2)],
        dolphin_config=DolphinConfig(vocab_size=32, n_embd=16, feromon_dim=8,
            identity_dim=8, echo_dim=8, sleep_dim=8, dropout=0),
        matriarca_config=MatriarcaConfig(embd_dim=12, infrasound_dim=8,
            n_heads=2, n_layers=1, max_memories=64, dropout=0,
            memory_path=str(tmp_path / 'public-memory.json'),
            checkpoint_path=str(tmp_path / 'public-encoder.pt')),
    )
    model = LixySwarm(cfg, load_matriarca=False, load_runtime_state=False).cpu()
    if memory:
        model.matriarca = Matriarca(cfg.matriarca_config, device='cpu')
        # More than the retrieval limit of 32, to exercise the actual top-k path.
        for i in range(34):
            model.matriarca.bank.add(torch.randn(12), f'public synthetic memory {i}')
    return model


def manual_aggregation(model, logits, infrasound):
    scores = []
    n_embd = model.config.agent_configs[0].n_embd
    for head, values in zip(model.confidence_heads, logits):
        rep = values.mean(dim=1)
        rep = rep[:, :n_embd] if rep.shape[-1] >= n_embd else F.pad(rep, (0, n_embd-rep.shape[-1]))
        scores.append(head(rep))
    scores = torch.stack(scores, dim=1)
    if infrasound is not None:
        scores = .8 * scores + .2 * model.matriarca_conf_proj(infrasound).unsqueeze(-1)
    weights = scores.softmax(dim=1)
    return (torch.stack(logits, dim=1) * weights.unsqueeze(-1)).sum(dim=1), weights


@pytest.mark.parametrize('memory', [False, True])
def test_shared_selector_preserves_forward_values_and_checkpoint_keys(tmp_path, memory):
    torch.manual_seed(151)
    model = public_swarm(tmp_path, memory=False).eval()
    keys = set(model.state_dict())
    logits = [torch.randn(2, 4, 32) for _ in model.agents]
    infra = torch.randn(2, 8) if memory else None
    with torch.no_grad():
        actual = model.aggregate_agent_logits(logits, infra)
        expected = manual_aggregation(model, logits, infra)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert set(model.state_dict()) == keys
    assert actual[1].shape == (2, 2, 1)
    assert torch.allclose(actual[1].sum(dim=1), torch.ones(2, 1))


@pytest.mark.parametrize('training', [False, True])
def test_collective_loss_trains_memory_readout_without_mutating_memory(tmp_path, training):
    torch.manual_seed(152)
    model = public_swarm(tmp_path).train(training)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith('matriarca_conf_proj.'))
    encoder_before = {k: v.clone() for k, v in model.matriarca.model.state_dict().items()}
    memory_before = deepcopy(model.matriarca.bank.metadata)
    embeddings_before = model.matriarca.bank.embeddings.clone()
    model_before = {k: v.clone() for k, v in model.state_dict().items()}
    x = torch.tensor([[3, 6, 9, 12]])
    y = torch.tensor([[7, 7, 7, 7]])
    options = dict(targets=y, use_memory=True, store_memory=False,
        update_runtime_state=False, update_memory_importance=False)
    logits_before, loss, _ = model(x, **options)
    loss.backward()
    assert sum(p.grad.abs().sum() for p in model.matriarca_conf_proj.parameters()) > 0
    assert all(p.grad is None for p in model.matriarca.model.parameters())
    optimizer = torch.optim.SGD(model.matriarca_conf_proj.parameters(), lr=.1)
    optimizer.step()
    logits_after, next_loss, _ = model(x, **options)
    assert next_loss < loss.detach()
    assert not torch.equal(logits_before, logits_after)
    assert any(not torch.equal(v, model_before[k]) for k, v in model.state_dict().items()
               if k.startswith('matriarca_conf_proj.'))
    assert all(torch.equal(v, model_before[k]) for k, v in model.state_dict().items()
               if not k.startswith('matriarca_conf_proj.'))
    assert model.matriarca.bank.metadata == memory_before
    assert torch.equal(model.matriarca.bank.embeddings, embeddings_before)
    assert all(torch.equal(v, encoder_before[k]) for k, v in model.matriarca.model.state_dict().items())


def test_memory_disabled_does_not_read_memory_or_train_its_vote(tmp_path, monkeypatch):
    torch.manual_seed(153)
    model = public_swarm(tmp_path).train()
    def forbidden(*args, **kwargs):
        raise AssertionError('Memory-disabled prediction accessed memory')
    monkeypatch.setattr(model.matriarca, 'emit_infrasound', forbidden)
    _, loss, _ = model(torch.tensor([[2, 3]]), targets=torch.tensor([[3, 4]]),
        use_memory=False, store_memory=False, update_runtime_state=False,
        update_memory_importance=False)
    loss.backward()
    assert all(p.grad is None for p in model.matriarca_conf_proj.parameters())


def test_retrieval_is_independent_for_each_batch_member(tmp_path):
    torch.manual_seed(154)
    model = public_swarm(tmp_path).eval()
    first, second = torch.randn(1, 8), torch.randn(1, 8)
    before = deepcopy(model.matriarca.bank.metadata)
    separate = torch.cat([model._get_infrasound(x, update_importance=False)
                          for x in [first, second]])
    together = model._get_infrasound(torch.cat([first, second]), update_importance=False)
    reversed_batch = model._get_infrasound(torch.cat([second, first]), update_importance=False)
    assert torch.equal(separate, together)
    assert torch.equal(together.flip(0), reversed_batch)
    assert not torch.equal(together[0], together[1])
    assert torch.equal(model._get_infrasound(first[0], update_importance=False), together[:1])
    assert model.matriarca.bank.metadata == before
    model._get_infrasound(first, update_importance=True)
    assert model.matriarca.bank.metadata != before


class SyntheticEncoding:
    _special_tokens = {}
    def encode(self, text):
        return [ord(char) % 32 for char in text]
    def decode(self, tokens):
        return ''.join(chr(65 + token) for token in tokens)


@pytest.mark.parametrize('memory', [False, True])
def test_runtime_uses_shared_learned_selection_between_refreshes(tmp_path, monkeypatch, memory):
    torch.manual_seed(155)
    model = public_swarm(tmp_path).eval()
    session = RuntimeSession(model, SyntheticEncoding(), device='cpu', max_new_tokens=5, verbose=False)
    session.FEROMON_REFRESH_INTERVAL = 2
    # A hostile lexical profile cannot override the learned weights.
    monkeypatch.setattr(session._role_adapter, 'profile_query', lambda _: SimpleNamespace(
        temperature=.8, task_type='factual', role_weights=[float('nan')]))
    def forbidden(*args, **kwargs):
        raise AssertionError('Generation used lexical agent weights')
    monkeypatch.setattr(session._role_adapter, 'get_weights_for_query', forbidden)
    if not memory:
        monkeypatch.setattr(model.matriarca, 'emit_infrasound', forbidden)
    original = model.aggregate_agent_logits
    calls = []
    def checked(logits, infrasound=None):
        actual = original(logits, infrasound)
        expected = manual_aggregation(model, logits, infrasound)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
        assert (infrasound is not None) == memory
        calls.append(actual[0].clone())
        return actual
    monkeypatch.setattr(model, 'aggregate_agent_logits', checked)
    samples = []
    def sample(logits, **_):
        # Each generated token must use the common aggregation's actual logits.
        assert torch.equal(logits, calls[-1])
        samples.append(logits.clone())
        return logits[:, -1].argmax(dim=-1, keepdim=True)
    monkeypatch.setattr('src.swarm.runtime_session.sample_token', sample)
    metadata = deepcopy(model.matriarca.bank.metadata)
    response = session.turn('code', greedy=True, use_memory=memory, store_memory=False,
        record_history=False, update_runtime_state=False)
    assert len(response) == 4 and len(samples) == 4
    assert len(calls) == 7  # warm-up + two refreshes + four generated tokens
    assert model.matriarca.bank.metadata == metadata


def test_optimizer_and_learned_vote_resume_exactly_after_checkpoint_reload(tmp_path):
    torch.manual_seed(156)
    model = public_swarm(tmp_path, memory=False).train()
    optimizer = torch.optim.AdamW(model.matriarca_conf_proj.parameters(), lr=.01)
    logits = [torch.randn(2, 3, 32) for _ in model.agents]
    infra = torch.randn(2, 8)
    targets = torch.tensor([[2, 3, 4], [5, 6, 7]])
    def step(net, opt):
        opt.zero_grad(set_to_none=True)
        aggregate, _ = net.aggregate_agent_logits(logits, infra)
        F.cross_entropy(aggregate.flatten(0, 1), targets.flatten()).backward()
        opt.step()
    step(model, optimizer)
    path = tmp_path / 'swarm-and-optimizer.pt'
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, path)
    restored = public_swarm(tmp_path, memory=False).train()
    restored_optimizer = torch.optim.AdamW(restored.matriarca_conf_proj.parameters(), lr=.01)
    saved = torch.load(path, weights_only=True)
    restored.load_state_dict(saved['model'], strict=True)
    restored_optimizer.load_state_dict(saved['optimizer'])
    assert all(torch.equal(a, b) for a, b in zip(model.aggregate_agent_logits(logits, infra),
                                               restored.aggregate_agent_logits(logits, infra)))
    step(model, optimizer)
    step(restored, restored_optimizer)
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())


def test_selector_rejects_missing_agents_and_cross_request_broadcast(tmp_path):
    model = public_swarm(tmp_path, memory=False)
    logits = [torch.randn(2, 3, 32) for _ in model.agents]
    with pytest.raises(ValueError, match='Every agent'):
        model.aggregate_agent_logits(logits[:1])
    with pytest.raises(ValueError, match='per request'):
        model.aggregate_agent_logits(logits, torch.randn(1, 8))
