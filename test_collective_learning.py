"""Collective predictions must train the routing heads and agent signals."""
import torch
import torch.nn.functional as F

from src.agents.agent_base import AgentConfig
from src.agents.dolphin_agent import DolphinConfig
from src.swarm.orchestrator import LixySwarm, SwarmConfig


def tiny_swarm():
    config = SwarmConfig(
        n_agents=2, swarm_rounds=2, feromon_dim=8, echolocation_dim=8,
        agent_configs=[AgentConfig(block_size=16, vocab_size=32, n_layer=1,
            n_head=2, n_embd=16, feromon_dim=8, identity_dim=8,
            agent_id=i, n_agents=2) for i in range(2)],
        dolphin_config=DolphinConfig(vocab_size=32, n_embd=16, feromon_dim=8,
                                    identity_dim=8, echo_dim=8, sleep_dim=8, dropout=0),
    )
    return LixySwarm(config, load_matriarca=False, load_runtime_state=False)


def test_collective_loss_reaches_every_confidence_head_and_agent():
    torch.manual_seed(47)
    model = tiny_swarm()
    model.train()
    x = torch.randint(0, 32, (2, 8))
    y = torch.randint(0, 32, (2, 8))
    _, loss, _ = model(x, targets=y, store_memory=False, use_memory=False,
                       update_runtime_state=False, update_memory_importance=False)
    loss.backward()
    for head in model.confidence_heads:
        assert sum(p.grad.abs().sum().item() for p in head.parameters() if p.grad is not None) > 0
    for agent in model.agents:
        assert sum(p.grad.abs().sum().item() for p in agent.parameters() if p.grad is not None) > 0


def test_diversity_penalty_has_gradient_and_stops_below_margin():
    a = torch.tensor([[1.0, 0.2]], requires_grad=True)
    b = torch.tensor([[1.0, 0.4]], requires_grad=True)
    penalty = LixySwarm.pheromone_diversity_loss([a, b], 0.7)
    penalty.backward()
    assert penalty > 0 and a.grad.abs().sum() > 0 and b.grad.abs().sum() > 0
    assert LixySwarm.pheromone_diversity_loss([a.detach(), -a.detach()], 0.7) == 0


def test_evaluation_loss_describes_the_actual_collective_prediction():
    torch.manual_seed(48)
    model = tiny_swarm().eval()
    x = torch.randint(0, 32, (2, 8)); y = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        logits, loss, _ = model(x, targets=y, store_memory=False, use_memory=False,
                               update_runtime_state=False, update_memory_importance=False)
    expected = F.cross_entropy(logits.reshape(-1, 32), y.reshape(-1))
    assert torch.allclose(loss, expected)
