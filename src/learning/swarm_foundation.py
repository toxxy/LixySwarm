"""Own byte-level foundation, with prefix-causal signals and no imported weights.

This is a new lineage. The historical byte reference and pretrained language
adapters keep their identities and are not silently converted into this model.
"""
from __future__ import annotations

FOUNDATION_CONFIG = {
    'family': 'lixyswarm-own-causal-byte-v1', 'vocab_size': 256,
    'n_agents': 3, 'n_embd': 128, 'n_head': 4, 'n_layer': 2,
    'block_size': 128, 'feromon_dim': 32, 'echo_dim': 32,
    'identity_dim': 16, 'sleep_dim': 32, 'swarm_rounds': 2,
    'causal_signals': True,
}


def make_foundation_model(*, seed=20261015, device='cpu'):
    """Construct every learned parameter locally; never search for checkpoints."""
    import torch
    from src.agents.agent_base import AgentConfig
    from src.agents.dolphin_agent import DolphinConfig
    from src.swarm.orchestrator import LixySwarm, SwarmConfig

    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Foundation initialization needs a reproducible integer seed')
    c = FOUNDATION_CONFIG
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        config = SwarmConfig(n_agents=c['n_agents'], feromon_dim=c['feromon_dim'],
            echolocation_dim=c['echo_dim'], swarm_rounds=c['swarm_rounds'],
            causal_signals=True,
            agent_configs=[AgentConfig(vocab_size=256, block_size=c['block_size'],
                n_embd=c['n_embd'], n_layer=c['n_layer'], n_head=c['n_head'],
                identity_dim=c['identity_dim'], feromon_dim=c['feromon_dim'],
                agent_id=i, n_agents=c['n_agents'], dropout=0)
                for i in range(c['n_agents'])],
            dolphin_config=DolphinConfig(vocab_size=256, n_embd=c['n_embd'],
                feromon_dim=c['feromon_dim'], identity_dim=c['identity_dim'],
                echo_dim=c['echo_dim'], sleep_dim=c['sleep_dim'], dropout=0))
        return LixySwarm(config, load_matriarca=False, load_runtime_state=False).to(device)


def predict(model, inputs, targets=None):
    return model(inputs, targets=targets, store_memory=False, use_memory=False,
                 update_runtime_state=False, update_memory_importance=False)
