"""
Tests para AntLifecycleManager — Hormigas Dinámicas + Legado Genético
=====================================================================
Test 1: hormiga con bajo fitness muere y transfiere legado
Test 2: nueva hormiga hereda identity_vec del legado
Test 3: enjambre nunca baja de MIN_ANTS
Test 4: diversidad baja → spawn de nueva hormiga
Test 5: legado se almacena en Matriarca con prefijo [LEGACY]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn

# ─── Mock classes ──────────────────────────────────────────────────────────────

class MockAgentConfig:
    def __init__(self, agent_id, n_embd=64, block_size=32, vocab_size=256, n_layer=2,
                 n_head=4, dropout=0.0, bias=False, feromon_dim=32, identity_dim=16,
                 n_agents=3):
        self.agent_id = agent_id
        self.n_embd = n_embd
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout
        self.bias = bias
        self.feromon_dim = feromon_dim
        self.identity_dim = identity_dim
        self.n_agents = n_agents


class MockAnt(nn.Module):
    """Hormiga mínima con identity_vec y config."""
    def __init__(self, agent_id, identity_dim=16):
        super().__init__()
        cfg = MockAgentConfig(agent_id=agent_id, identity_dim=identity_dim)
        self.config = cfg  # real AgentBase uses self.config
        self.identity_vec = nn.Parameter(torch.randn(identity_dim))
        self.linear = nn.Linear(16, 16)  # pesos para copiar

    def forward(self, x):
        return self.linear(x)


class MockSpecializationTracker:
    def __init__(self, fitnesses: dict):
        """fitnesses: {str(agent_id): float}"""
        self._fitnesses = fitnesses
        self.current = {k: type("AF", (), {"fitness": v})() for k, v in fitnesses.items()}

    def _infer_label(self, agent_id_str):
        return f"role_{agent_id_str}"


class MockMatriarca:
    """Matriarca mínima con add() y metadata."""
    def __init__(self, embd_dim=64):
        class MockCfg:
            pass
        self.cfg = MockCfg()
        self.cfg.embd_dim = embd_dim
        self.metadata = []
        self._embeddings = []

    def add(self, embedding: torch.Tensor, text: str, importance: float = 1.0):
        self._embeddings.append(embedding.detach().cpu().clone())
        self.metadata.append({"text": text, "importance": importance})

    def get_embeddings(self, device="cpu"):
        if not self._embeddings:
            return torch.zeros(0, self.cfg.embd_dim)
        return torch.stack(self._embeddings).to(device)


class MockSwarm:
    def __init__(self, ants, fitnesses: dict, matriarca=None):
        self.agents = nn.ModuleList(ants)
        self.specialization = MockSpecializationTracker(fitnesses)
        self.matriarca = matriarca


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_swarm(n_ants=3, fitnesses=None, matriarca=None):
    ants = [MockAnt(agent_id=i) for i in range(n_ants)]
    if fitnesses is None:
        fitnesses = {str(i): 0.8 for i in range(n_ants)}
    swarm = MockSwarm(ants, fitnesses, matriarca)
    return swarm, ants


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_1_low_fitness_ant_dies_and_stores_legacy():
    """Test 1: hormiga con bajo fitness muere y transfiere legado."""
    print("\n[Test 1] Hormiga con bajo fitness → muere + legado")

    from src.swarm.ant_lifecycle import AntLifecycleManager

    mat = MockMatriarca(embd_dim=64)
    # Ant 0 tiene fitness muy bajo
    fitnesses = {"0": 0.1, "1": 0.9, "2": 0.9}
    swarm, ants = make_swarm(3, fitnesses, mat)
    lcm = AntLifecycleManager(swarm, mat)

    # Simular LOW_FITNESS_STEPS ticks con fitness bajo
    lcm._low_fitness_counters["0"] = AntLifecycleManager.LOW_FITNESS_STEPS

    events = lcm.tick(step=1000, swarm_diversity=0.8, n_connected_nodes=1)

    death_events = [e for e in events if e["type"] == "death"]
    assert len(death_events) == 1, f"Se esperaba 1 muerte, got {death_events}"
    assert death_events[0]["ant_id"] == "0"
    assert death_events[0]["legacy_stored"] is True
    assert len(swarm.agents) == 2, f"Quedaron {len(swarm.agents)} hormigas, se esperaban 2"

    # Verificar que el legado quedó en la Matriarca
    assert len(mat.metadata) >= 1
    legacy_entries = [m for m in mat.metadata if "[LEGACY]" in m["text"]]
    assert len(legacy_entries) == 1, f"Legacy en Matriarca: {legacy_entries}"
    print(f"  ✓ Muerte registrada: {death_events[0]}")
    print(f"  ✓ Legado en Matriarca: {legacy_entries[0]['text'][:80]}")


def test_2_new_ant_inherits_identity_from_legacy():
    """Test 2: nueva hormiga hereda identity_vec del legado."""
    print("\n[Test 2] Spawn con herencia de identity_vec")

    from src.swarm.ant_lifecycle import AntLifecycleManager

    mat = MockMatriarca(embd_dim=64)
    fitnesses = {"0": 0.9, "1": 0.9}
    swarm, ants = make_swarm(2, fitnesses, mat)
    lcm = AntLifecycleManager(swarm, mat)

    # Meter un legado manualmente en la Matriarca
    legacy_emb = torch.ones(64) * 0.42
    mat.add(embedding=legacy_emb, text="[LEGACY] ant_99 role=X fitness=0.80 reason=test", importance=0.8)

    # spawn
    event = lcm._spawn_ant(step=1)
    assert event["type"] == "birth"
    assert event["inherited"] is True, "Se esperaba herencia"

    new_ant = swarm.agents[-1]
    id_slice = legacy_emb[:new_ant.config.identity_dim]
    assert torch.allclose(new_ant.identity_vec.data.cpu(), id_slice, atol=1e-5), \
        f"identity_vec no coincide con legado: {new_ant.identity_vec.data[:5]}"
    print(f"  ✓ Nueva hormiga id={event['ant_id']} heredó identity_vec del legado")


def test_3_swarm_never_below_min_ants():
    """Test 3: enjambre nunca baja de MIN_ANTS."""
    print("\n[Test 3] Enjambre nunca baja de MIN_ANTS")

    from src.swarm.ant_lifecycle import AntLifecycleManager

    mat = MockMatriarca(embd_dim=64)
    # Todas las hormigas con fitness bajo
    fitnesses = {"0": 0.05, "1": 0.05}
    swarm, ants = make_swarm(2, fitnesses, mat)
    lcm = AntLifecycleManager(swarm, mat)

    # Forzar contadores altos en ambas
    lcm._low_fitness_counters["0"] = AntLifecycleManager.LOW_FITNESS_STEPS + 1
    lcm._low_fitness_counters["1"] = AntLifecycleManager.LOW_FITNESS_STEPS + 1

    events = lcm.tick(step=500, swarm_diversity=0.8, n_connected_nodes=1)
    death_events = [e for e in events if e["type"] == "death"]

    assert len(swarm.agents) >= AntLifecycleManager.MIN_ANTS, \
        f"Enjambre bajó a {len(swarm.agents)}, mínimo es {AntLifecycleManager.MIN_ANTS}"
    print(f"  ✓ Enjambre en {len(swarm.agents)} hormigas (≥ MIN_ANTS={AntLifecycleManager.MIN_ANTS})")
    print(f"  ✓ Muertes ocurridas: {len(death_events)} (no podían morir todas)")


def test_4_low_diversity_triggers_spawn():
    """Test 4: diversidad baja → spawn de nueva hormiga."""
    print("\n[Test 4] Diversidad baja → spawn")

    from src.swarm.ant_lifecycle import AntLifecycleManager

    mat = MockMatriarca(embd_dim=64)
    fitnesses = {"0": 0.9, "1": 0.9}
    swarm, ants = make_swarm(2, fitnesses, mat)
    lcm = AntLifecycleManager(swarm, mat)

    n_before = len(swarm.agents)
    # Diversidad muy baja (0.1 < DIVERSITY_THRESHOLD=0.4) con espacio (2 < MAX_ANTS=8)
    events = lcm.tick(step=100, swarm_diversity=0.1, n_connected_nodes=1)
    birth_events = [e for e in events if e["type"] == "birth"]

    assert len(birth_events) == 1, f"Se esperaba 1 nacimiento, got {birth_events}"
    assert len(swarm.agents) == n_before + 1
    print(f"  ✓ Spawn: {birth_events[0]}")
    print(f"  ✓ Enjambre creció de {n_before} a {len(swarm.agents)} hormigas")


def test_5_legacy_stored_in_matriarca():
    """Test 5: legado se almacena en Matriarca con prefijo [LEGACY]."""
    print("\n[Test 5] Legado con prefijo [LEGACY] en Matriarca")

    from src.swarm.ant_lifecycle import AntLifecycleManager

    mat = MockMatriarca(embd_dim=64)
    fitnesses = {"0": 0.2, "1": 0.9, "2": 0.9}
    swarm, ants = make_swarm(3, fitnesses, mat)
    lcm = AntLifecycleManager(swarm, mat)
    lcm._low_fitness_counters["0"] = AntLifecycleManager.LOW_FITNESS_STEPS

    events = lcm.tick(step=600, swarm_diversity=0.8, n_connected_nodes=1)

    legacy_entries = [m for m in mat.metadata if m["text"].startswith("[LEGACY]")]
    assert len(legacy_entries) >= 1, f"No se encontraron entradas [LEGACY]: {mat.metadata}"

    entry = legacy_entries[0]
    assert "ant_0" in entry["text"], f"Texto inesperado: {entry['text']}"
    assert "fitness=" in entry["text"] or "fitness_avg" in entry["text"]
    assert 0.0 <= entry["importance"] <= 1.0

    print(f"  ✓ Entrada legacy: {entry['text']}")
    print(f"  ✓ Importancia: {entry['importance']:.3f}")

    # Verificar que el embedding quedó almacenado
    embs = mat.get_embeddings("cpu")
    assert embs.shape[0] >= 1, "No hay embeddings en Matriarca"
    assert embs.shape[1] == mat.cfg.embd_dim
    print(f"  ✓ Embedding almacenado: shape={list(embs.shape)}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_1_low_fitness_ant_dies_and_stores_legacy,
        test_2_new_ant_inherits_identity_from_legacy,
        test_3_swarm_never_below_min_ants,
        test_4_low_diversity_triggers_spawn,
        test_5_legacy_stored_in_matriarca,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  → PASS")
        except Exception as e:
            failed += 1
            import traceback
            print(f"  → FAIL: {e}")
            traceback.print_exc()

    print(f"\n{'='*50}")
    print(f"Tests: {passed}/{len(tests)} passed", "✅" if failed == 0 else "❌")
    if failed > 0:
        sys.exit(1)
