"""
Tests para AntLifecycleManager — Hormigas Dinámicas + Legado Genético
=====================================================================
Test 1: hormiga con bajo fitness muere y transfiere legado
Test 2: nueva hormiga hereda identity_vec del legado
Test 3: enjambre no queda vacío (la última hormiga sobrevive)
Test 4: diversidad baja + recursos disponibles → spawn de nueva hormiga
Test 5: legado se almacena en Matriarca con prefijo [LEGACY]
Test 6: sin recursos → no hay spawn
Test 7: más de MAX_ANTS (antiguo) hormigas pueden existir si hay recursos
Test 8: death natural por fitness bajo — múltiples muertes sin límite mínimo artificial
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


def test_3_last_ant_survives():
    """Test 3: la última hormiga siempre sobrevive — no hay enjambre vacío."""
    print("\n[Test 3] La última hormiga nunca muere (no hay mínimo artificial)")

    from src.swarm.ant_lifecycle import AntLifecycleManager

    mat = MockMatriarca(embd_dim=64)
    # Solo 1 hormiga con fitness muy bajo
    fitnesses = {"0": 0.05}
    swarm, ants = make_swarm(1, fitnesses, mat)
    lcm = AntLifecycleManager(swarm, mat)

    # Forzar contador muy alto
    lcm._low_fitness_counters["0"] = AntLifecycleManager.LOW_FITNESS_STEPS + 1

    events = lcm.tick(step=500, swarm_diversity=0.8, n_connected_nodes=1)
    death_events = [e for e in events if e["type"] == "death"]

    # La última hormiga no puede morir
    assert len(swarm.agents) == 1, f"Enjambre vacío! Quedó {len(swarm.agents)} hormigas"
    assert len(death_events) == 0, f"Se mató la última hormiga: {death_events}"
    print(f"  ✓ La última hormiga sobrevivió (enjambre: {len(swarm.agents)} hormiga/s)")


def test_4_low_diversity_triggers_spawn_with_resources():
    """Test 4: diversidad baja + recursos disponibles → spawn de nueva hormiga."""
    print("\n[Test 4] Diversidad baja + recursos → spawn")

    from src.swarm.ant_lifecycle import AntLifecycleManager
    from unittest.mock import patch

    mat = MockMatriarca(embd_dim=64)
    fitnesses = {"0": 0.9, "1": 0.9}
    swarm, ants = make_swarm(2, fitnesses, mat)
    lcm = AntLifecycleManager(swarm, mat)

    n_before = len(swarm.agents)
    # Simular recursos disponibles con mock
    with patch.object(lcm, "_has_resources", return_value=True):
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


def test_6_no_spawn_when_no_resources():
    """Test 6: sin recursos (RAM/CPU) → no se spawnea nueva hormiga."""
    print("\n[Test 6] Sin recursos → no spawn")

    from src.swarm.ant_lifecycle import AntLifecycleManager
    from unittest.mock import patch

    mat = MockMatriarca(embd_dim=64)
    fitnesses = {"0": 0.9, "1": 0.9}
    swarm, ants = make_swarm(2, fitnesses, mat)
    lcm = AntLifecycleManager(swarm, mat)

    n_before = len(swarm.agents)
    # Simular sin recursos
    with patch.object(lcm, "_has_resources", return_value=False):
        events = lcm.tick(step=100, swarm_diversity=0.1, n_connected_nodes=1)
    birth_events = [e for e in events if e["type"] == "birth"]

    assert len(birth_events) == 0, f"No debería spawnear sin recursos, got {birth_events}"
    assert len(swarm.agents) == n_before, "El enjambre no debería crecer sin recursos"
    print(f"  ✓ Sin recursos: no hubo spawn. Enjambre: {len(swarm.agents)} hormigas")


def test_7_can_exceed_old_max_ants():
    """Test 7: el enjambre puede superar el antiguo MAX_ANTS=8 si hay recursos."""
    print("\n[Test 7] El enjambre puede superar el antiguo límite artificial")

    from src.swarm.ant_lifecycle import AntLifecycleManager
    from unittest.mock import patch

    mat = MockMatriarca(embd_dim=64)
    # Crear un enjambre con 8 hormigas (el antiguo máximo)
    n_start = 8
    fitnesses = {str(i): 0.9 for i in range(n_start)}
    swarm, ants = make_swarm(n_start, fitnesses, mat)
    lcm = AntLifecycleManager(swarm, mat)

    # Con recursos y diversidad baja, debe poder spawnear aunque seamos 8
    with patch.object(lcm, "_has_resources", return_value=True):
        events = lcm.tick(step=100, swarm_diversity=0.1, n_connected_nodes=1)
    birth_events = [e for e in events if e["type"] == "birth"]

    assert len(birth_events) == 1, f"Debería spawnear más allá del límite antiguo, got {birth_events}"
    assert len(swarm.agents) == n_start + 1
    print(f"  ✓ Enjambre creció de {n_start} a {len(swarm.agents)} (sin límite artificial)")


def test_8_multiple_deaths_without_artificial_floor():
    """Test 8: muerte natural por fitness — múltiples muertes, el piso es 1 (no 2)."""
    print("\n[Test 8] Múltiples muertes sin piso artificial (mínimo=1)")

    from src.swarm.ant_lifecycle import AntLifecycleManager
    from unittest.mock import patch

    mat = MockMatriarca(embd_dim=64)
    # 4 hormigas, 3 con fitness muy bajo
    fitnesses = {"0": 0.05, "1": 0.05, "2": 0.05, "3": 0.9}
    swarm, ants = make_swarm(4, fitnesses, mat)
    lcm = AntLifecycleManager(swarm, mat)

    # Contadores al máximo para las 3 débiles
    for aid in ["0", "1", "2"]:
        lcm._low_fitness_counters[aid] = AntLifecycleManager.LOW_FITNESS_STEPS + 1

    with patch.object(lcm, "_has_resources", return_value=False):  # no spawn
        events = lcm.tick(step=1000, swarm_diversity=0.9, n_connected_nodes=1)
    death_events = [e for e in events if e["type"] == "death"]

    # Las 3 débiles deben morir (antes con MIN_ANTS=2 solo podía morir 1)
    assert len(death_events) == 3, f"Se esperaban 3 muertes, got {len(death_events)}"
    assert len(swarm.agents) == 1, f"Debe quedar 1 hormiga, quedan {len(swarm.agents)}"
    print(f"  ✓ {len(death_events)} muertes naturales — enjambre reducido a 1 (sin piso artificial)")


# ─── Integration tests ────────────────────────────────────────────────────────


def test_9_tick_lifecycle_integrado():
    """Test 9: tick_lifecycle integra hormigas + delfines en LixySwarm."""
    import sys; sys.path.insert(0, ".")
    import torch
    from src.swarm.orchestrator import LixySwarm, SwarmConfig, AgentConfig
    from src.matriarca.matriarca import MatriarcaConfig
    print("\n[Test 9] tick_lifecycle integrado en LixySwarm")
    agent_cfgs = [AgentConfig(agent_id=i, n_agents=3) for i in range(3)]
    cfg = SwarmConfig(n_agents=3, agent_configs=agent_cfgs,
        matriarca_config=MatriarcaConfig(memory_path="/tmp/test9_mat.json", checkpoint_path="/tmp/test9_mat.pt"))
    swarm = LixySwarm(cfg, load_matriarca=False).cuda()
    events = swarm.tick_lifecycle(step=500, swarm_diversity=0.3, n_nodes=1)
    assert isinstance(events, list), "debe retornar lista"
    swarm.tick_lifecycle(step=501, swarm_diversity=0.7, n_nodes=5)
    assert swarm.dolphin.n_dolphins >= 2, f"con 5 nodos debe haber >=2 delfines"
    swarm.tick_lifecycle(step=502, swarm_diversity=0.7, n_nodes=1)
    assert swarm.dolphin.n_dolphins == 1, f"con 1 nodo debe haber 1 delfin"
    print(f"  Delfines finales: {swarm.dolphin.n_dolphins}")
    print("  ✓ tick_lifecycle integra hormigas + delfines")


def test_10_dolphin_scales_beyond_old_max():
    """Test 10: delfines escalan más allá del antiguo MAX_DOLPHINS=4."""
    from src.swarm.dolphin_pool import DolphinPool, _target_pool_size
    print("\n[Test 10] Delfines escalan sin techo artificial")
    # Con el antiguo MAX_DOLPHINS=4, n_nodes=30 daba 4. Ahora debe dar más.
    target_30 = _target_pool_size(30)
    target_10 = _target_pool_size(10)
    target_1 = _target_pool_size(1)
    assert target_30 > 4, f"Con 30 nodos debe superar antiguo techo de 4, got {target_30}"
    assert target_10 >= 4, f"Con 10 nodos debe ser >=4, got {target_10}"
    assert target_1 == 1, f"Con 1 nodo debe ser 1, got {target_1}"
    print(f"  n_nodes=1  → {target_1} delfín")
    print(f"  n_nodes=10 → {target_10} delfines")
    print(f"  n_nodes=30 → {target_30} delfines (sin techo)")
    print("  ✓ Delfines escalan libremente")


def test_11_training_loop_tick():
    """Test 11: simula training loop con tick_lifecycle."""
    import sys; sys.path.insert(0, ".")
    import torch
    from src.swarm.orchestrator import LixySwarm, SwarmConfig, AgentConfig
    from src.matriarca.matriarca import MatriarcaConfig
    print("\n[Test 11] Simulación training loop con tick_lifecycle")
    agent_cfgs = [AgentConfig(agent_id=i, n_agents=3) for i in range(3)]
    cfg = SwarmConfig(n_agents=3, agent_configs=agent_cfgs,
        matriarca_config=MatriarcaConfig(memory_path="/tmp/test11_mat.json", checkpoint_path="/tmp/test11_mat.pt"))
    swarm = LixySwarm(cfg, load_matriarca=False).cuda()
    all_events = []
    for step in range(100, 301, 100):
        if hasattr(swarm, "ant_lifecycle") and swarm.ant_lifecycle:
            evs = swarm.tick_lifecycle(step=step, swarm_diversity=0.5, n_nodes=1)
            all_events.extend(evs)
    print(f"  Ticks: 3 | Eventos: {len(all_events)} | Hormigas: {len(swarm.agents)} | Delfines: {swarm.dolphin.n_dolphins}")
    assert isinstance(all_events, list)
    print("  ✓ Training loop tick funciona sin errores")


if __name__ == "__main__":
    tests = [
        test_1_low_fitness_ant_dies_and_stores_legacy,
        test_2_new_ant_inherits_identity_from_legacy,
        test_3_last_ant_survives,
        test_4_low_diversity_triggers_spawn_with_resources,
        test_5_legacy_stored_in_matriarca,
        test_6_no_spawn_when_no_resources,
        test_7_can_exceed_old_max_ants,
        test_8_multiple_deaths_without_artificial_floor,
        test_9_tick_lifecycle_integrado,
        test_10_dolphin_scales_beyond_old_max,
        test_11_training_loop_tick,
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
        import sys; sys.exit(1)