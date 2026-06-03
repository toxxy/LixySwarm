"""
Tests para NodeManager + SectManager — Nueva Arquitectura de Enjambre
=====================================================================
La arquitectura correcta:
  🐜 Hormiga = Nodo físico (máquina)
  🏛️ Secta  = Especialidad (grupo de agentes)
  🐘 Matriarca = Única constante (legado de sectas)

Test  1: nodo se une a la red → NodeRecord creado
Test  2: nodo se va → legado en Matriarca
Test  3: nodo timeout → prune_dead_nodes() lo elimina
Test  4: nodo débil solo hospeda 1 secta; nodo fuerte hospeda varias
Test  5: SectManager spawn sin nodos → retorna None
Test  6: SectManager spawn con nodo disponible → secta creada
Test  7: Secta muere por fitness bajo → legado en Matriarca
Test  8: SectManager tick diversidad baja → spawn automático
Test  9: SectManager nunca mata la última secta
Test 10: tick_lifecycle en LixySwarm integra NodeManager + SectManager
Test 11: nodo_joined → secta asignada al mejor nodo
Test 12: conteo de hormigas + sectas en stats()
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import time
import torch
import torch.nn as nn

# ─── Mock Matriarca ────────────────────────────────────────────────────────────

class MockMatriarca:
    def __init__(self, embd_dim=64):
        class MockCfg:
            pass
        self.cfg = MockCfg()
        self.cfg.embd_dim = embd_dim
        self.metadata = []
        self._embeddings = []

    def add(self, embedding, text: str, importance: float = 1.0):
        self._embeddings.append(embedding.detach().cpu().clone())
        self.metadata.append({"text": text, "importance": importance})

    def get_embeddings(self, device="cpu"):
        if not self._embeddings:
            return torch.zeros(0, self.cfg.embd_dim)
        return torch.stack(self._embeddings).to(device)


class BifurcatingMatriarca(MockMatriarca):
    def __init__(self, embd_dim=64):
        super().__init__(embd_dim=embd_dim)
        self.legacy_calls = []

    def suggest_bifurcation(self, sect, swarm_diversity: float):
        return {
            "should_bifurcate": True,
            "child_roles": ["refinador_logico", "refinador_creativo"],
            "reason": "test_low_diversity",
            "confidence": 0.91,
        }

    def store_sect_legacy(self, sect, reason: str, **kwargs):
        self.legacy_calls.append({"sect_id": sect.sect_id, "reason": reason, **kwargs})


# ─── Tests NodeManager ─────────────────────────────────────────────────────────

def test_1_node_joins():
    """Test 1: nodo se une → NodeRecord creado con hardware."""
    print("\n[Test 1] Nodo se une a la red")
    from src.swarm.node_manager import NodeManager, HardwareProfile

    nm = NodeManager()
    hw = HardwareProfile(cpu_cores=8, ram_gb=32.0, gpu_vram_gb=24.0, has_gpu=True)
    record = nm.node_joined("rtx-5090", hardware=hw)

    assert record.node_id == "rtx-5090"
    assert record.hardware.has_gpu is True
    assert record.hardware.gpu_vram_gb == 24.0
    assert nm.n_nodes() == 1
    print(f"  ✓ Nodo registrado: {record.node_id} | compute={record.hardware.compute_score:.1f}")
    print(f"  ✓ Max sectas concurrentes: {record.hardware.max_concurrent_sects}")


def test_2_node_leaves_legacy():
    """Test 2: nodo se va → legado en Matriarca."""
    print("\n[Test 2] Nodo se va → legado en Matriarca")
    from src.swarm.node_manager import NodeManager, HardwareProfile

    mat = MockMatriarca(embd_dim=64)
    nm = NodeManager(matriarca=mat)
    hw = HardwareProfile(cpu_cores=4, ram_gb=16.0, gpu_vram_gb=0.0, has_gpu=False)
    nm.node_joined("vps-01", hardware=hw)

    event = nm.node_left("vps-01", reason="disconnect")

    assert event is not None
    assert event["type"] == "node_left"
    assert event["node_id"] == "vps-01"
    assert event["legacy_stored"] is True
    assert nm.n_nodes() == 0

    # Verificar legado en Matriarca
    node_legacies = [m for m in mat.metadata if "[NODE_LEGACY]" in m["text"]]
    assert len(node_legacies) == 1
    assert "vps-01" in node_legacies[0]["text"]
    print(f"  ✓ Evento: {event['type']} | legacy_stored={event['legacy_stored']}")
    print(f"  ✓ Legado: {node_legacies[0]['text'][:80]}")


def test_3_node_timeout_prune():
    """Test 3: nodo sin heartbeat → prune_dead_nodes() lo elimina."""
    print("\n[Test 3] Timeout de nodo → prune")
    from src.swarm.node_manager import NodeManager, HardwareProfile, NodeRecord

    mat = MockMatriarca(embd_dim=64)
    nm = NodeManager(matriarca=mat)
    hw = HardwareProfile(cpu_cores=2, ram_gb=4.0)
    nm.node_joined("old-laptop", hardware=hw)

    # Simular que el nodo no ha dado señal por mucho tiempo
    nm._nodes["old-laptop"].last_seen = time.time() - NodeManager.NODE_TIMEOUT_SECONDS - 1

    events = nm.prune_dead_nodes()

    assert len(events) == 1
    assert events[0]["reason"] == "timeout"
    assert nm.n_nodes() == 0
    print(f"  ✓ Nodo eliminado por timeout: {events[0]['node_id']}")


def test_4_hardware_capacity():
    """Test 4: nodo débil = 1 secta, nodo fuerte = varias."""
    print("\n[Test 4] Capacidad de hardware → slots de sectas")
    from src.swarm.node_manager import HardwareProfile

    weak = HardwareProfile(cpu_cores=1, ram_gb=2.0, gpu_vram_gb=0.0, has_gpu=False)
    medium = HardwareProfile(cpu_cores=8, ram_gb=32.0, gpu_vram_gb=8.0, has_gpu=True)
    strong = HardwareProfile(cpu_cores=16, ram_gb=64.0, gpu_vram_gb=24.0, has_gpu=True)
    beast = HardwareProfile(cpu_cores=32, ram_gb=128.0, gpu_vram_gb=80.0, has_gpu=True)

    assert weak.max_concurrent_sects == 1, f"Nodo débil debe tener 1, got {weak.max_concurrent_sects}"
    assert medium.max_concurrent_sects >= 2
    assert strong.max_concurrent_sects >= 3
    assert beast.max_concurrent_sects >= 4

    print(f"  ✓ Débil (2GB RAM, sin GPU):    {weak.max_concurrent_sects} secta(s)")
    print(f"  ✓ Medio (8GB VRAM):             {medium.max_concurrent_sects} sectas")
    print(f"  ✓ Fuerte (24GB VRAM):           {strong.max_concurrent_sects} sectas")
    print(f"  ✓ Beast (80GB VRAM):            {beast.max_concurrent_sects} sectas")


# ─── Tests SectManager ─────────────────────────────────────────────────────────

def test_5_sect_spawn_without_nodes():
    """Test 5: sin nodos disponibles → spawn retorna None."""
    print("\n[Test 5] Sin nodos → no spawn de secta")
    from src.swarm.node_manager import NodeManager
    from src.swarm.sect_manager import SectManager

    nm = NodeManager()  # sin nodos
    sm = SectManager(node_manager=nm)

    sect = sm.spawn_sect("explorador")
    assert sect is None, f"Sin nodos no debe spawnear, got {sect}"
    assert sm.n_sects() == 0
    print(f"  ✓ Sin nodos: spawn retornó None, n_sects={sm.n_sects()}")


def test_6_sect_spawn_with_node():
    """Test 6: con nodo disponible → secta creada y asignada."""
    print("\n[Test 6] Con nodo disponible → spawn de secta")
    from src.swarm.node_manager import NodeManager, HardwareProfile
    from src.swarm.sect_manager import SectManager

    nm = NodeManager()
    hw = HardwareProfile(cpu_cores=8, ram_gb=32.0, gpu_vram_gb=24.0, has_gpu=True)
    nm.node_joined("rtx-5090", hardware=hw)

    sm = SectManager(node_manager=nm)
    sect = sm.spawn_sect("explorador", priority=0.8)

    assert sect is not None
    assert sect.role_type == "explorador"
    assert sm.n_sects() == 1

    # Verificar que el nodo tiene la secta asignada
    node = nm.get_node("rtx-5090")
    assert sect.sect_id in node.connected_sects

    print(f"  ✓ Secta creada: {sect.sect_id} | role={sect.role_type}")
    print(f"  ✓ Asignada al nodo: {node.node_id} | sects={node.connected_sects}")


def test_7_sect_death_legacy():
    """Test 7: secta muere → legado en Matriarca."""
    print("\n[Test 7] Muerte de secta → [SECT_LEGACY] en Matriarca")
    from src.swarm.node_manager import NodeManager, HardwareProfile
    from src.swarm.sect_manager import SectManager

    mat = MockMatriarca(embd_dim=64)
    nm = NodeManager()
    hw = HardwareProfile(cpu_cores=8, ram_gb=32.0, gpu_vram_gb=24.0, has_gpu=True)
    nm.node_joined("rtx-5090", hardware=hw)

    sm = SectManager(node_manager=nm, matriarca=mat)
    sect = sm.spawn_sect("refinador")
    assert sect is not None
    sect_id = sect.sect_id

    event = sm.kill_sect(sect_id, reason="low_fitness")

    assert event is not None
    assert event["type"] == "sect_death"
    assert event["legacy_stored"] is True
    assert sm.n_sects() == 0

    # Legado en Matriarca
    legacies = [m for m in mat.metadata if "[SECT_LEGACY]" in m["text"]]
    assert len(legacies) == 1
    assert "refinador" in legacies[0]["text"]

    print(f"  ✓ Evento: {event['type']} | reason={event['reason']}")
    print(f"  ✓ Legado: {legacies[0]['text'][:80]}")


def test_8_tick_low_diversity_spawns_sect():
    """Test 8: tick con diversidad baja → spawn automático de secta."""
    print("\n[Test 8] Tick diversidad baja → spawn de secta")
    from src.swarm.node_manager import NodeManager, HardwareProfile
    from src.swarm.sect_manager import SectManager

    mat = MockMatriarca(embd_dim=64)
    nm = NodeManager()
    hw = HardwareProfile(cpu_cores=16, ram_gb=64.0, gpu_vram_gb=24.0, has_gpu=True)
    nm.node_joined("strong-node", hardware=hw)

    sm = SectManager(node_manager=nm, matriarca=mat)
    n_before = sm.n_sects()

    events = sm.tick(step=100, swarm_diversity=0.1)  # diversidad muy baja
    birth_events = [e for e in events if e["type"] == "sect_birth"]

    assert len(birth_events) == 1, f"Se esperaba 1 spawn, got {birth_events}"
    assert sm.n_sects() == n_before + 1
    print(f"  ✓ Secta nacida: {birth_events[0]['sect_id']} role={birth_events[0]['role_type']}")
    print(f"  ✓ Trigger: {birth_events[0]['trigger']}")


def test_9_last_sect_never_dies():
    """Test 9: la última secta siempre sobrevive."""
    print("\n[Test 9] La última secta no puede morir")
    from src.swarm.node_manager import NodeManager, HardwareProfile
    from src.swarm.sect_manager import SectManager, SectRecord

    nm = NodeManager()
    hw = HardwareProfile(cpu_cores=8, ram_gb=32.0, gpu_vram_gb=24.0, has_gpu=True)
    nm.node_joined("node-1", hardware=hw)

    sm = SectManager(node_manager=nm)
    sect = sm.spawn_sect("explorador")
    assert sect is not None

    # Forzar que la secta esté "moribunda"
    sect.fitness_history = [0.05] * (SectRecord.LOW_FITNESS_STEPS_TO_DIE + 10)

    events = sm.tick(step=999, swarm_diversity=0.9)
    death_events = [e for e in events if e["type"] == "sect_death"]

    assert sm.n_sects() == 1, f"La última secta debe sobrevivir, quedan {sm.n_sects()}"
    assert len(death_events) == 0
    print(f"  ✓ Única secta sobrevivió: {sm.all_sects()[0].sect_id}")


# ─── Tests de integración ─────────────────────────────────────────────────────

def test_10_orchestrator_has_node_and_sect_managers():
    """Test 10: LixySwarm expone node_manager y sect_manager."""
    print("\n[Test 10] LixySwarm integra NodeManager + SectManager")
    from src.swarm.orchestrator import LixySwarm, SwarmConfig, AgentConfig
    from src.matriarca.matriarca import MatriarcaConfig

    agent_cfgs = [AgentConfig(agent_id=i, n_agents=3) for i in range(3)]
    cfg = SwarmConfig(n_agents=3, agent_configs=agent_cfgs,
        matriarca_config=MatriarcaConfig(
            memory_path="/tmp/test10_mat.json",
            checkpoint_path="/tmp/test10_mat.pt",
        ))
    swarm = LixySwarm(cfg, load_matriarca=False).cuda()

    assert hasattr(swarm, "node_manager"), "Debe tener node_manager"
    assert hasattr(swarm, "sect_manager"), "Debe tener sect_manager"
    assert swarm.node_manager.n_nodes() >= 1, "Al menos el nodo local"
    assert swarm.sect_manager.n_sects() >= 1, "Al menos una secta inicial"

    local_node = swarm.node_manager.get_node("local")
    assert local_node is not None, "Debe existir nodo local"
    assert local_node.is_local is True

    print(f"  ✓ Nodos: {swarm.node_manager.n_nodes()} | Sectas: {swarm.sect_manager.n_sects()}")
    print(f"  ✓ Nodo local: compute_score={local_node.hardware.compute_score:.1f}")
    print(f"  ✓ Sectas iniciales: {[s.role_type for s in swarm.sect_manager.all_sects()]}")


def test_11_tick_lifecycle_full():
    """Test 11: tick_lifecycle integra toda la cadena."""
    print("\n[Test 11] tick_lifecycle completo")
    from src.swarm.orchestrator import LixySwarm, SwarmConfig, AgentConfig
    from src.matriarca.matriarca import MatriarcaConfig

    agent_cfgs = [AgentConfig(agent_id=i, n_agents=3) for i in range(3)]
    cfg = SwarmConfig(n_agents=3, agent_configs=agent_cfgs,
        matriarca_config=MatriarcaConfig(
            memory_path="/tmp/test11_mat.json",
            checkpoint_path="/tmp/test11_mat.pt",
        ))
    swarm = LixySwarm(cfg, load_matriarca=False).cuda()

    events = swarm.tick_lifecycle(step=500, swarm_diversity=0.3, n_nodes=3)

    assert isinstance(events, list)
    # Con 3 nodos debe escalar delfines
    assert swarm.dolphin.n_dolphins >= 2, f"Con 3 nodos esperamos >=2 delfines"
    print(f"  ✓ Eventos: {len(events)} | Delfines: {swarm.dolphin.n_dolphins}")
    print(f"  ✓ Nodos: {swarm.node_manager.n_nodes()} | Sectas: {swarm.sect_manager.n_sects()}")


def test_12_stats():
    """Test 12: stats() retorna conteos correctos."""
    print("\n[Test 12] stats() NodeManager + SectManager")
    from src.swarm.node_manager import NodeManager, HardwareProfile
    from src.swarm.sect_manager import SectManager

    nm = NodeManager()
    hw = HardwareProfile(cpu_cores=16, ram_gb=64.0, gpu_vram_gb=24.0, has_gpu=True)
    nm.node_joined("node-A", hardware=hw)
    nm.node_joined("node-B", hardware=hw)

    sm = SectManager(node_manager=nm)
    sm.spawn_sect("explorador")
    sm.spawn_sect("refinador")

    nm_stats = nm.stats()
    sm_stats = sm.stats()

    assert nm_stats["total_nodes"] == 2
    assert nm_stats["joins"] == 2
    assert sm_stats["total_sects"] == 2
    assert sm_stats["births"] == 2

    print(f"  ✓ Nodos: {nm_stats['total_nodes']} joins={nm_stats['joins']}")
    print(f"  ✓ Sectas: {sm_stats['total_sects']} births={sm_stats['births']}")


def test_13_contribution_mode():
    """Test: ContributionMode MAXIMUM/MODERATE/RELAY en NodeRecord."""
    print("\n[Test 13] ContributionMode")
    from src.swarm.node_manager import NodeManager, ContributionMode, HardwareProfile

    nm = NodeManager()
    nm.node_joined("n1", HardwareProfile(gpu_vram_gb=24, has_gpu=True, cpu_cores=16, ram_gb=64))
    n = nm.get_node("n1")

    # Default es MAXIMUM
    assert n.contribution_mode == ContributionMode.MAXIMUM
    assert n.effective_gpu_fraction == 1.0
    assert n.can_host_sect("s1")

    # MODERATE: mitad de GPU, mitad de sectas
    nm.set_contribution_mode("n1", ContributionMode.MODERATE)
    assert n.effective_gpu_fraction == 0.5
    cap_max = HardwareProfile(gpu_vram_gb=24, has_gpu=True, cpu_cores=16, ram_gb=64).max_concurrent_sects
    assert n._effective_sect_capacity() == max(1, cap_max // 2)
    assert n.can_host_sect("s1")

    # RELAY: sin GPU, sin sectas
    nm.set_contribution_mode("n1", ContributionMode.RELAY)
    assert n.effective_gpu_fraction == 0.0
    assert n._effective_sect_capacity() == 0
    assert not n.can_host_sect("s1")

    # to_dict incluye los campos nuevos
    d = n.to_dict()
    assert d["contribution_mode"] == "relay"
    assert d["effective_gpu_fraction"] == 0.0
    assert d["effective_sect_capacity"] == 0

    # set_contribution_mode retorna False para nodo inexistente
    assert not nm.set_contribution_mode("no_existe", ContributionMode.MAXIMUM)

    print(f"  ✓ MAXIMUM/MODERATE/RELAY: GPU y sectas responden al modo")
    print(f"  ✓ to_dict: mode=relay gpu=0.0 sects=0")


def test_14_relay_disconnects_sects():
    """Test: bajar a RELAY desconecta sectas existentes."""
    print("\n[Test 14] RELAY desconecta sectas")
    from src.swarm.node_manager import NodeManager, ContributionMode, HardwareProfile

    nm = NodeManager()
    nm.node_joined("n1", HardwareProfile(gpu_vram_gb=24, has_gpu=True, cpu_cores=16, ram_gb=64))
    n = nm.get_node("n1")

    # Añadir sectas manualmente
    n.connected_sects.extend(["s1", "s2", "s3"])
    assert len(n.connected_sects) == 3

    # Bajar a RELAY limpia sectas
    nm.set_contribution_mode("n1", ContributionMode.RELAY)
    assert len(n.connected_sects) == 0, f"RELAY debe limpiar sectas, got {n.connected_sects}"
    print(f"  ✓ RELAY limpió {3} sectas automáticamente")


def test_15_sect_bifurcation():
    """Test: diversidad baja + secta fuerte → bifurcación guiada por Matriarca."""
    print("\n[Test 15] Bifurcación de secta")
    from src.swarm.node_manager import NodeManager, HardwareProfile
    from src.swarm.sect_manager import SectManager

    mat = BifurcatingMatriarca(embd_dim=64)
    nm = NodeManager()
    nm.node_joined("beast", HardwareProfile(cpu_cores=32, ram_gb=128, gpu_vram_gb=80, has_gpu=True))

    sm = SectManager(node_manager=nm, matriarca=mat)
    parent = sm.spawn_sect("refinador", priority=0.9)
    assert parent is not None
    slot = sm.add_agent_to_sect(parent.sect_id, "beast")
    assert slot is not None
    sm.update_agent_fitness(parent.sect_id, slot.agent_id, 0.92)

    events = sm.tick(step=1000, swarm_diversity=0.1)
    bifurcations = [e for e in events if e["type"] == "sect_bifurcation"]

    assert len(bifurcations) == 1, f"Se esperaba bifurcación, got {events}"
    event = bifurcations[0]
    assert event["sect_id"] == parent.sect_id
    assert len(event["children"]) >= 1
    assert parent.bifurcated_at is not None
    assert len(parent.children_sect_ids) == len(event["children"])
    assert mat.legacy_calls and mat.legacy_calls[0]["reason"] == "bifurcation"

    child = sm.get_sect(event["children"][0])
    assert child is not None
    assert child.parent_sect_id == parent.sect_id
    print(f"  ✓ Padre: {parent.sect_id} → hijas={event['children']}")
    print(f"  ✓ Legado guardado: {mat.legacy_calls[0]}")


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_1_node_joins,
        test_2_node_leaves_legacy,
        test_3_node_timeout_prune,
        test_4_hardware_capacity,
        test_5_sect_spawn_without_nodes,
        test_6_sect_spawn_with_node,
        test_7_sect_death_legacy,
        test_8_tick_low_diversity_spawns_sect,
        test_9_last_sect_never_dies,
        test_10_orchestrator_has_node_and_sect_managers,
        test_11_tick_lifecycle_full,
        test_12_stats,
        test_13_contribution_mode,
        test_14_relay_disconnects_sects,
        test_15_sect_bifurcation,
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
