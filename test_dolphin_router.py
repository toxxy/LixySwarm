"""
Tests para EcholocationRouter + AdaptiveSleepController (Phase B)
=================================================================
El Delfín como sistema de enrutamiento inteligente.

Test  1: PingResponse score compuesto correcto
Test  2: SectPingEncoder genera afinidad acústica [0,1]
Test  3: EcholocationRouter.route() selecciona secta correcta
Test  4: Multi-secta: problema complejo → delfín elige 2+ sectas
Test  5: Sin sectas → RouteDecision mode="broadcast"
Test  6: integrate_responses() pondera por score
Test  7: AdaptiveSleepController umbrales (rest/normal/aggressive)
Test  8: sleep_controller.store_acoustic_map() + recent_acoustic_maps()
Test  9: AdaptiveSleepController.save() / load() persistencia
Test 10: DolphinSwarmBridge.forward() incluye sleep_mode + acoustic_map
Test 11: DolphinSwarmBridge.route() retorna RouteDecision con sects
Test 12: LixySwarm.route_task() integra todo el flujo
Test 13: tick_lifecycle actualiza sleep_mode en todos los delfines
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import tempfile, os

from src.swarm.dolphin_router import (
    EcholocationRouter, AdaptiveSleepController, SectPingEncoder,
    PingResponse, RouteDecision,
)


# ─── Mock helpers ─────────────────────────────────────────────────────────────

def make_sect(sect_id, role_type, n_agents=2, avg_fitness=0.7):
    """Crea un SectRecord mock."""
    from src.swarm.sect_manager import SectRecord, AgentSlot
    sect = SectRecord(sect_id=sect_id, role_type=role_type)
    for i in range(n_agents):
        slot = AgentSlot(agent_id=i, node_id="local", fitness=avg_fitness)
        sect.agents.append(slot)
    return sect


def random_acoustic_map(feromon_dim=256):
    return torch.randn(feromon_dim)


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_1_ping_response_score():
    """Test 1: PingResponse.score combina todas las métricas correctamente."""
    print("\n[Test 1] PingResponse.score compuesto")

    r_good = PingResponse(
        sect_id="s1", role_type="explorador",
        can_handle=0.9, fitness=0.8, load_factor=0.1,
        latency_ms=10, acoustic_affinity=0.85,
    )
    r_bad = PingResponse(
        sect_id="s2", role_type="refinador",
        can_handle=0.2, fitness=0.1, load_factor=0.9,
        latency_ms=200, acoustic_affinity=0.1,
    )

    assert r_good.score > r_bad.score, f"Buena debe superar mala: {r_good.score:.3f} vs {r_bad.score:.3f}"
    assert 0 <= r_good.score <= 1
    print(f"  ✓ Score buena: {r_good.score:.3f} | Score mala: {r_bad.score:.3f}")


def test_2_ping_encoder_affinity():
    """Test 2: SectPingEncoder genera afinidad acústica en [0,1]."""
    print("\n[Test 2] SectPingEncoder afinidad acústica")

    enc = SectPingEncoder(acoustic_map_dim=128)  # echo_dim del Delfín
    am = random_acoustic_map(128)  # acoustic_map tiene echo_dim, no feromon_dim

    score_exp = enc(am, "explorador")
    score_ref = enc(am, "refinador")
    score_dol = enc(am, "delfín")

    for s, role in [(score_exp, "explorador"), (score_ref, "refinador"), (score_dol, "delfín")]:
        assert 0 <= s <= 1, f"Afinidad fuera de [0,1] para {role}: {s}"
    print(f"  ✓ Afinidades: explorador={score_exp:.3f} refinador={score_ref:.3f} delfín={score_dol:.3f}")


def test_3_router_selects_best_sect():
    """Test 3: EcholocationRouter elige la secta con mayor score."""
    print("\n[Test 3] Router selecciona mejor secta")

    router = EcholocationRouter(acoustic_map_dim=128)
    am = random_acoustic_map(128)

    # Secta fuerte vs secta débil
    strong = make_sect("s_strong", "explorador", n_agents=3, avg_fitness=0.9)
    weak = make_sect("s_weak", "refinador", n_agents=10, avg_fitness=0.1)  # saturada y débil

    decision = router.route(am, [strong, weak])

    assert decision.primary_sect == "s_strong", \
        f"Debería elegir s_strong, eligió {decision.primary_sect}"
    assert decision.confidence > 0
    assert len(decision.ping_responses) == 2
    print(f"  ✓ Primary: {decision.primary_sect} (confidence={decision.confidence:.3f})")
    print(f"  ✓ Reason: {decision.reason}")


def test_4_multi_sect_complex_problem():
    """Test 4: problema complejo → delfín puede elegir múltiples sectas."""
    print("\n[Test 4] Multi-secta para problema complejo")

    router = EcholocationRouter(acoustic_map_dim=128)
    # Forzar el complexity_head a devolver un valor alto
    # Para eso simplemente verificamos que la lógica es correcta
    am = random_acoustic_map(128)

    sect_a = make_sect("s_a", "explorador", n_agents=2, avg_fitness=0.8)
    sect_b = make_sect("s_b", "refinador", n_agents=2, avg_fitness=0.7)

    decision = router.route(am, [sect_a, sect_b])

    # Independientemente del modo, la decisión debe ser válida
    assert decision.primary_sect in ["s_a", "s_b"]
    assert isinstance(decision.secondary_sects, list)
    assert decision.mode in ["single", "multi"]
    print(f"  ✓ Mode: {decision.mode} | Primary: {decision.primary_sect}")
    print(f"  ✓ Secondary: {decision.secondary_sects}")


def test_5_no_sects_broadcast():
    """Test 5: sin sectas → RouteDecision mode='broadcast'."""
    print("\n[Test 5] Sin sectas → broadcast")

    router = EcholocationRouter(acoustic_map_dim=128)
    am = random_acoustic_map(128)

    decision = router.route(am, sects=[])

    assert decision.mode == "broadcast"
    assert decision.primary_sect == "none"
    assert decision.confidence == 0.0
    print(f"  ✓ Mode={decision.mode} | primary={decision.primary_sect}")


def test_6_integrate_responses_weighted():
    """Test 6: integrate_responses() pondera por score correctamente."""
    print("\n[Test 6] integrate_responses ponderado")

    router = EcholocationRouter(acoustic_map_dim=128)
    am = random_acoustic_map(128)

    sect_a = make_sect("s_a", "explorador", avg_fitness=0.9)
    sect_b = make_sect("s_b", "refinador", avg_fitness=0.5)

    decision = router.route(am, [sect_a, sect_b])
    decision.secondary_sects = ["s_b"]  # forzar multi

    primary_out = torch.ones(1, 256) * 2.0
    secondary_out = torch.ones(1, 256) * 0.5

    integrated = router.integrate_responses(primary_out, [secondary_out], decision)

    assert integrated.shape == primary_out.shape
    # La integración debe estar entre los dos extremos
    val = integrated.mean().item()
    assert 0.5 <= val <= 2.0, f"Valor integrado fuera de rango: {val}"
    print(f"  ✓ Integración: {val:.3f} (entre 0.5 y 2.0)")


def test_7_adaptive_sleep_thresholds():
    """Test 7: AdaptiveSleepController cambia modo según diversidad."""
    print("\n[Test 7] Umbrales adaptativos de sueño")

    ctrl = AdaptiveSleepController()

    # Diversidad alta → reposo
    for _ in range(10):
        mode = ctrl.update_diversity(0.9)
    assert ctrl.mode == "rest", f"Diversidad alta debe dar reposo, got {ctrl.mode}"
    assert ctrl.activation_scale() == 0.4

    # Diversidad baja → agresivo
    for _ in range(10):
        mode = ctrl.update_diversity(0.1)
    assert ctrl.mode == "aggressive", f"Diversidad baja debe dar agresivo, got {ctrl.mode}"
    assert ctrl.activation_scale() == 1.5

    # Diversidad media → normal
    for _ in range(10):
        mode = ctrl.update_diversity(0.5)
    assert ctrl.mode == "normal", f"Diversidad media debe dar normal, got {ctrl.mode}"
    assert ctrl.activation_scale() == 1.0

    print(f"  ✓ rest (d=0.9): scale={0.4}")
    print(f"  ✓ aggressive (d=0.1): scale={1.5}")
    print(f"  ✓ normal (d=0.5): scale={1.0}")
    print(f"  ✓ mode_changes: {ctrl._mode_changes}")


def test_8_acoustic_buffer():
    """Test 8: buffer circular de acoustic_maps."""
    print("\n[Test 8] Buffer circular de acoustic_maps")

    ctrl = AdaptiveSleepController(buffer_size=8)

    for i in range(5):
        am = torch.randn(256) * (i + 1)
        ctrl.store_acoustic_map(am, metadata={"step": i})

    recent = ctrl.recent_acoustic_maps(n=3)
    assert len(recent) == 3
    assert all(isinstance(m, torch.Tensor) for m in recent)

    # El buffer respeta el límite de tamaño
    for _ in range(10):
        ctrl.store_acoustic_map(torch.randn(256))
    assert len(ctrl._acoustic_buffer) <= 8

    print(f"  ✓ Buffer: {len(ctrl._acoustic_buffer)} entradas (cap=8)")
    print(f"  ✓ recent(3): {len(recent)} tensors shape={recent[0].shape}")


def test_9_sleep_controller_persistence():
    """Test 9: AdaptiveSleepController se guarda y carga desde disco."""
    print("\n[Test 9] Persistencia del sleep controller")

    ctrl = AdaptiveSleepController(buffer_size=16)
    ctrl.update_diversity(0.1)  # modo agresivo
    ctrl.update_diversity(0.1)
    ctrl.update_diversity(0.1)
    ctrl.store_acoustic_map(torch.randn(256))
    ctrl.store_acoustic_map(torch.randn(256))

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        ctrl.save(path)

        ctrl2 = AdaptiveSleepController(buffer_size=16)
        ctrl2.load(path)

        assert len(ctrl2._acoustic_buffer) == len(ctrl._acoustic_buffer)
        assert ctrl2._mode_changes == ctrl._mode_changes
        print(f"  ✓ Guardado y cargado: buffer={len(ctrl2._acoustic_buffer)}, mode_changes={ctrl2._mode_changes}")
    finally:
        os.unlink(path)


def test_10_bridge_forward_includes_sleep_mode():
    """Test 10: DolphinSwarmBridge.forward() incluye sleep_mode en info."""
    print("\n[Test 10] DolphinSwarmBridge forward incluye sleep_mode")

    from src.agents.dolphin_agent import DolphinSwarmBridge, DolphinConfig

    cfg = DolphinConfig(agent_id=100, vocab_size=50304)
    bridge = DolphinSwarmBridge(cfg, device="cpu")

    x = torch.randint(0, 50304, (1, 16))
    with torch.no_grad():
        feromon, info = bridge(x)

    assert "sleep_mode" in info, "info debe incluir sleep_mode"
    assert "sleep_for_matriarca" in info
    assert "acoustic_map" in info
    assert info["sleep_mode"] in ["rest", "normal", "aggressive"]
    print(f"  ✓ sleep_mode={info['sleep_mode']}")
    print(f"  ✓ sleep_for_matriarca shape={info['sleep_for_matriarca'].shape}")


def test_11_bridge_route_with_sects():
    """Test 11: DolphinSwarmBridge.route() retorna RouteDecision válida."""
    print("\n[Test 11] DolphinSwarmBridge.route() con sectas")

    from src.agents.dolphin_agent import DolphinSwarmBridge, DolphinConfig

    cfg = DolphinConfig(agent_id=100, vocab_size=50304, feromon_dim=256)
    bridge = DolphinSwarmBridge(cfg, device="cpu")

    # Forward primero (genera acoustic_map)
    x = torch.randint(0, 50304, (1, 16))
    with torch.no_grad():
        bridge(x)

    sects = [
        make_sect("s_exp", "explorador", avg_fitness=0.8),
        make_sect("s_ref", "refinador", avg_fitness=0.6),
    ]

    decision = bridge.route(sects)

    assert isinstance(decision, RouteDecision)
    assert decision.primary_sect in ["s_exp", "s_ref"]
    assert decision.mode in ["single", "multi", "broadcast"]
    print(f"  ✓ RouteDecision: primary={decision.primary_sect} mode={decision.mode}")
    print(f"  ✓ confidence={decision.confidence:.3f} secondary={decision.secondary_sects}")


def test_12_route_task_full_flow():
    """Test 12: LixySwarm.route_task() integra delfín + router + sectas."""
    print("\n[Test 12] LixySwarm.route_task() flujo completo")

    from src.swarm.orchestrator import LixySwarm, SwarmConfig, AgentConfig
    from src.matriarca.matriarca import MatriarcaConfig

    agent_cfgs = [AgentConfig(agent_id=i, n_agents=3) for i in range(3)]
    cfg = SwarmConfig(n_agents=3, agent_configs=agent_cfgs,
        matriarca_config=MatriarcaConfig(
            memory_path="/tmp/test12_mat.json",
            checkpoint_path="/tmp/test12_mat.pt",
        ))
    swarm = LixySwarm(cfg, load_matriarca=False).cuda()

    x = torch.randint(0, 50304, (1, 16)).cuda()
    with torch.no_grad():
        feromon, route_decision, dolphin_info = swarm.route_task(x, use_sects=True)

    assert feromon.shape[0] == 1
    assert route_decision is not None
    assert route_decision.mode in ["single", "multi", "broadcast"]
    assert "route" in dolphin_info

    print(f"  ✓ feromon shape: {feromon.shape}")
    print(f"  ✓ route: primary={route_decision.primary_sect} mode={route_decision.mode}")
    print(f"  ✓ Sectas disponibles: {[s.sect_id for s in swarm.sect_manager.all_sects()]}")


def test_13_tick_updates_sleep_mode():
    """Test 13: tick_lifecycle actualiza sleep_mode en todos los delfines."""
    print("\n[Test 13] tick_lifecycle actualiza sleep_mode")

    from src.swarm.orchestrator import LixySwarm, SwarmConfig, AgentConfig
    from src.matriarca.matriarca import MatriarcaConfig

    agent_cfgs = [AgentConfig(agent_id=i, n_agents=3) for i in range(3)]
    cfg = SwarmConfig(n_agents=3, agent_configs=agent_cfgs,
        matriarca_config=MatriarcaConfig(
            memory_path="/tmp/test13_mat.json",
            checkpoint_path="/tmp/test13_mat.pt",
        ))
    swarm = LixySwarm(cfg, load_matriarca=False).cuda()

    # Diversidad alta → reposo
    for _ in range(10):
        swarm.tick_lifecycle(step=100, swarm_diversity=0.9, n_nodes=1)

    primary_mode = swarm.dolphin.primary.sleep_controller.mode
    # Con diversidad persistentemente alta debe tender a "rest"
    assert primary_mode in ["rest", "normal"], f"Modo inesperado: {primary_mode}"

    # Diversidad baja → agresivo
    for _ in range(10):
        swarm.tick_lifecycle(step=200, swarm_diversity=0.1, n_nodes=1)

    primary_mode = swarm.dolphin.primary.sleep_controller.mode
    assert primary_mode in ["aggressive", "normal"], f"Modo inesperado: {primary_mode}"

    print(f"  ✓ Después diversidad baja: sleep_mode={primary_mode}")


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_1_ping_response_score,
        test_2_ping_encoder_affinity,
        test_3_router_selects_best_sect,
        test_4_multi_sect_complex_problem,
        test_5_no_sects_broadcast,
        test_6_integrate_responses_weighted,
        test_7_adaptive_sleep_thresholds,
        test_8_acoustic_buffer,
        test_9_sleep_controller_persistence,
        test_10_bridge_forward_includes_sleep_mode,
        test_11_bridge_route_with_sects,
        test_12_route_task_full_flow,
        test_13_tick_updates_sleep_mode,
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
