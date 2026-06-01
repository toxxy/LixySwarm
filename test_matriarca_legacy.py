"""
Tests para 1d: Matriarca — Legado de Sectas + Arquitectura Dual
===============================================================
Test  1: SectLegacyRecord — campos correctos, props calculadas
Test  2: SectLegacyBank.add() + save() + load()
Test  3: SectLegacyBank.query() por role_type
Test  4: SectLegacyBank.query() por embedding similitud
Test  5: MatriarcaEnriched.store_sect_legacy() almacena en bank + matriarca base
Test  6: MatriarcaEnriched.query_sect_history() orienta spawn
Test  7: MatriarcaEnriched.suggest_bifurcation() — fitness alto + baja diversidad
Test  8: MatriarcaEnriched.suggest_bifurcation() — NO bifurcar cuando fitness bajo
Test  9: SectManager.kill_sect() usa MatriarcaEnriched cuando disponible
Test 10: SectRecord tiene parent_sect_id, children_sect_ids, n_agents_peak, born_at
Test 11: MatriarcaDual emit_combined (Personal + Global)
Test 12: MatriarcaDual.merge_global_update() fusiona conocimiento remoto
Test 13: LixySwarm usa MatriarcaEnriched por defecto
Test 14: route_task() + kill_sect() flujo completo integrado
Test 15: 15/15 integration tests base siguen pasando
"""

import sys, os, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import time

from src.matriarca.matriarca_legacy import (
    SectLegacyRecord, SectLegacyBank, MatriarcaDual, MatriarcaEnriched,
)
from src.matriarca.matriarca import Matriarca, MatriarcaConfig
from src.swarm.sect_manager import SectRecord, SectManager, AgentSlot


# ─── Helpers ──────────────────────────────────────────────────────────────────

def fresh_matriarca(tmpdir, name="mat") -> Matriarca:
    cfg = MatriarcaConfig(
        memory_path=f"{tmpdir}/{name}.json",
        checkpoint_path=f"{tmpdir}/{name}.pt",
    )
    return Matriarca(cfg, device="cpu")


def make_sect(sect_id, role_type, fitness=0.7, n=2) -> SectRecord:
    sect = SectRecord(sect_id=sect_id, role_type=role_type)
    for i in range(n):
        sect.agents.append(AgentSlot(agent_id=i, node_id="local", fitness=fitness))
    sect.fitness_history = [fitness * 0.9, fitness, fitness * 1.1]
    return sect


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_1_sect_legacy_record():
    """Test 1: SectLegacyRecord campos correctos."""
    print("\n[Test 1] SectLegacyRecord campos")

    r = SectLegacyRecord(
        sect_id="s1",
        role_type="explorador",
        parent_sect_id=None,
        born_at=time.time() - 3600,
        died_at=time.time(),
        death_reason="low_fitness",
        fitness_history=[0.3, 0.4, 0.35, 0.28, 0.25, 0.2, 0.22, 0.18],
        peak_fitness=0.4,
        final_fitness=0.18,
        n_agents_peak=5,
        n_agents_final=2,
    )

    assert r.lifespan_s > 3590
    assert r.fitness_avg > 0
    assert r.fitness_trend < 0, f"Tendencia debe ser negativa: {r.fitness_trend}"
    summary = r.to_summary()
    assert "explorador" in summary
    assert "low_fitness" in summary
    print(f"  ✓ lifespan={r.lifespan_s:.0f}s trend={r.fitness_trend:.3f}")
    print(f"  ✓ summary: {summary}")


def test_2_legacy_bank_persist():
    """Test 2: SectLegacyBank.add() + save() + load()."""
    print("\n[Test 2] SectLegacyBank persistencia")

    with tempfile.TemporaryDirectory() as d:
        bank = SectLegacyBank(f"{d}/legacy.json", embd_dim=512)

        r = SectLegacyRecord(
            sect_id="s1", role_type="explorador",
            final_fitness=0.6, fitness_history=[0.5, 0.6, 0.65],
        )
        bank.add(r)
        assert bank.size == 1

        # Recargar
        bank2 = SectLegacyBank(f"{d}/legacy.json", embd_dim=512)
        assert bank2.size == 1
        assert bank2.records[0].role_type == "explorador"
        print(f"  ✓ Guardado y cargado: {bank2.size} registro")


def test_3_legacy_bank_query_role():
    """Test 3: query() por role_type."""
    print("\n[Test 3] SectLegacyBank query por role_type")

    with tempfile.TemporaryDirectory() as d:
        bank = SectLegacyBank(f"{d}/legacy.json", embd_dim=512)
        for i in range(4):
            r = SectLegacyRecord(sect_id=f"s{i}", role_type="explorador" if i < 3 else "refinador",
                                 final_fitness=0.5 + i * 0.1)
            bank.add(r)

        exp = bank.query(role_type="explorador", top_k=10)
        ref = bank.query(role_type="refinador", top_k=10)

        assert len(exp) == 3, f"Esperaba 3 exploradores, got {len(exp)}"
        assert len(ref) == 1, f"Esperaba 1 refinador, got {len(ref)}"
        print(f"  ✓ explorador: {len(exp)} | refinador: {len(ref)}")


def test_4_legacy_bank_query_embedding():
    """Test 4: query() por similitud de embedding."""
    print("\n[Test 4] SectLegacyBank query por embedding")

    with tempfile.TemporaryDirectory() as d:
        bank = SectLegacyBank(f"{d}/legacy.json", embd_dim=512)
        for i in range(5):
            r = SectLegacyRecord(sect_id=f"s{i}", role_type="explorador", final_fitness=0.4 + i * 0.1)
            emb = torch.zeros(512)
            emb[i] = 1.0  # cada uno tiene un vector único
            bank.add(r, embedding=emb)

        # Query con vector parecido al tercero
        query_emb = torch.zeros(512)
        query_emb[2] = 0.95
        results = bank.query(embedding=query_emb, top_k=2)

        assert len(results) == 2
        # El más similar debe ser el tercero (index 2)
        best, score = results[0]
        assert best.sect_id == "s2", f"El más similar debe ser s2, got {best.sect_id}"
        print(f"  ✓ Mejor match: {best.sect_id} (score={score:.3f})")


def test_5_enriched_store_legacy():
    """Test 5: MatriarcaEnriched.store_sect_legacy()."""
    print("\n[Test 5] MatriarcaEnriched store_sect_legacy")

    with tempfile.TemporaryDirectory() as d:
        mat = fresh_matriarca(d)
        enriched = MatriarcaEnriched(mat,
            legacy_bank=SectLegacyBank(f"{d}/legacy.json", embd_dim=mat.cfg.embd_dim))

        assert enriched.legacy_count == 0
        sect = make_sect("s1", "explorador", fitness=0.75)
        record = enriched.store_sect_legacy(sect, reason="test")

        assert enriched.legacy_count == 1
        assert record.role_type == "explorador"
        assert record.final_fitness == 0.75
        assert record.death_reason == "test"
        # También almacenado en el banco general de Matriarca
        assert enriched.memory_count > 1  # +1 el resumen
        print(f"  ✓ legacy_count={enriched.legacy_count} memory_count={enriched.memory_count}")
        print(f"  ✓ record: {record.to_summary()}")


def test_6_enriched_query_history():
    """Test 6: query_sect_history() orienta spawn."""
    print("\n[Test 6] MatriarcaEnriched query_sect_history")

    with tempfile.TemporaryDirectory() as d:
        mat = fresh_matriarca(d)
        enriched = MatriarcaEnriched(mat,
            legacy_bank=SectLegacyBank(f"{d}/legacy.json", embd_dim=mat.cfg.embd_dim))

        for i in range(5):
            sect = make_sect(f"s{i}", "explorador" if i < 4 else "refinador", fitness=0.5 + i * 0.1)
            enriched.store_sect_legacy(sect)

        hist = enriched.query_sect_history("explorador", top_k=3)
        assert 1 <= len(hist) <= 3
        assert all(r.role_type == "explorador" for r in hist)

        hist_ref = enriched.query_sect_history("refinador")
        assert len(hist_ref) >= 1
        print(f"  ✓ explorador: {len(hist)} | refinador: {len(hist_ref)}")


def test_7_suggest_bifurcation_yes():
    """Test 7: suggest_bifurcation — fitness alto + baja diversidad → bifurcar."""
    print("\n[Test 7] suggest_bifurcation → SÍ")

    with tempfile.TemporaryDirectory() as d:
        mat = fresh_matriarca(d)
        enriched = MatriarcaEnriched(mat)
        sect = make_sect("s1", "refinador", fitness=0.8)

        sugg = enriched.suggest_bifurcation(sect, swarm_diversity=0.25)
        assert sugg["should_bifurcate"] is True
        assert "refinador-logico" in sugg["child_roles"]
        assert "refinador-creativo" in sugg["child_roles"]
        assert sugg["confidence"] > 0
        print(f"  ✓ should=True roles={sugg['child_roles']} conf={sugg['confidence']:.2f}")


def test_8_suggest_bifurcation_no():
    """Test 8: suggest_bifurcation — fitness bajo → NO bifurcar."""
    print("\n[Test 8] suggest_bifurcation → NO")

    with tempfile.TemporaryDirectory() as d:
        mat = fresh_matriarca(d)
        enriched = MatriarcaEnriched(mat)
        sect = make_sect("s1", "refinador", fitness=0.4)  # fitness bajo

        sugg = enriched.suggest_bifurcation(sect, swarm_diversity=0.5)  # diversidad ok
        assert sugg["should_bifurcate"] is False
        print(f"  ✓ should=False conf={sugg['confidence']:.2f} reason={sugg['reason']}")


def test_9_sect_manager_uses_enriched():
    """Test 9: SectManager.kill_sect() usa MatriarcaEnriched cuando disponible."""
    print("\n[Test 9] SectManager + MatriarcaEnriched")

    with tempfile.TemporaryDirectory() as d:
        mat = fresh_matriarca(d)
        enriched = MatriarcaEnriched(mat,
            legacy_bank=SectLegacyBank(f"{d}/legacy.json", embd_dim=mat.cfg.embd_dim))

        sm = SectManager(matriarca=enriched)
        sect = sm.spawn_sect("explorador")
        assert sect is not None

        event = sm.kill_sect(sect.sect_id, reason="test")
        assert event["legacy_stored"] is True
        assert enriched.legacy_count == 1
        print(f"  ✓ legacy_stored=True legacy_count={enriched.legacy_count}")


def test_10_sect_record_new_fields():
    """Test 10: SectRecord tiene nuevos campos genéticos."""
    print("\n[Test 10] SectRecord campos genéticos")

    sect = SectRecord(sect_id="s1", role_type="refinador")

    # Nuevos campos con defaults
    assert sect.parent_sect_id is None
    assert isinstance(sect.children_sect_ids, list)
    assert sect.n_agents_peak == 0
    assert sect.born_at == sect.created_at

    # n_agents_peak se actualiza al añadir agentes
    from src.swarm.sect_manager import SectManager
    sm = SectManager()
    sect2 = sm.spawn_sect("explorador")
    sm.add_agent_to_sect(sect2.sect_id, "local")
    sm.add_agent_to_sect(sect2.sect_id, "local2")
    assert sect2.n_agents_peak == 2

    print(f"  ✓ born_at alias: {sect.born_at > 0}")
    print(f"  ✓ n_agents_peak tracking: {sect2.n_agents_peak}")


def test_11_matriarca_dual():
    """Test 11: MatriarcaDual emit_combined."""
    print("\n[Test 11] MatriarcaDual emit_combined")

    with tempfile.TemporaryDirectory() as d:
        personal_cfg = MatriarcaConfig(memory_path=f"{d}/personal.json", checkpoint_path=f"{d}/personal.pt")
        global_cfg = MatriarcaConfig(memory_path=f"{d}/global.json", checkpoint_path=f"{d}/global.pt")

        dual = MatriarcaDual(personal_cfg, global_cfg, device="cpu")
        state = torch.randn(512)

        inf = dual.emit_combined(state, personal_weight=0.4, global_weight=0.6)
        assert inf.shape == torch.Size([256]), f"Shape inesperado: {inf.shape}"

        # Almacenar en ambas
        dual.store_personal(state, "test personal", importance=0.9)
        dual.store_global(state, "test global", importance=0.8)
        assert dual.personal_memories > 0
        assert dual.global_memories > 0
        print(f"  ✓ emit shape={inf.shape} personal={dual.personal_memories} global={dual.global_memories}")


def test_12_matriarca_dual_merge():
    """Test 12: MatriarcaDual.merge_global_update() fusiona conocimiento remoto."""
    print("\n[Test 12] MatriarcaDual merge_global_update")

    with tempfile.TemporaryDirectory() as d:
        personal_cfg = MatriarcaConfig(memory_path=f"{d}/p.json", checkpoint_path=f"{d}/p.pt")
        global_cfg = MatriarcaConfig(memory_path=f"{d}/g.json", checkpoint_path=f"{d}/g.pt")
        dual = MatriarcaDual(personal_cfg, global_cfg, device="cpu")

        before = dual.global_memories
        remote_embs = torch.randn(3, 512)
        remote_meta = [
            {"text": "remoto 1", "importance": 0.7},
            {"text": "remoto 2", "importance": 0.6},
            {"text": "remoto 3", "importance": 0.5},
        ]
        dual.merge_global_update(remote_embs, remote_meta)
        after = dual.global_memories

        assert after > before, f"Debería haber más memorias: {before} → {after}"
        print(f"  ✓ global memories: {before} → {after} (+{after - before})")


def test_13_swarm_uses_enriched():
    """Test 13: LixySwarm usa MatriarcaEnriched por defecto."""
    print("\n[Test 13] LixySwarm usa MatriarcaEnriched")

    from src.swarm.orchestrator import LixySwarm, SwarmConfig, AgentConfig

    with tempfile.TemporaryDirectory() as d:
        agent_cfgs = [AgentConfig(agent_id=i, n_agents=3) for i in range(3)]
        cfg = SwarmConfig(n_agents=3, agent_configs=agent_cfgs,
            matriarca_config=MatriarcaConfig(
                memory_path=f"{d}/mat.json",
                checkpoint_path=f"{d}/mat.pt",
            ))
        swarm = LixySwarm(cfg, load_matriarca=True).cuda()

        assert isinstance(swarm.matriarca, MatriarcaEnriched)
        assert swarm.matriarca.legacy_count >= 0
        print(f"  ✓ MatriarcaEnriched: memories={swarm.matriarca.memory_count} legados={swarm.matriarca.legacy_count}")


def test_14_full_integration():
    """Test 14: route_task() + kill_sect() + query_sect_history() integrado."""
    print("\n[Test 14] Flujo completo integrado")

    import torch
    from src.swarm.orchestrator import LixySwarm, SwarmConfig, AgentConfig

    with tempfile.TemporaryDirectory() as d:
        agent_cfgs = [AgentConfig(agent_id=i, n_agents=3) for i in range(3)]
        cfg = SwarmConfig(n_agents=3, agent_configs=agent_cfgs,
            matriarca_config=MatriarcaConfig(
                memory_path=f"{d}/mat.json",
                checkpoint_path=f"{d}/mat.pt",
            ))
        swarm = LixySwarm(cfg, load_matriarca=True).cuda()

        # 1. route_task — el delfín enruta
        x = torch.randint(0, 50304, (1, 16)).cuda()
        with torch.no_grad():
            feromon, route, dolphin_info = swarm.route_task(x, use_sects=True)

        assert route is not None
        assert route.mode in ["single", "multi", "broadcast"]

        # 2. Matar una secta → legado guardado
        sects = swarm.sect_manager.all_sects()
        if sects:
            event = swarm.sect_manager.kill_sect(sects[0].sect_id, reason="validation")
            assert event["legacy_stored"]
            assert swarm.matriarca.legacy_count >= 1

        # 3. Consultar historia antes de spawn
        hist = swarm.matriarca.query_sect_history(sects[0].role_type)
        assert isinstance(hist, list)

        print(f"  ✓ route={route.primary_sect} mode={route.mode}")
        print(f"  ✓ legacy_count={swarm.matriarca.legacy_count}")
        print(f"  ✓ history({sects[0].role_type})={len(hist)} legados")


def test_15_integration_base():
    """Test 15: 15/15 integration tests base siguen pasando."""
    print("\n[Test 15] Integration tests base (subprocess)")
    import subprocess
    result = subprocess.run(
        ["python3", "test_integration.py"],
        capture_output=True, text=True,
        cwd="/home/toxxy/Dropbox/Lixy/clawd/workspace/lixy-llm"
    )
    output = result.stdout + result.stderr
    assert "15/15" in output, f"Integration tests fallaron:\n{output[-500:]}"
    print(f"  ✓ 15/15 base integration tests pasaron")


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_1_sect_legacy_record,
        test_2_legacy_bank_persist,
        test_3_legacy_bank_query_role,
        test_4_legacy_bank_query_embedding,
        test_5_enriched_store_legacy,
        test_6_enriched_query_history,
        test_7_suggest_bifurcation_yes,
        test_8_suggest_bifurcation_no,
        test_9_sect_manager_uses_enriched,
        test_10_sect_record_new_fields,
        test_11_matriarca_dual,
        test_12_matriarca_dual_merge,
        test_13_swarm_uses_enriched,
        test_14_full_integration,
        test_15_integration_base,
    ]

    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print("  → PASS")
        except Exception as e:
            failed += 1
            import traceback
            print(f"  → FAIL: {e}")
            traceback.print_exc()

    print(f"\n{'='*52}")
    print(f"Tests 1d: {passed}/{len(tests)} passed", "✅" if failed == 0 else "❌")
    if failed > 0:
        sys.exit(1)
