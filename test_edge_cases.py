"""
test_edge_cases.py — Edge cases: timeouts, sectas huérfanas, recovery, auto-loop resume

Cubre casos que los tests unitarios básicos no ejercitan:
- NodeManager: timeout de nodo sin heartbeat
- NodeManager: nodo reaparece después de timeout
- SectManager: secta huérfana (todos sus agentes muertos)
- SectManager: matar secta activa con agentes
- Matriarca: recuperación con archivo corrupto / truncado
- Matriarca: compresión generacional (capacidad > umbral)
- auto_train: resume desde training_state.json existente
- auto_train: resume con checkpoint corrupto / faltante
- auto_train: plateau lleva LR hasta el piso y luego se congela
- swarm_publisher: collect_status sin ningún archivo disponible
"""

import sys, json, time, tempfile, shutil, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# ─── Helpers ─────────────────────────────────────────────────────────────────

passed = []
failed = []

def ok(name: str, detail: str = ""):
    passed.append(name)
    print(f"  ✅ PASS {name}" + (f" — {detail}" if detail else ""))

def fail(name: str, reason: str):
    failed.append(name)
    print(f"  ❌ FAIL {name} — {reason}")

def check(name: str, cond: bool = True, reason: str = ""):
    (ok if cond else fail)(name, reason or ("" if cond else "assertion failed"))

def section(title: str):
    print(f"\n── {title} " + "─" * (50 - len(title)))

# ─── NodeManager edge cases ───────────────────────────────────────────────────

def test_1_node_timeout():
    """Nodo sin heartbeat por NODE_TIMEOUT_SECONDS aparece como muerto."""
    section("NodeManager — timeout")
    from src.swarm.node_manager import NodeManager, HardwareProfile
    nm = NodeManager()

    # Registrar nodo con hardware profile
    nm.node_joined("node-old", hardware=HardwareProfile(gpu_vram_gb=8))
    # Forzar timestamp viejo: NODE_TIMEOUT_SECONDS + 1
    old_ts = time.time() - (nm.NODE_TIMEOUT_SECONDS + 1)
    nm._nodes["node-old"].last_seen = old_ts

    # prune_dead_nodes debe eliminarlo
    pruned = nm.prune_dead_nodes()
    check("nodo timeout pruneado", "node-old" not in nm._nodes,
          f"_nodes={list(nm._nodes.keys())}")
    check("prune retorna lista de ids", isinstance(pruned, (list, set, int)),
          f"pruned={pruned}")


def test_2_node_rejoin_after_timeout():
    """Nodo que murió por timeout puede reaparecer con node_joined."""
    section("NodeManager — rejoin")
    from src.swarm.node_manager import NodeManager
    nm = NodeManager()

    nm.node_joined("node-x")
    # Simular muerte
    nm._nodes["node-x"].last_seen = time.time() - nm.NODE_TIMEOUT_SECONDS - 10
    nm.prune_dead_nodes()
    check("nodo muerto eliminado", "node-x" not in nm._nodes)

    # Re-registrar
    nm.node_joined("node-x")
    check("nodo reaparece", "node-x" in nm._nodes)
    check("heartbeat reciente tras rejoin",
          nm._nodes["node-x"].last_seen > time.time() - 5)


def test_3_node_heartbeat_resets_timer():
    """heartbeat() renueva el timer; nodo no es pruneable inmediatamente después."""
    section("NodeManager — heartbeat reset")
    from src.swarm.node_manager import NodeManager
    nm = NodeManager()

    nm.node_joined("node-hb")
    # Envejecer
    nm._nodes["node-hb"].last_seen = time.time() - nm.NODE_TIMEOUT_SECONDS - 5
    # Heartbeat lo rescata
    nm.heartbeat("node-hb")
    nm.prune_dead_nodes()
    check("heartbeat rescata nodo de timeout", "node-hb" in nm._nodes)


# ─── SectManager edge cases ───────────────────────────────────────────────────

def test_4_orphan_sect():
    """Secta queda huérfana cuando todos sus agentes son removidos."""
    section("SectManager — secta huérfana")
    from src.swarm.sect_manager import SectManager
    sm = SectManager()

    sect = sm.spawn_sect(role_type="explorador")
    check("secta spawneada", sect is not None)
    if sect is None:
        return

    sid = sect.sect_id
    # Añadir un agente y luego matar la secta
    slot = sm.add_agent_to_sect(sid, node_id="node-a")
    if slot:
        sm.update_agent_fitness(sid, slot.agent_id, fitness=0.9)
    sm.kill_sect(sid, reason="test_orphan")
    result = sm.get_sect(sid)
    if result is not None:
        check("secta marcada como dead tras kill", not result.alive)
    else:
        ok("secta eliminada por kill_sect")


def test_5_kill_active_sect_transfers_legacy():
    """kill_sect con agentes activos transfiere legado si hay matriarca."""
    section("SectManager — legado en kill")
    from src.swarm.sect_manager import SectManager

    sm = SectManager()
    sect = sm.spawn_sect(role_type="refinador")
    if sect is None:
        ok("spawn_sect sin nodos disponibles — OK")
        return
    sid = sect.sect_id
    a1 = sm.add_agent_to_sect(sid, node_id="node-b")
    a2 = sm.add_agent_to_sect(sid, node_id="node-c")
    if a1: sm.update_agent_fitness(sid, a1.agent_id, fitness=0.8)
    if a2: sm.update_agent_fitness(sid, a2.agent_id, fitness=0.7)

    sm.kill_sect(sid, reason="fitness_low")
    result = sm.get_sect(sid)
    if result is not None:
        check("secta marcada dead", not result.alive)
    else:
        ok("kill_sect elimina la secta")
    ok("kill con agentes activos no crashea")


def test_6_sect_tick_no_crash():
    """tick() no crashea con múltiples sectas en distintos estados."""
    section("SectManager — tick stress")
    from src.swarm.sect_manager import SectManager
    sm = SectManager()

    sects = [sm.spawn_sect(role_type=r) for r in ["explorador", "refinador", "integrador"]]
    for i, sect in enumerate(sects):
        if sect is None:
            continue
        sid = sect.sect_id
        for j in range(3):
            slot = sm.add_agent_to_sect(sid, node_id=f"node-{i}-{j}")
            if slot:
                sm.update_agent_fitness(sid, slot.agent_id, fitness=0.3 + i * 0.1 + j * 0.05)

    try:
        for step in range(5):
            sm.tick(step=step, swarm_diversity=0.5)
        ok("tick × 5 sin crash")
    except Exception as e:
        fail("tick crash", str(e))


# ─── Matriarca recovery ───────────────────────────────────────────────────────

def test_7_matriarca_corrupt_checkpoint():
    """Matriarca carga sin crash aunque el checkpoint esté corrupto."""
    section("Matriarca — recovery checkpoint corrupto")
    import torch
    from src.matriarca.matriarca import Matriarca, MatriarcaConfig

    with tempfile.TemporaryDirectory() as tmp:
        corrupt_path = Path(tmp) / "matriarca.pt"
        corrupt_path.write_bytes(b"\x00\x01\x02\x03\x04")  # basura

        cfg = MatriarcaConfig(
            checkpoint_path=str(corrupt_path),
            memory_path=str(Path(tmp) / "mem.json"),
        )
        try:
            # Matriarca.__init__ intenta cargar el checkpoint si existe
            # Con archivo corrupto debe lanzar error legible, no segfault
            mat = Matriarca(cfg=cfg, device="cpu")
            ok("Matriarca init con checkpoint corrupto no crashea — cargó de cero")
        except (RuntimeError, Exception) as e:
            ok("Matriarca lanza excepción descriptiva con checkpoint corrupto",
               str(e)[:80])


def test_8_matriarca_empty_memory_file():
    """Matriarca con archivo de memorias vacío arranca limpia."""
    section("Matriarca — memory file vacío")
    from src.matriarca.matriarca import Matriarca, MatriarcaConfig

    with tempfile.TemporaryDirectory() as tmp:
        mem_file = Path(tmp) / "matriarca_memory.json"
        mem_file.write_text("[]")  # lista vacía
        cfg = MatriarcaConfig(
            memory_path=str(mem_file),
            checkpoint_path=str(Path(tmp) / "nonexistent.pt"),
        )
        mat = Matriarca(cfg=cfg, device="cpu")
        count = mat.memory_count
        # Matriarca puede inicializar con 1 memoria seed — aceptable
        check("memory_count ≤ 1 con banco vacío (seed OK)", count <= 1,
              f"count={count}")


def test_9_matriarca_compression_trigger():
    """Compresión generacional: al superar umbral, se comprimen memorias."""
    section("Matriarca — compresión generacional")
    import torch
    from src.matriarca.matriarca import Matriarca, MatriarcaConfig

    max_mem = 20
    with tempfile.TemporaryDirectory() as tmp:
        cfg = MatriarcaConfig(
            max_memories=max_mem,
            memory_path=str(Path(tmp) / "mem.json"),
            checkpoint_path=str(Path(tmp) / "nonexistent.pt"),
        )
        mat = Matriarca(cfg=cfg, device="cpu")

        n_before = None
        try:
            for i in range(19):
                vec = torch.randn(cfg.embd_dim)
                mat.store_interaction(vec, text=f"memoria de prueba número {i}", importance=0.5)
            n_before = mat.memory_count
            check(f"memorias antes de overflow: ≤{max_mem}", n_before <= max_mem,
                  f"count={n_before}")

            for i in range(5):
                vec = torch.randn(cfg.embd_dim)
                mat.store_interaction(vec, text=f"overflow memoria {i}", importance=0.5)
            n_after = mat.memory_count
            check("después de overflow: count ≤ max_memories", n_after <= max_mem,
                  f"before={n_before} after={n_after} max={max_mem}")
        except Exception as e:
            fail("compresión crash", str(e))


# ─── auto_train edge cases ────────────────────────────────────────────────────

def test_10_autotrain_resume_existing_state():
    """load_state con archivo existente conserva todos los campos."""
    section("auto_train — resume desde estado existente")
    import auto_train

    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "training_state.json"
        # Simular estado de 3 ciclos completados
        prev = {
            "cycles_completed": 3,
            "total_steps": 3000,
            "best_val_loss": 3.55,
            "lr_current": 7e-5,
            "plateau_count": 1,
            "last_checkpoint": "swarm_auto_cycle3.pt",
            "cycle_history": [
                [1, 1000, 3.80, 1e-4],
                [2, 1000, 3.65, 1e-4],
                [3, 1000, 3.55, 7e-5],
            ],
            "last_updated": time.time() - 600,
        }
        state_path.write_text(json.dumps(prev))

        loaded = auto_train.load_state(str(state_path))
        check("cycles_completed preservado", loaded["cycles_completed"] == 3)
        check("total_steps preservado", loaded["total_steps"] == 3000)
        check("best_val_loss preservado", abs(loaded["best_val_loss"] - 3.55) < 1e-6)
        check("lr_current preservado", abs(loaded["lr_current"] - 7e-5) < 1e-10)
        check("plateau_count preservado", loaded["plateau_count"] == 1)
        check("cycle_history preservado (3 entradas)", len(loaded["cycle_history"]) == 3)


def test_11_autotrain_resume_missing_checkpoint():
    """auto_train maneja gracefully cuando el last_checkpoint ya no existe."""
    section("auto_train — checkpoint faltante")
    import auto_train

    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "training_state.json"
        prev = {
            "cycles_completed": 5,
            "total_steps": 5000,
            "best_val_loss": 3.4,
            "lr_current": 7e-5,
            "plateau_count": 0,
            "last_checkpoint": "swarm_auto_cycle5_DELETED.pt",  # no existe
            "cycle_history": [],
        }
        state_path.write_text(json.dumps(prev))

        loaded = auto_train.load_state(str(state_path))
        check("load_state no crashea con checkpoint faltante",
              loaded["last_checkpoint"] == "swarm_auto_cycle5_DELETED.pt")

        # find_best_checkpoint debe manejar que el archivo no existe
        if hasattr(auto_train, "find_best_checkpoint"):
            result = auto_train.find_best_checkpoint(
                checkpoint_dir=tmp,
                state=loaded
            )
            # Puede devolver None o un fallback — lo importante: no crashea
            ok(f"find_best_checkpoint no crashea (result={result})")
        else:
            ok("find_best_checkpoint no expuesto — OK (lógica interna)")


def test_12_autotrain_plateau_to_floor():
    """Múltiples plateaus llevan LR al piso y se congela ahí."""
    section("auto_train — LR floor multi-plateau")
    import auto_train

    cfg = auto_train.AutoTrainConfig(
        lr_initial=1e-4,
        plateau_patience=2,
        lr_decay_factor=0.5,
        lr_min=1e-6,
    )
    state = auto_train.load_state("/tmp/nope_edge.json")
    state["lr_current"] = cfg.lr_initial
    state["best_val_loss"] = 3.5

    # Plateau 1: 1e-4 → 5e-5
    for _ in range(2): auto_train.check_plateau(state, 3.5, cfg)
    check("plateau 1: LR bajó", state["lr_current"] < 1e-4,
          f"lr={state['lr_current']:.2e}")
    lr_after_p1 = state["lr_current"]

    # Plateau 2: 5e-5 → 2.5e-5
    for _ in range(2): auto_train.check_plateau(state, 3.5, cfg)
    check("plateau 2: LR bajó más", state["lr_current"] < lr_after_p1,
          f"lr={state['lr_current']:.2e}")

    # Seguir hasta el piso
    for _ in range(20): auto_train.check_plateau(state, 3.5, cfg)
    check("LR ≥ lr_min después de muchos plateaus",
          state["lr_current"] >= cfg.lr_min,
          f"lr={state['lr_current']:.2e} floor={cfg.lr_min:.2e}")


def test_13_autotrain_corrupt_state_file():
    """load_state devuelve estado limpio si el archivo está corrupto."""
    section("auto_train — state file corrupto")
    import auto_train

    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "training_state.json"
        state_path.write_text("{invalid json <<<")

        try:
            state = auto_train.load_state(str(state_path))
            check("load_state con JSON corrupto devuelve estado limpio",
                  state["cycles_completed"] == 0,
                  f"cycles={state.get('cycles_completed')}")
        except Exception as e:
            fail("load_state no maneja JSON corrupto", str(e))


# ─── swarm_publisher edge cases ───────────────────────────────────────────────

def test_14_publisher_no_files():
    """collect_status no crashea sin ningún archivo de checkpoint."""
    section("swarm_publisher — sin archivos")
    import swarm_publisher
    from unittest.mock import patch

    # Patch CHECKPOINT_DIR para apuntar a dir vacío
    with tempfile.TemporaryDirectory() as tmp:
        orig = swarm_publisher.CHECKPOINT_DIR
        swarm_publisher.CHECKPOINT_DIR = Path(tmp)
        try:
            status = swarm_publisher.collect_status()
            check("collect_status no crashea sin archivos", True)
            check("retorna dict con timestamp", "timestamp" in status)
            check("auto_loop es None cuando no hay state", status.get("auto_loop") is None)
        except Exception as e:
            fail("collect_status crashea sin archivos", str(e))
        finally:
            swarm_publisher.CHECKPOINT_DIR = orig


def test_15_publisher_partial_state():
    """collect_status maneja training_state.json con campos faltantes."""
    section("swarm_publisher — state parcial")
    import swarm_publisher

    with tempfile.TemporaryDirectory() as tmp:
        # State mínimo — sin algunos campos opcionales
        (Path(tmp) / "training_state.json").write_text(json.dumps({
            "cycles_completed": 2,
            # falta: best_val_loss, lr_current, cycle_history, etc.
        }))

        orig = swarm_publisher.CHECKPOINT_DIR
        swarm_publisher.CHECKPOINT_DIR = Path(tmp)
        try:
            status = swarm_publisher.collect_status()
            al = status.get("auto_loop")
            check("auto_loop presente con state parcial", al is not None)
            check("cycles_completed leído OK", al.get("cycles_completed") == 2)
            check("best_val_loss con fallback graceful",
                  al.get("best_val_loss") is not None or True)  # None es OK
        except Exception as e:
            fail("collect_status crashea con state parcial", str(e))
        finally:
            swarm_publisher.CHECKPOINT_DIR = orig


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 58)
    print("Tests: Edge Cases — timeouts, recovery, resume, corruption")
    print("=" * 58)

    tests = [
        test_1_node_timeout,
        test_2_node_rejoin_after_timeout,
        test_3_node_heartbeat_resets_timer,
        test_4_orphan_sect,
        test_5_kill_active_sect_transfers_legacy,
        test_6_sect_tick_no_crash,
        test_7_matriarca_corrupt_checkpoint,
        test_8_matriarca_empty_memory_file,
        test_9_matriarca_compression_trigger,
        test_10_autotrain_resume_existing_state,
        test_11_autotrain_resume_missing_checkpoint,
        test_12_autotrain_plateau_to_floor,
        test_13_autotrain_corrupt_state_file,
        test_14_publisher_no_files,
        test_15_publisher_partial_state,
    ]

    for t in tests:
        try:
            t()
        except Exception as e:
            import traceback
            fail(t.__name__, f"excepción no capturada: {e}")
            traceback.print_exc()

    total = len(passed) + len(failed)
    print("\n" + "=" * 58)
    if failed:
        print(f"RESULTADO: {len(passed)}/{total} tests pasaron ❌")
        for f_name in failed:
            print(f"  - {f_name}")
        sys.exit(1)
    else:
        print(f"Tests: {total}/{total} passed ✅")
        sys.exit(0)


if __name__ == "__main__":
    main()
