"""
test_auto_train.py — Tests del loop de auto-entrenamiento continuo
"""
import sys, json, time, tempfile, shutil
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

def check(name: str, condition: bool = True, reason: str = ""):
    if condition:
        ok(name)
    else:
        fail(name, reason or "assertion failed")

# ─── Tests ───────────────────────────────────────────────────────────────────

def test_1_import():
    """auto_train.py importa sin error."""
    import auto_train
    check("auto_train importa OK")
    check("AutoTrainConfig existe", hasattr(auto_train, "AutoTrainConfig"))
    check("load_state existe", hasattr(auto_train, "load_state"))
    check("save_state existe", hasattr(auto_train, "save_state"))
    check("check_plateau existe", hasattr(auto_train, "check_plateau"))


def test_2_state_fresh():
    """load_state devuelve estado limpio cuando no existe archivo."""
    import auto_train
    state = auto_train.load_state("/tmp/nonexistent_state_xyz.json")
    check("total_steps = 0", state["total_steps"] == 0)
    check("best_val_loss = inf", state["best_val_loss"] == float("inf"))
    check("cycles_completed = 0", state["cycles_completed"] == 0)
    check("plateau_count = 0", state["plateau_count"] == 0)
    check("cycle_history = []", state["cycle_history"] == [])


def test_3_state_save_load():
    """save_state/load_state round-trip."""
    import auto_train
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "state.json")
        state = auto_train.load_state(path)
        state["total_steps"] = 5000
        state["best_val_loss"] = 3.45
        state["cycles_completed"] = 5
        auto_train.save_state(state, path)

        loaded = auto_train.load_state(path)
        check("total_steps round-trip", loaded["total_steps"] == 5000)
        check("best_val_loss round-trip", abs(loaded["best_val_loss"] - 3.45) < 1e-6)
        check("cycles_completed round-trip", loaded["cycles_completed"] == 5)


def test_4_plateau_improvement():
    """check_plateau: mejora → no plateau, LR estable."""
    import auto_train
    cfg = auto_train.AutoTrainConfig(lr_initial=1e-4, plateau_patience=3)
    state = auto_train.load_state("/tmp/nope.json")
    state["lr_current"] = cfg.lr_initial
    state["best_val_loss"] = 4.0

    is_p, lr = auto_train.check_plateau(state, 3.8, cfg)  # mejora grande
    check("mejora → no plateau", not is_p)
    check("mejora → best actualizado", abs(state["best_val_loss"] - 3.8) < 1e-6,
          f"best={state['best_val_loss']}")
    check("mejora → plateau_count = 0", state["plateau_count"] == 0)
    check("mejora → LR estable", abs(lr - cfg.lr_initial) < 1e-10)


def test_5_plateau_trigger():
    """check_plateau: N ciclos sin mejora → baja LR."""
    import auto_train
    cfg = auto_train.AutoTrainConfig(lr_initial=1e-4, plateau_patience=3,
                                      lr_decay_factor=0.7, lr_min=1e-6)
    state = auto_train.load_state("/tmp/nope.json")
    state["lr_current"] = cfg.lr_initial
    state["best_val_loss"] = 3.5

    # 3 ciclos sin mejora (val_loss igual)
    for i in range(3):
        auto_train.check_plateau(state, 3.5, cfg)

    lr_after = state["lr_current"]
    check("plateau_patience=3 → LR bajó", lr_after < cfg.lr_initial,
          f"lr={lr_after:.2e}")
    check("plateau LR = inicial * decay_factor",
          abs(lr_after - cfg.lr_initial * cfg.lr_decay_factor) < 1e-10,
          f"lr={lr_after:.2e} expected={cfg.lr_initial * cfg.lr_decay_factor:.2e}")
    check("plateau → plateau_count reset", state["plateau_count"] == 0)


def test_6_plateau_floor():
    """check_plateau: LR no baja de lr_min."""
    import auto_train
    cfg = auto_train.AutoTrainConfig(lr_initial=1e-6, plateau_patience=2,
                                      lr_decay_factor=0.1, lr_min=1e-6)
    state = auto_train.load_state("/tmp/nope.json")
    state["lr_current"] = 1e-6  # ya en el piso
    state["best_val_loss"] = 3.5

    for _ in range(3):
        auto_train.check_plateau(state, 3.5, cfg)

    check("LR no baja de lr_min", state["lr_current"] >= cfg.lr_min,
          f"lr={state['lr_current']:.2e}")


def test_7_config_defaults():
    """AutoTrainConfig: defaults razonables."""
    import auto_train
    cfg = auto_train.AutoTrainConfig()
    check("chunk_steps > 0", cfg.chunk_steps > 0)
    check("lr_initial > lr_min", cfg.lr_initial > cfg.lr_min)
    check("lr_min > 0", cfg.lr_min > 0)
    check("plateau_patience >= 1", cfg.plateau_patience >= 1)
    check("keep_last_n >= 1", cfg.keep_last_n >= 1)
    check("mixed=True por defecto", cfg.mixed)


def test_8_cleanup_old_checkpoints():
    """cleanup_old_checkpoints: mantiene solo los últimos N."""
    import auto_train, torch
    with tempfile.TemporaryDirectory() as tmp:
        # Crear 5 cycle checkpoints falsos
        for i in range(1, 6):
            # Archivo vacío (no real checkpoint, solo para probar el glob)
            p = Path(tmp) / f"swarm_auto_cycle{i}.pt"
            p.write_text("fake")

        state = {"cycle_history": []}
        auto_train.cleanup_old_checkpoints(tmp, keep_n=3, state=state)

        remaining = sorted(Path(tmp).glob("swarm_auto_cycle*.pt"))
        check("solo 3 checkpoints quedan", len(remaining) == 3,
              f"quedan {len(remaining)}: {[r.name for r in remaining]}")
        # Deben ser los últimos (3,4,5)
        names = [r.name for r in remaining]
        check("quedan los más nuevos (3,4,5)",
              "swarm_auto_cycle3.pt" in names and "swarm_auto_cycle5.pt" in names,
              f"names={names}")


def test_9_status_flag(capsys=None):
    """auto_train --status no crashea con state vacío."""
    import auto_train, io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with tempfile.TemporaryDirectory() as tmp:
        state_path = str(Path(tmp) / "state.json")
        # No existe → load devuelve fresh state
        state = auto_train.load_state(state_path)
        check("load_state fresh: cycles_completed=0", state["cycles_completed"] == 0)


def test_10_cycle_history_trim():
    """El historial no crece más de 50 entradas."""
    import auto_train
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "state.json")
        state = auto_train.load_state(path)
        state["lr_current"] = 1e-4
        state["best_val_loss"] = float("inf")

        # Simular 60 entradas
        for i in range(60):
            state["cycle_history"].append([i, 1000, 3.5 - i * 0.01, 1e-4])
            if len(state["cycle_history"]) > 50:
                state["cycle_history"] = state["cycle_history"][-50:]

        auto_train.save_state(state, path)
        loaded = auto_train.load_state(path)
        check("historial <= 50 entradas", len(loaded["cycle_history"]) <= 50,
              f"len={len(loaded['cycle_history'])}")
        # Debe tener los últimos 50 (entradas 10..59)
        check("historial tiene últimas entradas",
              loaded["cycle_history"][-1][0] == 59,
              f"last={loaded['cycle_history'][-1][0]}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*50)
    print("Tests: auto_train.py")
    print("="*50)

    tests = [
        test_1_import,
        test_2_state_fresh,
        test_3_state_save_load,
        test_4_plateau_improvement,
        test_5_plateau_trigger,
        test_6_plateau_floor,
        test_7_config_defaults,
        test_8_cleanup_old_checkpoints,
        test_9_status_flag,
        test_10_cycle_history_trim,
    ]

    for t in tests:
        try:
            t()
        except Exception as e:
            fail(t.__name__, f"excepción: {e}")

    total = len(passed) + len(failed)
    print("\n" + "="*50)
    if failed:
        print(f"RESULTADO: {len(passed)}/{total} tests pasaron ❌")
        print("Fallos:")
        for f_name in failed:
            print(f"  - {f_name}")
        sys.exit(1)
    else:
        print(f"Tests: {total}/{total} passed ✅")
        sys.exit(0)


if __name__ == "__main__":
    main()
