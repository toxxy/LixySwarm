"""
auto_train.py — Loop de Auto-Entrenamiento Continuo del LixySwarm
=================================================================
El enjambre aprende solo. Sin intervención manual.

Ciclos infinitos de N pasos (chunk_steps) con:
- Evaluación automática cada ciclo
- Ajuste de LR si val_loss se estanca (plateau detection)
- Checkpoints automáticos por ciclo y por best
- Estado persistido en training_state.json → resume limpiamente
- SIGTERM/SIGINT: termina el ciclo actual y sale limpiamente
- Publicación de feromonas a red P2P si --network

Uso:
    # Arrancar (o resumir desde donde quedó):
    python3 auto_train.py

    # Con red P2P activada:
    python3 auto_train.py --network

    # Chunk más corto para validar:
    python3 auto_train.py --chunk 200

    # Correr solo N ciclos (para CI/test):
    python3 auto_train.py --cycles 3 --chunk 100

    # Ver estado actual sin entrenar:
    python3 auto_train.py --status

Estado en checkpoints/training_state.json:
    {
        "total_steps": 53920,
        "best_val_loss": 3.4376,
        "cycles_completed": 12,
        "lr_current": 1e-4,
        "plateau_count": 0,
        "last_checkpoint": "swarm_auto_cycle12.pt",
        "started_at": 1234567890,
        "last_updated": 1234567990
    }
"""

import sys
import os
import json
import time
import signal
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class AutoTrainConfig:
    # Steps por ciclo (chunk)
    chunk_steps: int = 1000

    # Ciclos máximos (0 = infinito)
    max_cycles: int = 0

    # Checkpoint dir
    checkpoint_dir: str = "checkpoints"

    # Dataset
    mixed: bool = True           # 90% FineWeb + 10% personal por defecto
    fw_ratio: float = 0.9
    spanish: bool = False
    triple: bool = False

    # Hyperparams base
    batch_size: int = 4
    grad_accum: int = 4
    lr_initial: float = 1e-4
    lr_min: float = 5e-6         # piso absoluto de LR
    lr_decay_factor: float = 0.7 # multiplicador cuando hay plateau
    plateau_patience: int = 3    # ciclos sin mejora antes de bajar LR
    warmup_steps: int = 30       # warmup dentro de cada ciclo
    eval_steps: int = 20         # batches para estimar val_loss
    eval_interval: int = 50      # cada cuántos steps evaluar dentro del ciclo
    block_size: int = 512

    # Red P2P
    network: bool = False        # publicar feromonas via SwarmNetwork

    # Keeper checkpoints (últimos N ciclos)
    keep_last_n: int = 3


STATE_FILE = "checkpoints/training_state.json"

# ─── Estado persistido ────────────────────────────────────────────────────────

def load_state(path: str) -> dict:
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {
        "total_steps": 0,
        "best_val_loss": float("inf"),
        "cycles_completed": 0,
        "lr_current": None,       # None = usar lr_initial de config
        "plateau_count": 0,
        "last_checkpoint": None,
        "cycle_history": [],      # [(cycle, steps, val_loss, lr)]
        "started_at": time.time(),
        "last_updated": time.time(),
    }

def save_state(state: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

# ─── Plateau detection ────────────────────────────────────────────────────────

def check_plateau(state: dict, new_val_loss: float, cfg: AutoTrainConfig) -> tuple[bool, float]:
    """
    Retorna (plateau_detected, lr_recomendado).
    Mejora: reset plateau count y mantener LR.
    Plateau: incrementar contador; si supera patience → bajar LR.
    """
    improved = new_val_loss < state["best_val_loss"] - 1e-4  # umbral mínimo

    if improved:
        state["best_val_loss"] = new_val_loss
        state["plateau_count"] = 0
        lr = state["lr_current"]
    else:
        state["plateau_count"] += 1
        lr = state["lr_current"]
        if state["plateau_count"] >= cfg.plateau_patience:
            new_lr = max(lr * cfg.lr_decay_factor, cfg.lr_min)
            if new_lr < lr:
                print(f"  📉 Plateau detectado ({state['plateau_count']} ciclos) — LR: {lr:.2e} → {new_lr:.2e}")
                lr = new_lr
                state["plateau_count"] = 0  # reset después de bajar LR
            else:
                print(f"  ⚠️  Plateau — LR ya en mínimo ({cfg.lr_min:.2e}), continuando...")

    state["lr_current"] = lr
    return not improved, lr

# ─── Red P2P (opcional) ───────────────────────────────────────────────────────

def maybe_start_network(cfg: AutoTrainConfig):
    """Arranca SwarmNetwork si --network. Retorna (net, None) o (None, None)."""
    if not cfg.network:
        return None
    try:
        from src.network.swarm_network import SwarmNetwork
        net = SwarmNetwork.create(swarm=None, mode="lan", protocol="v1")
        net.start()
        print(f"  🌐 Red P2P activa — modo LAN")
        return net
    except Exception as e:
        print(f"  ⚠️  Red P2P no disponible: {e}")
        return None

def maybe_broadcast_feromons(net, swarm):
    """Publica feromonas del swarm a la red si está activa."""
    if net is None or swarm is None:
        return
    try:
        import torch
        combined = sum(
            swarm.feromon_bank[i].detach().cpu()
            for i in range(len(swarm.agents))
        ) / len(swarm.agents)
        net.broadcast_feromon(combined)
    except Exception as e:
        pass  # no fatal

# ─── Cleanup de checkpoints viejos ───────────────────────────────────────────

def cleanup_old_checkpoints(checkpoint_dir: str, keep_n: int, state: dict):
    """Mantiene solo los últimos keep_n checkpoints de ciclo."""
    history = state.get("cycle_history", [])
    ckpt_dir = Path(checkpoint_dir)
    # Obtener lista de archivos auto_cycle_*.pt ordenados por ciclo
    cycle_files = sorted(
        ckpt_dir.glob("swarm_auto_cycle*.pt"),
        key=lambda p: int(p.stem.replace("swarm_auto_cycle", "") or 0)
    )
    to_delete = cycle_files[:-keep_n] if len(cycle_files) > keep_n else []
    for p in to_delete:
        try:
            p.unlink()
            print(f"  🗑️  Checkpoint antiguo eliminado: {p.name}")
        except Exception:
            pass

# ─── Un ciclo de training ─────────────────────────────────────────────────────

def run_cycle(cycle_n: int, state: dict, cfg: AutoTrainConfig, net=None) -> Optional[float]:
    """
    Ejecuta un chunk de training. Retorna val_loss o None si falló.
    """
    import subprocess

    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Determinar checkpoint de entrada
    latest_ckpt = checkpoint_dir / "swarm_latest.pt"
    best_ckpt = checkpoint_dir / "swarm_best.pt"

    # Preferir latest, luego best
    input_ckpt = None
    if latest_ckpt.exists():
        input_ckpt = str(latest_ckpt)
    elif best_ckpt.exists():
        input_ckpt = str(best_ckpt)

    lr = state["lr_current"] or cfg.lr_initial
    min_lr = max(lr * 0.1, cfg.lr_min)

    print(f"\n{'='*60}")
    print(f"🔄 Ciclo {cycle_n} | steps={cfg.chunk_steps} | lr={lr:.2e} | total_steps_prev={state['total_steps']}")
    if input_ckpt:
        print(f"   → Continuando desde: {Path(input_ckpt).name}")
    else:
        print(f"   → Arrancando desde cero (sin checkpoint previo)")
    print(f"{'='*60}")

    # Construir comando para train_swarm.py
    cmd = [
        sys.executable, "train_swarm.py",
        "--steps", str(cfg.chunk_steps),
        "--batch", str(cfg.batch_size),
        "--lr", str(lr),
        "--min-lr", str(min_lr),
        "--warmup", str(cfg.warmup_steps),
        "--grad-accum", str(cfg.grad_accum),
        "--eval-steps", str(cfg.eval_steps),
        "--eval-interval", str(cfg.eval_interval),
        "--block-size", str(cfg.block_size),
    ]
    if input_ckpt:
        cmd += ["--checkpoint", input_ckpt]
    if cfg.mixed:
        cmd += ["--mixed"]
    if cfg.spanish:
        cmd += ["--spanish"]
    if cfg.triple:
        cmd += ["--triple"]

    # Log path para que Matriarca aprenda del ciclo
    log_path = str(checkpoint_dir / f"cycle_{cycle_n}_train.log")
    cmd += ["--log-path", log_path]

    start_t = time.time()
    print(f"⚙️  Ejecutando: {' '.join(cmd[:6])} ...")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).parent),
            capture_output=False,  # stdout directo a terminal
            timeout=7200,  # 2h máximo por ciclo
        )
        elapsed = time.time() - start_t
        if result.returncode != 0:
            print(f"\n❌ Ciclo {cycle_n} falló (returncode={result.returncode})")
            return None
    except subprocess.TimeoutExpired:
        print(f"\n⏰ Ciclo {cycle_n} timeout (>2h)")
        return None
    except KeyboardInterrupt:
        print(f"\n🛑 Ciclo {cycle_n} interrumpido por usuario")
        return None

    elapsed = time.time() - start_t

    # Leer val_loss del checkpoint guardado
    val_loss = _read_val_loss_from_checkpoint(checkpoint_dir)

    if val_loss is None:
        print(f"\n⚠️  No se pudo leer val_loss del checkpoint")
        return None

    # Copiar latest a cycle checkpoint
    cycle_ckpt = checkpoint_dir / f"swarm_auto_cycle{cycle_n}.pt"
    if latest_ckpt.exists():
        import shutil
        shutil.copy2(latest_ckpt, cycle_ckpt)

    # Broadcast feromonas si hay red
    # Nota: el swarm ya no está en memoria (fue subprocess), solo publicamos
    # el checkpoint como señal de vida (feromonas reales en el próximo ciclo integrado)

    print(f"\n✅ Ciclo {cycle_n} completado")
    print(f"   val_loss: {val_loss:.4f} | best: {state['best_val_loss']:.4f}")
    print(f"   elapsed: {elapsed/60:.1f}min | total_steps: {state['total_steps'] + cfg.chunk_steps}")

    return val_loss


def _read_val_loss_from_checkpoint(checkpoint_dir: Path) -> Optional[float]:
    """Lee val_loss del último checkpoint guardado."""
    import torch

    # Intentar swarm_latest.pt primero, luego swarm_best.pt
    for name in ["swarm_latest.pt", "swarm_best.pt", "swarm_final.pt"]:
        path = checkpoint_dir / name
        if path.exists():
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                vl = ckpt.get("val_loss")
                if vl is not None:
                    return float(vl)
            except Exception:
                continue
    return None


# ─── Main loop ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Auto-training loop del LixySwarm 🐜")
    parser.add_argument("--chunk", type=int, default=1000, help="Steps por ciclo")
    parser.add_argument("--cycles", type=int, default=0, help="Ciclos máximos (0=infinito)")
    parser.add_argument("--lr", type=float, default=1e-4, help="LR inicial")
    parser.add_argument("--min-lr", type=float, default=5e-6, help="LR mínimo absoluto")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--mixed", action="store_true", default=True,
                        help="90%% FineWeb + 10%% personal (default ON)")
    parser.add_argument("--no-mixed", action="store_true",
                        help="Desactivar modo mixto (solo personal)")
    parser.add_argument("--spanish", action="store_true")
    parser.add_argument("--triple", action="store_true")
    parser.add_argument("--network", action="store_true", help="Activar red P2P")
    parser.add_argument("--plateau-patience", type=int, default=3)
    parser.add_argument("--keep-last", type=int, default=3,
                        help="Mantener últimos N checkpoints de ciclo")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--state-file", type=str, default=STATE_FILE)
    parser.add_argument("--status", action="store_true", help="Mostrar estado y salir")
    args = parser.parse_args()

    # ─── Status ───
    if args.status:
        state = load_state(args.state_file)
        print("\n📊 Estado del auto-training:")
        print(f"   Ciclos completados:  {state['cycles_completed']}")
        print(f"   Steps totales:       {state['total_steps']}")
        print(f"   Mejor val_loss:      {state['best_val_loss']:.4f}")
        print(f"   LR actual:           {state.get('lr_current', 'N/A')}")
        print(f"   Plateau count:       {state['plateau_count']}")
        print(f"   Último checkpoint:   {state.get('last_checkpoint', 'N/A')}")
        if state.get("cycle_history"):
            print(f"\n   Últimos ciclos:")
            for c in state["cycle_history"][-5:]:
                print(f"     Ciclo {c[0]:3d}: steps={c[1]} val_loss={c[2]:.4f} lr={c[3]:.2e}")
        return

    # ─── Config ───
    cfg = AutoTrainConfig(
        chunk_steps=args.chunk,
        max_cycles=args.cycles,
        checkpoint_dir=args.checkpoint_dir,
        mixed=(args.mixed and not args.no_mixed),
        spanish=args.spanish,
        triple=args.triple,
        batch_size=args.batch,
        grad_accum=args.grad_accum,
        lr_initial=args.lr,
        lr_min=args.min_lr,
        plateau_patience=args.plateau_patience,
        keep_last_n=args.keep_last,
        network=args.network,
    )

    # ─── Estado ───
    state = load_state(args.state_file)
    if state["lr_current"] is None:
        state["lr_current"] = cfg.lr_initial

    # ─── Señales ───
    _stop_after_cycle = [False]

    def _handle_signal(sig, frame):
        print(f"\n🛑 Señal {sig} recibida — terminando al final del ciclo actual...")
        _stop_after_cycle[0] = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # ─── Red P2P ───
    net = maybe_start_network(cfg)

    # ─── Banner ───
    print("\n" + "🐜" * 20)
    print("  LixySwarm — Auto-Training Loop")
    print("🐜" * 20)
    print(f"  Ciclos completados previos: {state['cycles_completed']}")
    print(f"  Steps totales previos:      {state['total_steps']}")
    print(f"  Mejor val_loss previo:      {state['best_val_loss']:.4f}")
    print(f"  LR inicial:                 {state['lr_current']:.2e}")
    print(f"  Chunk: {cfg.chunk_steps} steps/ciclo | mixed={cfg.mixed}")
    print(f"  {'∞ ciclos' if cfg.max_cycles == 0 else f'{cfg.max_cycles} ciclos máximos'}")
    print()

    # ─── Loop principal ───
    cycle_n = state["cycles_completed"] + 1

    while True:
        if _stop_after_cycle[0]:
            print("\n🛑 Stop solicitado — saliendo limpiamente")
            break

        if cfg.max_cycles > 0 and cycle_n > cfg.max_cycles:
            print(f"\n✅ {cfg.max_cycles} ciclos completados — auto-training terminado")
            break

        # Ejecutar ciclo
        val_loss = run_cycle(cycle_n, state, cfg, net)

        if val_loss is None:
            print(f"⚠️  Ciclo {cycle_n} sin val_loss válido — reintentando en el siguiente ciclo")
            # No actualizar state para que el próximo ciclo use el mismo checkpoint
            time.sleep(5)
            cycle_n += 1
            continue

        # Actualizar estado
        state["total_steps"] += cfg.chunk_steps
        state["cycles_completed"] = cycle_n
        state["last_checkpoint"] = f"swarm_auto_cycle{cycle_n}.pt"
        state["last_updated"] = time.time()

        # Detectar plateau y ajustar LR
        is_plateau, new_lr = check_plateau(state, val_loss, cfg)
        state["lr_current"] = new_lr

        # Historial
        state["cycle_history"].append([cycle_n, cfg.chunk_steps, val_loss, new_lr])
        # Mantener solo últimos 50 para no crecer infinito
        if len(state["cycle_history"]) > 50:
            state["cycle_history"] = state["cycle_history"][-50:]

        # Guardar estado
        save_state(state, args.state_file)

        # Limpiar checkpoints viejos
        cleanup_old_checkpoints(cfg.checkpoint_dir, cfg.keep_last_n, state)

        print(f"\n💾 Estado guardado — ciclo {cycle_n} completado")
        print(f"   val_loss: {val_loss:.4f} | best: {state['best_val_loss']:.4f} | plateau: {state['plateau_count']}/{cfg.plateau_patience}")

        # Pausa breve entre ciclos (deja respirar GPU/memoria)
        if not _stop_after_cycle[0]:
            time.sleep(3)

        cycle_n += 1

    # ─── Cleanup ───
    if net:
        net.stop()

    print(f"\n📊 Resumen final:")
    print(f"   Ciclos completados: {state['cycles_completed']}")
    print(f"   Steps totales:      {state['total_steps']}")
    print(f"   Mejor val_loss:     {state['best_val_loss']:.4f}")
    save_state(state, args.state_file)
    print(f"   Estado guardado en: {args.state_file}")


if __name__ == "__main__":
    main()
