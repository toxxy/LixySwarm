"""
swarm_publisher.py — Publica estado del swarm local al VPS periódicamente.
Corre en la máquina local (con GPU) y escribe swarm_status.json en el VPS.

Uso:
    python3 swarm_publisher.py --vps-host <vps-ip> --vps-path /opt/lixyswarm/swarm_status.json

# Configure your VPS in .env or environment:
    (o simplemente escribe el JSON local y rsync lo sube)
"""
import os
import json
import time
import glob
import re
import subprocess
import argparse
from pathlib import Path
from datetime import datetime, timezone

BASE = Path(__file__).parent
CHECKPOINT_DIR = BASE / "checkpoints"
ANT_SPEC_FILE  = BASE / "checkpoints" / "ant_specialization.json"
SWARM_LOG_PATTERN = "/tmp/swarm_*.log"

VPS_HOST = os.environ.get("LIXYSWARM_VPS_HOST", "root@localhost")
VPS_PATH = "/opt/lixyswarm/swarm_status.json"
PUBLISH_INTERVAL = 15   # segundos entre publicaciones


def _read_json(path: Path) -> dict | None:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return None


def collect_status() -> dict:
    """Recolecta todo el estado del swarm local."""
    status = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "local",
        "agents": [],
        "matriarca": {},
        "dolphin": {},
        "metrics": {},
        "training_active": False,
    }

    # ─── Agentes ───────────────────────────────────────────────────────────────
    spec = _read_json(ANT_SPEC_FILE)
    if spec and isinstance(spec, dict):
        current = spec.get('current', {})
        label_history = spec.get('label_history', spec.get('labels', {}))
        for k, v in current.items():
            if isinstance(v, dict):
                lbl = label_history.get(k, 'refinador')
                # label_history puede ser dict {id: str} o dict {id: list}
                role = lbl[-1] if isinstance(lbl, list) else str(lbl)
                status["agents"].append({
                    "id": k,
                    "role": role,
                    "fitness": round(v.get("fitness", 0), 3),
                    "diversity": round(v.get("feromon_divergence", 0), 3),
                    "confidence": round(v.get("confidence", 0), 3),
                    "lr_factor": round(v.get("lr_factor", 1.0), 3),
                })
        status["swarm_diversity"] = round(
            sum(a["diversity"] for a in status["agents"]) / len(status["agents"]), 3
        ) if status["agents"] else None
        status["swarm_step"] = max((v.get("step", 0) for v in current.values()), default=None)

    # ─── Matriarca ─────────────────────────────────────────────────────────────
    mat_mem_file = CHECKPOINT_DIR / "matriarca_memory.json"
    if mat_mem_file.exists():
        try:
            mem_data = json.loads(mat_mem_file.read_text())
            memories = mem_data if isinstance(mem_data, list) else mem_data.get("memories", [])
            importances = [m.get("importance", 0) for m in memories if isinstance(m, dict)]
            status["matriarca"] = {
                "memory_count": len(memories),
                "avg_importance": round(sum(importances)/len(importances), 3) if importances else 0,
                "active_pct": round(sum(1 for i in importances if i > 0.2)/len(importances)*100, 1) if importances else 0,
            }
            # Legados de sectas si existe el banco
            legacy_file = CHECKPOINT_DIR / "sect_legacy_bank.json"
            if legacy_file.exists():
                try:
                    legacy_data = json.loads(legacy_file.read_text())
                    records = legacy_data if isinstance(legacy_data, list) else legacy_data.get("records", [])
                    status["matriarca"]["legacy_count"] = len(records)
                    if records:
                        roles = {}
                        for r in records:
                            role = r.get("role_type", "?") if isinstance(r, dict) else "?"
                            roles[role] = roles.get(role, 0) + 1
                        status["matriarca"]["legacy_by_role"] = roles
                except Exception:
                    pass
        except Exception:
            pass
    # Fallback: último log de training tiene el conteo
    if not status["matriarca"]:
        log_files = sorted(glob.glob(SWARM_LOG_PATTERN), key=lambda f: Path(f).stat().st_mtime if Path(f).exists() else 0, reverse=True)
        for lf in log_files[:2]:
            try:
                lines = Path(lf).read_text().splitlines()[-300:]
                for line in reversed(lines):
                    m = re.search(r"memorias[:\s]+(\d+)", line, re.I)
                    if m:
                        status["matriarca"]["memory_count"] = int(m.group(1))
                        break
            except Exception:
                pass

    # ─── Métricas de training ──────────────────────────────────────────────────
    log_files = sorted(glob.glob(SWARM_LOG_PATTERN), key=lambda f: Path(f).stat().st_mtime if Path(f).exists() else 0, reverse=True)
    for lf in log_files[:3]:
        try:
            lf_path = Path(lf)
            # Si el log fue modificado hace menos de 5 min → training activo
            age_s = time.time() - lf_path.stat().st_mtime
            if age_s < 300:
                status["training_active"] = True

            lines = lf_path.read_text().splitlines()[-100:]
            for line in reversed(lines):
                if "tok/s" in line and status["metrics"].get("toks_per_sec") is None:
                    m = re.search(r"([\d,]+)\s*tok/s", line)
                    if m:
                        status["metrics"]["toks_per_sec"] = int(m.group(1).replace(",", ""))
                if "val_loss" in line and status["metrics"].get("val_loss") is None:
                    m = re.search(r"val_loss[:\s=]+([\d.]+)", line)
                    if m:
                        status["metrics"]["val_loss"] = float(m.group(1))
                if "step" in line and status["metrics"].get("step") is None:
                    m = re.search(r"step\s+(\d+)\s*\|", line)
                    if m:
                        status["metrics"]["step"] = int(m.group(1))
                if all(k in status["metrics"] for k in ["toks_per_sec", "val_loss", "step"]):
                    break
        except Exception:
            pass

    # ─── Checkpoint meta ───────────────────────────────────────────────────────
    for name in ["swarm_best.pt", "swarm_latest.pt"]:
        pt = CHECKPOINT_DIR / name
        if pt.exists():
            try:
                import torch
                ckpt = torch.load(pt, map_location="cpu", weights_only=False)
                status["checkpoint"] = {
                    "name": name,
                    "step": ckpt.get("step"),
                    "val_loss": ckpt.get("val_loss"),
                    "params_M": round(sum(t.numel() for t in ckpt["model"].values()) / 1e6, 1),
                }
                if status["metrics"].get("val_loss") is None:
                    status["metrics"]["val_loss"] = ckpt.get("val_loss")
                if status["metrics"].get("step") is None:
                    status["metrics"]["step"] = ckpt.get("step")
                break
            except Exception:
                pass

    # ─── Delfín (pool dinámico) ─────────────────────────────────────────────
    # Obtener n_dolphins real del pool si está disponible
    status["dolphin"] = {
        "phase": "A",
        "active_pings": 5,
        "ping_names": ["topic", "intent", "need", "context", "emotion"],
        "n_dolphins": 1,  # se actualiza si hay runtime activo
        "dolphin_pool": True,
    }

    # ─── LSP ────────────────────────────────────────────────────────────────────
    # Leer identidad LSP local (node_id + puertos)
    import socket as _socket
    local_node_id = None
    hostname = _socket.gethostname()
    lsp_identity_file = CHECKPOINT_DIR / f"lsp_identity_{hostname}.pem"
    # Fallback: buscar cualquier lsp_identity_*.pem
    if not lsp_identity_file.exists():
        candidates = sorted(CHECKPOINT_DIR.glob("lsp_identity_*.pem"))
        if candidates:
            lsp_identity_file = candidates[0]
    if lsp_identity_file.exists():
        try:
            from src.network.lsp import LSPIdentity
            ident = LSPIdentity.load(str(lsp_identity_file))
            if ident:
                local_node_id = ident.node_id_hex
        except Exception:
            pass

    # Puertos LSP v2 del nodo local
    local_feromon_port = int(os.environ.get("LIXYSWARM_FEROMON_PORT", "7337"))
    local_gossip_port  = int(os.environ.get("LIXYSWARM_GOSSIP_PORT", "7338"))

    status["lsp"] = {
        "protocol": "LSP v2",
        "wire_format": "LYSW",
        "identity": "Ed25519",
        "identity_persistent": lsp_identity_file.exists(),
        "status": "active",
        "float16": True,
        "merge_on_transit": True,
        "internet": {
            "ready": bool(os.environ.get("LIXYSWARM_PUBLIC_HOST") or os.environ.get("LIXYSWARM_VPS_HOST")),
            "mode": "relay/public-host" if (os.environ.get("LIXYSWARM_PUBLIC_HOST") or os.environ.get("LIXYSWARM_VPS_HOST")) else "lan-only",
        },
    }

    # ─── Peers / nodo local ──────────────────────────────────────────────────
    # Informar al VPS sobre el nodo local para que aparezca en la UI
    if local_node_id:
        # Detectar IP de la máquina local (preferir la que ve el VPS)
        local_host = os.environ.get("LIXYSWARM_LOCAL_HOST", "")
        if not local_host:
            try:
                import socket as _socket
                s = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
                s.settimeout(0.1)
                # Conecta al VPS para descubrir IP de salida
                vps_for_discovery = os.environ.get("LIXYSWARM_VPS_HOST", "8.8.8.8")
                s.connect((vps_for_discovery.split("@")[-1], 7338))
                local_host = s.getsockname()[0]
                s.close()
            except Exception:
                local_host = "127.0.0.1"
        status["peers"] = [{
            "id": local_node_id,
            "host": local_host,
            "feromon_port": local_feromon_port,
            "gossip_port": local_gossip_port,
            "role": "local-gpu",
            "last_seen": datetime.now(timezone.utc).isoformat(),
        }]
    elif status.get("peers") is None:
        status["peers"] = []

    # ─── Auto-training loop ──────────────────────────────────────────────────────
    train_state_file = CHECKPOINT_DIR / "training_state.json"
    auto_state = _read_json(train_state_file)
    if auto_state:
        history = auto_state.get("cycle_history", [])
        status["auto_loop"] = {
            "cycles_completed":  auto_state.get("cycles_completed", 0),
            "total_steps":       auto_state.get("total_steps", 0),
            "best_val_loss":     auto_state.get("best_val_loss"),
            "lr_current":        auto_state.get("lr_current"),
            "plateau_count":     auto_state.get("plateau_count", 0),
            "last_checkpoint":   auto_state.get("last_checkpoint"),
            # Últimos 5 ciclos para graficar tendencia
            "recent_history":    history[-5:] if history else [],
            "last_hunger":       auto_state.get("last_hunger"),
            "last_updated":      auto_state.get("last_updated"),
        }
    else:
        status["auto_loop"] = None

    return status


def publish(status: dict, vps_host: str, vps_path: str):
    """Escribe el JSON en local y lo sube al VPS vía scp."""
    local_tmp = Path("/tmp/swarm_status_publish.json")
    local_tmp.write_text(json.dumps(status, indent=2))

    result = subprocess.run(
        ["sshpass", "-p", "YY6m1XInz..+", "scp",
         "-o", "StrictHostKeyChecking=no",
         str(local_tmp), f"{vps_host}:{vps_path}"],
        capture_output=True, text=True, timeout=15
    )
    return result.returncode == 0


def run(vps_host: str, vps_path: str, interval: int, once: bool = False):
    print(f"🐜 SwarmPublisher arrancando (→ {vps_host}:{vps_path} cada {interval}s)")
    while True:
        try:
            status = collect_status()
            ok = publish(status, vps_host, vps_path)
            ts = status["timestamp"]
            step = status["metrics"].get("step", "?")
            mem = status.get("matriarca", {}).get("memory_count", "?")
            active = "🟢 training" if status["training_active"] else "💤 idle"
            print(f"  [{ts[:19]}] {active} | step={step} | memorias={mem} | upload={'✅' if ok else '❌'}")
        except Exception as e:
            print(f"  ⚠ Error: {e}")

        if once:
            break
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vps-host", default=VPS_HOST)
    parser.add_argument("--vps-path", default=VPS_PATH)
    parser.add_argument("--interval", type=int, default=PUBLISH_INTERVAL)
    parser.add_argument("--once", action="store_true", help="Publicar una vez y salir")
    args = parser.parse_args()
    run(args.vps_host, args.vps_path, args.interval, args.once)
