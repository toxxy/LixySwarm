"""
swarm_state.py — Lector de estado del LixySwarm (solo lectura, sin cargar el modelo)
Lee JSON/logs del disco para exponer estado al SwarmExplorer.
"""

import json
import re
import glob
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

BASE = Path(__file__).parent.parent
CHECKPOINT_DIR = BASE / "checkpoints"
ANT_SPEC_FILE = BASE / "ant_specialization.json"
SWARM_LOG_PATTERN = "/tmp/swarm_*.log"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict | None:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return None


def _latest_checkpoint_meta() -> dict:
    """Read step/val_loss from latest checkpoint without loading the full model."""
    meta: dict[str, Any] = {"step": None, "val_loss": None, "checkpoint": None}
    # Prefer swarm_best.pt meta sidecar, then scan directory
    for name in ["swarm_best_meta.json", "swarm_latest_meta.json"]:
        p = CHECKPOINT_DIR / name
        d = _read_json(p)
        if d:
            meta.update(d)
            meta["checkpoint"] = name
            return meta
    # Try to find any .pt and read its meta json companion
    for pt in sorted(CHECKPOINT_DIR.glob("*.pt"), key=lambda f: f.stat().st_mtime, reverse=True):
        meta_path = pt.with_suffix(".json")
        d = _read_json(meta_path)
        if d:
            meta.update(d)
            meta["checkpoint"] = pt.name
            return meta
    # Last resort: read checkpoint via torch (just metadata keys)
    try:
        import torch
        candidates = sorted(CHECKPOINT_DIR.glob("*.pt"), key=lambda f: f.stat().st_mtime, reverse=True)
        if candidates:
            ckpt = torch.load(candidates[0], map_location="cpu", weights_only=False)
            meta["step"] = ckpt.get("step") or ckpt.get("iter") or ckpt.get("best_val_loss_step")
            meta["val_loss"] = ckpt.get("best_val_loss") or ckpt.get("val_loss")
            meta["checkpoint"] = candidates[0].name
    except Exception:
        pass
    return meta


def get_agents_state() -> list[dict]:
    """Read agent specialization from ant_specialization.json."""
    data = _read_json(ANT_SPEC_FILE)
    if not data:
        return []
    agents = []
    # Support both list and dict formats
    if isinstance(data, list):
        for item in data:
            agents.append({
                "id": item.get("id", item.get("agent_id", "?")),
                "role": item.get("role", "unknown"),
                "fitness": item.get("fitness", 0.0),
                "diversity": item.get("diversity", 0.0),
                "active": item.get("active", True),
            })
    elif isinstance(data, dict):
        for agent_id, info in data.items():
            if isinstance(info, dict):
                agents.append({
                    "id": agent_id,
                    "role": info.get("role", "unknown"),
                    "fitness": info.get("fitness", 0.0),
                    "diversity": info.get("diversity", 0.0),
                    "active": info.get("active", True),
                })
    return agents


def get_matriarca_state() -> dict:
    """Read Matriarca memory stats without loading the full model."""
    # Try JSON sidecar first
    for name in ["matriarca_memory.json", "matriarca_state.json"]:
        d = _read_json(CHECKPOINT_DIR / name)
        if d:
            memories = d.get("memories", d.get("memory_bank", []))
            importances = [m.get("importance", 0.0) for m in memories if isinstance(m, dict)]
            avg_importance = sum(importances) / len(importances) if importances else 0.0
            return {
                "memory_count": len(memories),
                "avg_importance": round(avg_importance, 4),
                "diversity": d.get("diversity", None),
                "last_updated": d.get("last_updated", None),
            }
    # Try loading checkpoint for matriarca keys
    try:
        import torch
        candidates = sorted(CHECKPOINT_DIR.glob("*.pt"), key=lambda f: f.stat().st_mtime, reverse=True)
        if candidates:
            ckpt = torch.load(candidates[0], map_location="cpu", weights_only=False)
            mat = ckpt.get("matriarca") or ckpt.get("matriarca_state") or {}
            mem = mat.get("memory_bank", mat.get("memories", []))
            importances = [m["importance"] for m in mem if isinstance(m, dict) and "importance" in m]
            avg = sum(importances) / len(importances) if importances else 0.0
            return {
                "memory_count": len(mem),
                "avg_importance": round(avg, 4),
                "diversity": None,
                "last_updated": None,
            }
    except Exception:
        pass
    return {"memory_count": 0, "avg_importance": 0.0, "diversity": None, "last_updated": None}


def get_dolphin_state() -> dict:
    """Read dolphin state from checkpoint or sidecar JSON."""
    # Try sidecar
    d = _read_json(CHECKPOINT_DIR / "dolphin_state.json")
    if d:
        return {
            "active_pings": d.get("active_pings", 0),
            "sleep_state_norm": d.get("sleep_state_norm", 0.0),
            "confidence": d.get("confidence", None),
            "hemisphere": d.get("hemisphere", None),
        }
    # Try checkpoint
    try:
        import torch
        candidates = sorted(CHECKPOINT_DIR.glob("*.pt"), key=lambda f: f.stat().st_mtime, reverse=True)
        if candidates:
            ckpt = torch.load(candidates[0], map_location="cpu", weights_only=False)
            dol = ckpt.get("dolphin") or ckpt.get("dolphin_state") or {}
            sleep = dol.get("sleep_state")
            norm = float(sleep.norm()) if sleep is not None else 0.0
            return {
                "active_pings": dol.get("active_pings", 0),
                "sleep_state_norm": round(norm, 6),
                "confidence": dol.get("confidence", None),
                "hemisphere": dol.get("hemisphere", None),
            }
    except Exception:
        pass
    return {"active_pings": 0, "sleep_state_norm": 0.0, "confidence": None, "hemisphere": None}


def get_network_state() -> dict:
    """Read P2P node state from node.log or network logs."""
    nodes = []
    # Try node.log in project root or /tmp
    for log_path in [BASE / "node.log", Path("/tmp/node.log"), BASE / "p2p_nodes.json"]:
        if log_path.suffix == ".json":
            d = _read_json(log_path)
            if d:
                return {"nodes": d.get("nodes", []), "connected_count": len(d.get("nodes", []))}
        elif log_path.exists():
            try:
                lines = log_path.read_text().splitlines()[-100:]
                for line in lines:
                    # Look for patterns like "Connected to peer: <addr>"
                    m = re.search(r"(?:peer|node|connected)[:\s]+([0-9a-zA-Z.:/_-]+)", line, re.I)
                    if m:
                        addr = m.group(1).strip()
                        if addr not in nodes:
                            nodes.append(addr)
            except Exception:
                pass
            if nodes:
                return {"nodes": [{"addr": a} for a in nodes], "connected_count": len(nodes)}
    return {"nodes": [], "connected_count": 0}


def get_metrics() -> dict:
    """Read tok/s and training metrics from /tmp/swarm_*.log and checkpoint meta."""
    toks_per_sec = None
    val_loss = None
    step = None

    # Read tok/s from swarm logs
    log_files = sorted(glob.glob(SWARM_LOG_PATTERN), key=lambda f: Path(f).stat().st_mtime if Path(f).exists() else 0, reverse=True)
    for log_file in log_files[:3]:
        try:
            lines = Path(log_file).read_text().splitlines()[-200:]
            for line in reversed(lines):
                m = re.search(r"(\d+(?:\.\d+)?)\s*tok[/s]", line, re.I)
                if m:
                    toks_per_sec = float(m.group(1))
                    break
                m2 = re.search(r"val_loss[:\s=]+([0-9.]+)", line, re.I)
                if m2 and val_loss is None:
                    val_loss = float(m2.group(1))
                m3 = re.search(r"step[:\s=]+(\d+)", line, re.I)
                if m3 and step is None:
                    step = int(m3.group(1))
        except Exception:
            pass
        if toks_per_sec:
            break

    # Supplement from checkpoint meta
    if val_loss is None or step is None:
        meta = _latest_checkpoint_meta()
        val_loss = val_loss or meta.get("val_loss")
        step = step or meta.get("step")

    # Also try nohup log
    if toks_per_sec is None:
        for nohup in sorted(BASE.glob("nohup*.out"), key=lambda f: f.stat().st_mtime, reverse=True)[:2]:
            try:
                lines = nohup.read_text().splitlines()[-100:]
                for line in reversed(lines):
                    m = re.search(r"(\d+(?:\.\d+)?)\s*tok[/s]", line, re.I)
                    if m:
                        toks_per_sec = float(m.group(1))
                        break
            except Exception:
                pass
            if toks_per_sec:
                break

    return {
        "toks_per_sec": toks_per_sec,
        "val_loss": val_loss,
        "step": step,
    }


def get_full_swarm_status() -> dict:
    agents = get_agents_state()
    matriarca = get_matriarca_state()
    dolphin = get_dolphin_state()
    network = get_network_state()
    metrics = get_metrics()
    return {
        "agents": agents,
        "agent_count": len(agents),
        "matriarca": matriarca,
        "dolphin": dolphin,
        "network": network,
        "metrics": metrics,
    }
