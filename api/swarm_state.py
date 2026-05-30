"""
swarm_state.py — Lector de estado del LixySwarm para la API.

Arquitectura:
  - Nodo local (GPU)  → swarm_publisher.py → sube swarm_status.json al VPS cada 15s
  - VPS (API)         → lee swarm_status.json → expone endpoints al frontend

Sin cargar modelos. Sin torch en el VPS para las métricas del swarm.
"""

import json
import re
import glob
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

BASE           = Path(__file__).parent.parent
STATUS_FILE    = BASE / "swarm_status.json"     # publicado por swarm_publisher.py
NODE_LOG       = BASE / "node.log"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_status() -> dict:
    """Lee swarm_status.json publicado por el nodo local."""
    try:
        if STATUS_FILE.exists():
            age_s = time.time() - STATUS_FILE.stat().st_mtime
            data = json.loads(STATUS_FILE.read_text())
            data["_age_seconds"] = round(age_s, 1)
            data["_stale"] = age_s > 60   # más de 1 min → stale
            return data
    except Exception:
        pass
    return {}


def get_agents_state() -> list[dict]:
    s = _read_status()
    return s.get("agents", [])


def get_matriarca_state() -> dict:
    s = _read_status()
    mat = s.get("matriarca", {})
    stale = s.get("_stale", True)
    return {
        "memory_count":   mat.get("memory_count", 0),
        "avg_importance": mat.get("avg_importance", 0.0),
        "active_pct":     mat.get("active_pct", None),
        "diversity":      mat.get("diversity", None),
        "data_fresh":     not stale,
    }


def get_dolphin_state() -> dict:
    s = _read_status()
    dol = s.get("dolphin", {})
    return {
        "phase":           dol.get("phase", "A"),
        "active_pings":    dol.get("active_pings", 5),
        "ping_names":      dol.get("ping_names", ["topic","intent","need","context","emotion"]),
        "sleep_state_norm": dol.get("sleep_state_norm", None),
        "confidence":      dol.get("confidence", None),
    }


def get_network_state() -> dict:
    """Lee nodos P2P del node.log del VPS."""
    nodes = []
    connected = 0
    if NODE_LOG.exists():
        try:
            lines = NODE_LOG.read_text().splitlines()[-200:]
            # Buscar líneas de heartbeat con peers
            for line in reversed(lines):
                m = re.search(r"peers=(\d+)", line)
                if m:
                    connected = int(m.group(1))
                    break
            # Node ID propio
            for line in lines:
                m = re.search(r"Node\(([0-9a-f]+)@([\d.]+):(\d+)\)", line)
                if m:
                    nodes.append({
                        "node_id": m.group(1),
                        "host":    m.group(2),
                        "port":    int(m.group(3)),
                        "role":    "vps-relay",
                        "self":    True,
                    })
                    break
        except Exception:
            pass

    # Agrega nodos del status publicado
    s = _read_status()
    if s.get("peers"):
        for p in s["peers"]:
            nodes.append({"node_id": p.get("id","?"), "host": p.get("host","?"), "role": "local", "self": False})
            connected += 1

    return {
        "nodes":           nodes,
        "connected_count": connected,
        "vps_node_id":     nodes[0]["node_id"] if nodes else None,
    }


def get_metrics() -> dict:
    s = _read_status()
    m = s.get("metrics", {})
    ckpt = s.get("checkpoint", {})
    return {
        "toks_per_sec":    m.get("toks_per_sec"),
        "val_loss":        m.get("val_loss") or ckpt.get("val_loss"),
        "step":            m.get("step") or ckpt.get("step"),
        "training_active": s.get("training_active", False),
        "checkpoint":      ckpt.get("name"),
        "params_M":        ckpt.get("params_M"),
        "data_age_s":      s.get("_age_seconds"),
    }


def get_full_swarm_status() -> dict:
    s = _read_status()
    agents    = get_agents_state()
    matriarca = get_matriarca_state()
    dolphin   = get_dolphin_state()
    network   = get_network_state()
    metrics   = get_metrics()
    return {
        "agents":          agents,
        "agent_count":     len(agents),
        "swarm_diversity": s.get("swarm_diversity"),
        "matriarca":       matriarca,
        "dolphin":         dolphin,
        "network":         network,
        "metrics":         metrics,
        "data_fresh":      not s.get("_stale", True),
        "last_update":     s.get("timestamp"),
    }
