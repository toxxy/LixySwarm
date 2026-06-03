"""
swarm_state.py — Lector de estado del LixySwarm para la API.

Arquitectura:
  - Nodo local (GPU)  → swarm_publisher.py → sube swarm_status.json al VPS cada 15s
  - VPS (API)         → lee swarm_status.json → expone endpoints al frontend

Sin cargar modelos. Sin torch en el VPS para las métricas del swarm.
"""

import json
import os
import re
import glob
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

BASE           = Path(__file__).parent.parent
STATUS_FILE    = BASE / "swarm_status.json"     # publicado por swarm_publisher.py
NODE_LOG       = BASE / "node.log"
TRAINING_STATE = BASE / "checkpoints" / "training_state.json"
LSP_IDENTITIES = [
    BASE / "checkpoints" / "lsp_identity.pem",
    BASE / ".lixyswarm" / "identity.key",
]


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


def _read_json(path: Path) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text())
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

    lsp = get_lsp_state()
    return {
        "nodes":           nodes,
        "connected_count": connected,
        "vps_node_id":     nodes[0]["node_id"] if nodes else None,
        "protocol":        lsp["protocol"],
        "lsp":             lsp,
        "internet":        lsp["internet"],
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


def get_auto_loop_state() -> dict | None:
    """Lee el estado del loop de auto-training, publicado o local."""
    s = _read_status()
    if isinstance(s.get("auto_loop"), dict):
        return s["auto_loop"]

    auto_state = _read_json(TRAINING_STATE)
    if not auto_state:
        return None

    history = auto_state.get("cycle_history", [])
    return {
        "cycles_completed": auto_state.get("cycles_completed", 0),
        "total_steps": auto_state.get("total_steps", 0),
        "best_val_loss": auto_state.get("best_val_loss"),
        "lr_current": auto_state.get("lr_current"),
        "plateau_count": auto_state.get("plateau_count", 0),
        "last_checkpoint": auto_state.get("last_checkpoint"),
        "recent_history": history[-5:] if history else [],
        "last_hunger": auto_state.get("last_hunger"),
        "last_updated": auto_state.get("last_updated"),
    }


def get_lsp_state() -> dict:
    """Describe capacidades de red reales sin prometer WAN automático."""
    s = _read_status()
    published = s.get("lsp", {}) if isinstance(s.get("lsp"), dict) else {}
    public_host = os.environ.get("LIXYSWARM_PUBLIC_HOST")
    relay_host = os.environ.get("LIXYSWARM_VPS_HOST")
    wan_ready = bool(public_host or relay_host)

    return {
        "protocol": published.get("protocol", "LSP v2"),
        "wire_format": published.get("wire_format", "LYSW"),
        "identity": published.get("identity", "Ed25519"),
        "identity_persistent": any(path.exists() for path in LSP_IDENTITIES),
        "status": published.get("status", "available"),
        "float16": published.get("float16", True),
        "merge_on_transit": published.get("merge_on_transit", True),
        "discovery": "mDNS LAN automático",
        "ports": {
            "v1_feromon_udp": 4444,
            "v1_gossip_tcp": 4445,
            "v2_feromon_udp": 4454,
            "v2_gossip_tcp": 4455,
            "standalone_feromon_udp": 7337,
            "standalone_gossip_tcp": 7338,
        },
        "internet": {
            "ready": wan_ready,
            "mode": "relay/public-host" if wan_ready else "lan-only",
            "requires": [] if wan_ready else [
                "VPS relay o IP pública",
                "puertos abiertos/reenviados",
                "configuración explícita de peers",
            ],
        },
    }


def get_full_swarm_status() -> dict:
    s = _read_status()
    agents    = get_agents_state()
    matriarca = get_matriarca_state()
    dolphin   = get_dolphin_state()
    network   = get_network_state()
    metrics   = get_metrics()
    auto_loop = get_auto_loop_state()
    lsp       = get_lsp_state()
    return {
        "agents":          agents,
        "agent_count":     len(agents),
        "swarm_diversity": s.get("swarm_diversity"),
        "matriarca":       matriarca,
        "dolphin":         dolphin,
        "network":         network,
        "lsp":             lsp,
        "metrics":         metrics,
        "auto_loop":       auto_loop,
        "data_fresh":      not s.get("_stale", True),
        "last_update":     s.get("timestamp"),
    }
