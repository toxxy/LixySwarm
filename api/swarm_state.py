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
    BASE / ".lixyswarm" / "identity.key",
] + sorted((BASE / "checkpoints").glob("lsp_identity_*.pem"))


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
    """Lee nodos P2P del node.log del VPS + status publicado por el nodo local."""
    nodes = []
    known_peer_count = 0
    vps_node_id = None

    # 1. Identidad del VPS relay (self) desde node.log
    if NODE_LOG.exists():
        try:
            lines = NODE_LOG.read_text().splitlines()[-300:]
            # Buscar líneas de heartbeat con peers
            for line in reversed(lines):
                m = re.search(r"peers=(\d+)", line)
                if m:
                    known_peer_count = max(known_peer_count, int(m.group(1)))
                    break
            # Buscar Node ID del VPS (formato: "Node ID: <hex>")
            for line in lines:
                m = re.search(r"Node ID:\s*([0-9a-f]+)", line)
                if m:
                    vps_node_id = m.group(1)
                    break
        except Exception:
            pass

    # VPS relay como nodo self (si tiene identidad)
    standalone_identity = BASE / ".lixyswarm" / "identity.key"
    if vps_node_id or standalone_identity.exists():
        if not vps_node_id and standalone_identity.exists():
            try:
                from src.network.lsp import LSPIdentity
                ident = LSPIdentity.load(str(standalone_identity))
                if ident:
                    vps_node_id = ident.node_id_hex[:16]
            except Exception:
                pass
        if vps_node_id:
            nodes.append({
                "node_id": vps_node_id[:16],
                "host":    "0.0.0.0",
                "port":    7338,
                "role":    "vps-relay",
                "self":    True,
            })

    # 2. Nodo local desde el status publicado (swarm_publisher.py sube swarm_status.json)
    s = _read_status()
    published_peers = s.get("peers", [])
    status_stale = s.get("_stale", True)
    if published_peers:
        for p in published_peers:
            p_node_id = p.get("id", "?")
            # Evitar duplicados
            if not any(n["node_id"] == p_node_id for n in nodes):
                nodes.append({
                    "node_id": p_node_id[:16],
                    "host":    p.get("host", "?"),
                    "feromon_port": p.get("feromon_port"),
                    "gossip_port": p.get("gossip_port"),
                    "role":    p.get("role", "local-gpu"),
                    "self":    False,
                    "active":  not status_stale,
                    "stale":   status_stale,
                    "age_s":   s.get("_age_seconds"),
                })

    lsp = get_lsp_state()
    connected = sum(1 for n in nodes if n.get("self") or n.get("active"))
    return {
        "nodes":           nodes,
        "connected_count": connected,
        "known_peer_count": known_peer_count,
        "vps_node_id":     vps_node_id[:16] if vps_node_id else None,
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
    published_internet = published.get("internet", {}) if isinstance(published.get("internet"), dict) else {}
    standalone_identity = BASE / ".lixyswarm" / "identity.key"
    is_vps_relay = standalone_identity.exists()
    wan_ready = bool(public_host or relay_host or published_internet.get("ready") or is_vps_relay)
    wan_mode = (
        "vps-relay"
        if is_vps_relay
        else published_internet.get("mode")
        or ("relay/public-host" if wan_ready else "lan-only")
    )

    return {
        "protocol": published.get("protocol", "LSP v2"),
        "wire_format": published.get("wire_format", "LYSW"),
        "identity": published.get("identity", "Ed25519"),
        "identity_persistent": any(path.exists() for path in LSP_IDENTITIES),
        "status": published.get("status", "available"),
        "float16": published.get("float16", True),
        "merge_on_transit": published.get("merge_on_transit", True),
        "discovery": "TCP handshake + swarm_status.json",
        "ports": {
            "feromon_udp": 7337,
            "gossip_tcp": 7338,
        },
        "internet": {
            "ready": wan_ready,
            "mode": wan_mode,
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
