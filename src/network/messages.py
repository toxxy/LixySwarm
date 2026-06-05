"""
LixySwarm Network — Formatos de Mensajes Legacy v1
===================================================
LSP v2 vive en `src.network.lsp_v2` y es el protocolo principal.
Este módulo se conserva para compatibilidad con pruebas y herramientas antiguas.
"""
import struct
import hmac
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Optional, List

import torch
import numpy as np


# Clave HMAC compartida (en producción sería por-sesión negociada)
# Para LAN local es suficiente — previene mensajes de procesos no-Lixy
SHARED_KEY = b"lixy-swarm-v1-local"

# Formato: node_id(16s) + agent_id(B) + timestamp_ms(Q) + feromon_f16(512s) + sig(8s)
# Total: 16 + 1 + 8 + 512 + 8 = 545 bytes
FEROMON_FMT = "!16sBQ512s8s"
FEROMON_SIZE = struct.calcsize(FEROMON_FMT)  # 545


@dataclass
class FeromonMessage:
    """
    Mensaje de feromona entre nodos — ultra ligero para UDP.
    float16 × 256 = 512 bytes de payload efectivo.
    """
    node_id: str          # 16 chars hex
    agent_id: int         # 0-2
    timestamp_ms: int     # unix ms
    feromon: torch.Tensor  # [256] float32 (se convierte a f16 en wire)
    valid: bool = True    # si pasó validación HMAC

    def pack(self) -> bytes:
        """Serializa a bytes para UDP."""
        node_bytes = self.node_id.encode("ascii")[:16].ljust(16, b"\x00")
        ts = self.timestamp_ms or int(time.time() * 1000)
        # float32 → float16 para ahorrar espacio
        f16 = self.feromon.half().cpu().numpy().tobytes()
        # HMAC truncado a 8 bytes
        sig = hmac.new(SHARED_KEY, f16, "sha256").digest()[:8]
        return struct.pack(FEROMON_FMT, node_bytes, self.agent_id, ts, f16, sig)

    @classmethod
    def unpack(cls, data: bytes) -> Optional["FeromonMessage"]:
        """Deserializa desde bytes UDP."""
        if len(data) < FEROMON_SIZE:
            return None
        try:
            node_bytes, agent_id, ts, f16_bytes, sig = struct.unpack(FEROMON_FMT, data[:FEROMON_SIZE])
            # Verificar HMAC
            expected_sig = hmac.new(SHARED_KEY, f16_bytes, "sha256").digest()[:8]
            valid = hmac.compare_digest(sig, expected_sig)
            # Deserializar feromona
            f16_arr = np.frombuffer(f16_bytes, dtype=np.float16).copy()
            feromon = torch.from_numpy(f16_arr).float()  # f16 → f32
            node_id = node_bytes.rstrip(b"\x00").decode("ascii", errors="replace")
            return cls(node_id=node_id, agent_id=agent_id, timestamp_ms=ts, feromon=feromon, valid=valid)
        except Exception:
            return None

    def is_fresh(self, max_age_ms: int = 5000) -> bool:
        """True si el mensaje tiene menos de max_age_ms milisegundos."""
        age = int(time.time() * 1000) - self.timestamp_ms
        return 0 <= age <= max_age_ms


@dataclass
class GossipMessage:
    """
    Mensaje de gossip para sincronización de Matriarca.
    Viaja por TCP (confiable).
    """
    kind: str          # "digest" | "request" | "memories"
    node_id: str
    timestamp: float = 0.0
    payload: dict = None  # contenido específico por tipo

    def to_bytes(self) -> bytes:
        """JSON + longitud prefix para framing TCP."""
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        msg = {
            "kind": self.kind,
            "node_id": self.node_id,
            "timestamp": self.timestamp,
            "payload": self.payload or {},
        }
        data = json.dumps(msg, ensure_ascii=False).encode("utf-8")
        # 4 bytes length prefix (big-endian)
        return struct.pack("!I", len(data)) + data

    @classmethod
    def from_bytes(cls, data: bytes) -> Optional["GossipMessage"]:
        """Deserializa desde bytes TCP (con length prefix)."""
        if len(data) < 4:
            return None
        try:
            length = struct.unpack("!I", data[:4])[0]
            if len(data) < 4 + length:
                return None
            msg = json.loads(data[4:4 + length].decode("utf-8"))
            return cls(
                kind=msg["kind"],
                node_id=msg["node_id"],
                timestamp=msg.get("timestamp", 0.0),
                payload=msg.get("payload", {}),
            )
        except Exception:
            return None

    # ─── Constructores de tipos específicos ───

    @classmethod
    def make_digest(cls, node_id: str, memory_count: int, newest_ts: float, bank_hash: str) -> "GossipMessage":
        """Digest del banco de memorias para anti-entropy."""
        return cls(kind="digest", node_id=node_id, payload={
            "memory_count": memory_count,
            "newest_ts": newest_ts,
            "bank_hash": bank_hash,
        })

    @classmethod
    def make_request(cls, node_id: str, since_ts: float) -> "GossipMessage":
        """Solicitar memorias más nuevas que since_ts."""
        return cls(kind="request", node_id=node_id, payload={"since_ts": since_ts})

    @classmethod
    def make_memories(cls, node_id: str, memories: List[dict]) -> "GossipMessage":
        """Enviar lista de memorias (embeddings como listas Python)."""
        return cls(kind="memories", node_id=node_id, payload={"memories": memories})

    @classmethod
    def make_ping(cls, node_id: str, feromon_port: int, gossip_port: int) -> "GossipMessage":
        """Ping de descubrimiento / heartbeat."""
        return cls(kind="ping", node_id=node_id, payload={
            "feromon_port": feromon_port,
            "gossip_port": gossip_port,
        })
