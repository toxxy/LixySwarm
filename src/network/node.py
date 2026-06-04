"""
LixySwarm Network — Node Identity & Peer Table
===============================================
Node ID derivado de IdentityVec del agente — reutiliza arquitectura existente.
"""
import hashlib
import time
import socket
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from pathlib import Path

import torch


@dataclass
class NodeIdentity:
    """
    Identidad del nodo en la red P2P.
    El node_id se deriva del IdentityVec del primer agente del enjambre.
    Así cada instancia de LixySwarm tiene una identidad única y reproducible.
    """
    node_id: str           # 16-char hex (64-bit SHA256 del IdentityVec)
    host: str              # IP local
    feromon_port: int = 7337   # UDP — feromonas float16
    gossip_port: int = 7338    # TCP — handshake + gossip

    @classmethod
    def from_swarm(cls, swarm, host: str = None, feromon_port: int = 7337, gossip_port: int = 7338) -> "NodeIdentity":
        """Genera identidad desde el IdentityVec del primer AgentBase."""
        # Usar identity_vec del primer agente
        identity_vec = swarm.agents[0].identity_vec
        identity_bytes = identity_vec.cpu().numpy().tobytes()
        node_id = hashlib.sha256(identity_bytes).hexdigest()[:16]

        # Detectar IP local automáticamente
        if host is None:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                host = s.getsockname()[0]
                s.close()
            except Exception:
                host = "127.0.0.1"

        return cls(node_id=node_id, host=host, feromon_port=feromon_port, gossip_port=gossip_port)

    @classmethod
    def generate_anonymous(cls, host: str = "127.0.0.1") -> "NodeIdentity":
        """Genera una identidad aleatoria para testing."""
        import os
        node_id = hashlib.sha256(os.urandom(32)).hexdigest()[:16]
        return cls(node_id=node_id, host=host)

    def __repr__(self):
        return f"Node({self.node_id[:8]}@{self.host}:{self.feromon_port})"


@dataclass
class Peer:
    """Un nodo vecino conocido."""
    node_id: str
    host: str
    feromon_port: int = 7337
    gossip_port: int = 7338
    last_seen: float = field(default_factory=time.time)
    is_alive: bool = True
    # Última feromona recibida de este peer
    latest_feromon: Optional[torch.Tensor] = field(default=None, repr=False)
    # Contadores
    feromons_received: int = 0
    gossip_rounds: int = 0

    @property
    def feromon_addr(self):
        return (self.host, self.feromon_port)

    @property
    def gossip_addr(self):
        return (self.host, self.gossip_port)

    def mark_seen(self):
        self.last_seen = time.time()
        self.is_alive = True

    def age_seconds(self) -> float:
        return time.time() - self.last_seen

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "host": self.host,
            "feromon_port": self.feromon_port,
            "gossip_port": self.gossip_port,
            "last_seen": self.last_seen,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Peer":
        return cls(**{k: v for k, v in d.items() if k in ("node_id", "host", "feromon_port", "gossip_port", "last_seen")})


class PeerTable:
    """
    Tabla de peers conocidos con timeout automático.
    Thread-safe para acceso concurrente.
    """
    PEER_TIMEOUT_S = 120   # peer muerto si no se ve en 2 min
    MAX_PEERS = 50

    def __init__(self, self_id: str):
        self.self_id = self_id
        self._peers: Dict[str, Peer] = {}
        import threading
        self._lock = threading.Lock()

    def add(self, peer: Peer) -> bool:
        """Agrega o actualiza un peer. Retorna True si es nuevo."""
        if peer.node_id == self.self_id:
            return False  # no agregar a sí mismo
        with self._lock:
            is_new = peer.node_id not in self._peers
            if is_new and len(self._peers) >= self.MAX_PEERS:
                # Remover el más viejo
                oldest = min(self._peers.values(), key=lambda p: p.last_seen)
                del self._peers[oldest.node_id]
            self._peers[peer.node_id] = peer
            return is_new

    def get(self, node_id: str) -> Optional[Peer]:
        with self._lock:
            return self._peers.get(node_id)

    def remove(self, node_id: str):
        with self._lock:
            self._peers.pop(node_id, None)

    def alive_peers(self) -> List[Peer]:
        """Peers activos (vistos recientemente)."""
        now = time.time()
        with self._lock:
            alive = [p for p in self._peers.values() if now - p.last_seen < self.PEER_TIMEOUT_S]
        return alive

    def mark_dead(self):
        """Marca como muertos los peers con timeout."""
        now = time.time()
        with self._lock:
            for p in self._peers.values():
                if now - p.last_seen > self.PEER_TIMEOUT_S:
                    p.is_alive = False

    @property
    def count(self) -> int:
        return len(self.alive_peers())

    def update_feromon(self, node_id: str, feromon: torch.Tensor):
        """Actualiza la última feromona recibida de un peer."""
        with self._lock:
            if node_id in self._peers:
                self._peers[node_id].latest_feromon = feromon.cpu()
                self._peers[node_id].feromons_received += 1
                self._peers[node_id].mark_seen()

    def collect_feromons(self) -> List[torch.Tensor]:
        """Recolecta las últimas feromonas de todos los peers activos."""
        with self._lock:
            return [
                p.latest_feromon for p in self._peers.values()
                if p.latest_feromon is not None and p.is_alive
            ]
