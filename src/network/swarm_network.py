"""
LixySwarm Network — SwarmNetwork (LSP v2)
==========================================
P2P sin configuración. Como Bitcoin:
  - peers.json: cache persistente de peers (sobrevive reinicios)
  - Bootstrap seeds: nodos relay iniciales
  - Peer exchange: al conectar, intercambiar listas de peers
  - Auto-bootstrap: se conecta solo al arrancar

Zero-config: arranca y se integra a la red automáticamente.
"""
import time
import logging
import threading
import json
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .node import NodeIdentity, Peer, PeerTable
from .bootstrap import PeersDB, bootstrap_network, encode_peer_list

log = logging.getLogger("lixy.network")


@dataclass
class NetworkStats:
    mode: str = "local"
    peers_known: int = 0
    feromons_sent: int = 0
    feromons_received: int = 0
    global_deltas_sent: int = 0
    global_deltas_received: int = 0
    global_memories_received: int = 0
    gossip_rounds: int = 0
    started_at: float = 0.0

    def summary(self) -> str:
        uptime = time.time() - self.started_at if self.started_at else 0
        return (
            f"mode={self.mode} | peers={self.peers_known} | "
            f"feromon_rx={self.feromons_received} | feromon_tx={self.feromons_sent} | "
            f"gossip={self.gossip_rounds} | global_rx={self.global_memories_received} | "
            f"uptime={uptime:.0f}s"
        )


class SwarmNetwork:
    """
    LSP v2 — zero-config P2P.
    7337 UDP (feromonas float16) + 7338 TCP (handshake + peer exchange).

    Auto-bootstrap: peers.json → seeds → peer exchange. Sin flags.
    """

    @classmethod
    def create(cls, swarm=None, mode="auto", feromon_port=7337, gossip_port=7338,
               checkpoint_dir="checkpoints"):
        if swarm is not None:
            identity = NodeIdentity.from_swarm(swarm, feromon_port=feromon_port, gossip_port=gossip_port)
        else:
            identity = NodeIdentity.generate_anonymous(
                feromon_port=feromon_port,
                gossip_port=gossip_port,
            )
        return cls(identity=identity, mode=mode, checkpoint_dir=checkpoint_dir, swarm=swarm)

    def __init__(self, identity, mode="auto", checkpoint_dir="checkpoints", swarm=None):
        self.identity = identity
        self.mode = mode
        self.checkpoint_dir = Path(checkpoint_dir)
        self.swarm = swarm
        self.peers = PeerTable(self_id=identity.node_id)
        self.stats = NetworkStats(mode=mode)
        self.peers_db = PeersDB(str(self.checkpoint_dir / "peers.json"))
        self._lsp_v2_node = None
        self._running = False
        self._bootstrap_thread = None
        self._stop_event = threading.Event()
        self._global_sync_lock = threading.RLock()
        self._on_peer_connected = None
        self._on_peer_lost = None
        self.matriarca_dual = None

    def on_peer_connected(self, fn):
        self._on_peer_connected = fn
        return fn

    def on_peer_lost(self, fn):
        self._on_peer_lost = fn
        return fn

    def attach_global_matriarca(self, matriarca_dual):
        """Adjunta una MatriarcaDual para sincronizar su memoria global."""
        self.matriarca_dual = matriarca_dual
        return matriarca_dual

    # ─── WAN / Relay ──────────────────────────────────────────────────────────

    def connect_peer(self, host: str, gossip_port: int = 7338):
        """Handshake LSP v2 TCP + guardar en peers_db."""
        if self._lsp_v2_node is None:
            log.warning(f"Cannot connect to {host}:{gossip_port} — LSP v2 not started")
            return False
        try:
            self._lsp_v2_node.connect_peer(host, gossip_port)
            self.peers_db.mark_connected(host, gossip_port)
            self.stats.peers_known += 1
            log.info(f"🔗 Peer conectado: {host}:{gossip_port}")
            return True
        except Exception as e:
            self.peers_db.mark_failed(host, gossip_port)
            log.debug(f"connect_peer {host}:{gossip_port}: {e}")
            return False

    def _bootstrap_loop(self):
        """Auto-bootstrap: intenta peers guardados, seeds, luego peer exchange."""
        if self._stop_event.wait(2):  # dejar que LSP v2 termine de arrancar
            return
        if not self._running:
            return
        n = bootstrap_network(self, self.peers_db)
        if n > 0:
            log.info(f"Bootstrap: connected to {n} peers")
            self._exchange_peers()
        else:
            log.info("Bootstrap: no peers yet (listening for incoming connections)")

        # Bootstrap periódico: cada 5 min reintentar si tenemos pocos peers
        while self._running:
            if self._stop_event.wait(300):
                break
            if self.peer_count < 3 and self._lsp_v2_node:
                n = bootstrap_network(self, self.peers_db, max_bootstrap=4)
                if n > 0:
                    log.info(f"Periodic bootstrap: +{n} peers")
                    self._exchange_peers()

    def _exchange_peers(self):
        """Intercambia listas de peers con todos los conectados."""
        if self._lsp_v2_node is None:
            return
        my_peers = [
            {"host": p["host"], "gossip_port": p.get("gossip_port", 7338)}
            for p in self._lsp_v2_node.peers()
        ]
        if my_peers:
            self._lsp_v2_node.send_peer_list(my_peers)

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    def start(self):
        """Arranca LSP v2 (UDP 7337 + TCP 7338) + auto-bootstrap."""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self.stats.started_at = time.time()

        if self.mode == "local":
            log.info(f"SwarmNetwork [local] — {self.identity}")
            return

        try:
            from src.network.lsp_v2 import LSPNodeV2

            lsp_identity = self._load_or_create_lsp_identity()
            self._lsp_v2_node = LSPNodeV2(
                lsp_identity,
                feromon_port=self.identity.feromon_port,
                gossip_port=self.identity.gossip_port,
            )
            self._lsp_v2_node.start()

            @self._lsp_v2_node.on_feromon_received
            def _on_v2_feromon(feromon, node_id_hex):
                self.stats.feromons_received += 1
                self.peers.update_feromon(node_id_hex, feromon)

            @self._lsp_v2_node.on_peer_connected
            def _on_v2_peer(node_id_hex, host, port):
                peer_info = next(
                    (p for p in self._lsp_v2_node.peers() if p.get("node_id") == node_id_hex),
                    {},
                )
                feromon_port = peer_info.get("feromon_port", port)
                gossip_port = peer_info.get("gossip_port", port)
                self.peers_db.add_peer(host, gossip_port)
                peer = Peer(
                    node_id=node_id_hex,
                    host=host,
                    feromon_port=feromon_port,
                    gossip_port=gossip_port,
                )
                self.peers.add(peer)
                self.stats.peers_known += 1
                if self._on_peer_connected:
                    self._on_peer_connected(peer)

            # Peer exchange: recibir peers de otros nodos y guardarlos
            @self._lsp_v2_node.on_peer_list_received
            def _on_peer_list(peer_addrs):
                self.peers_db.add_peers_batch(peer_addrs)
                log.debug(f"Peer exchange: received {len(peer_addrs)} addrs")

            @self._lsp_v2_node.on_gossip_delta_received
            def _on_gossip_delta(delta, node_id_hex):
                self.stats.gossip_rounds += 1
                self.stats.global_deltas_received += 1
                if self.matriarca_dual is None:
                    return
                try:
                    with self._global_sync_lock:
                        merged = self.matriarca_dual.merge_global_delta(delta, source_id=node_id_hex)
                    self.stats.global_memories_received += merged
                except Exception as e:
                    log.debug(f"global matriarca merge error: {e}")

            # Auto-bootstrap en thread separado (zero-config)
            self._bootstrap_thread = threading.Thread(
                target=self._bootstrap_loop, daemon=True, name="bootstrap")
            self._bootstrap_thread.start()

            self.stats.mode = "lan"
            log.info(f"SwarmNetwork [LSP v2] — {self.identity} "
                     f"UDP:{self.identity.feromon_port} TCP:{self.identity.gossip_port} "
                     f"peers_db:{self.peers_db.count}")

        except Exception as e:
            log.warning(f"LSP v2 no pudo arrancar: {e}")
            if self.mode == "auto":
                self.stats.mode = "local"
            else:
                raise

    def stop(self):
        self._running = False
        self._stop_event.set()
        if self._lsp_v2_node:
            self._lsp_v2_node.stop()
            self._lsp_v2_node = None
        if self._bootstrap_thread and self._bootstrap_thread.is_alive():
            self._bootstrap_thread.join(timeout=2.0)
        self._bootstrap_thread = None
        log.info("SwarmNetwork detenida")

    def _load_or_create_lsp_identity(self):
        from src.network.lsp import LSPIdentity
        import uuid as _uuid
        machine_id = _uuid.getnode()  # MAC address como int — único por máquina
        identity_path = self.checkpoint_dir / f"lsp_identity_{machine_id:012x}.pem"

        # Buscar identidad de ESTA máquina (MAC exacto)
        identity = LSPIdentity.load(str(identity_path))
        if identity is not None:
            return identity

        # No existe para esta máquina — generar nueva
        identity = LSPIdentity.generate()
        identity.save(str(identity_path))
        log.info(f"New LSP identity (machine={machine_id:012x}): {identity.node_id_hex[:16]}...")
        return identity

    # ─── API principal ────────────────────────────────────────────────────────

    def broadcast_feromon(self, feromon: torch.Tensor, agent_id: int = 0, fitness: float = 0.5):
        """Envía feromona float16 a todos los peers vía LSP v2."""
        if self._lsp_v2_node is None:
            return
        try:
            self._lsp_v2_node.send_feromon_v2(feromon.detach().cpu(), fitness=fitness, step=0)
            self.stats.feromons_sent += max(1, len(self.peers.alive_peers()))
        except Exception as e:
            log.debug(f"broadcast_feromon v2 error: {e}")

    def broadcast_global_delta(self, max_items: int = 64, min_importance: float = 0.0) -> int:
        """Publica memorias globales compartibles de la MatriarcaDual adjunta."""
        if self._lsp_v2_node is None or self.matriarca_dual is None:
            return 0
        try:
            with self._global_sync_lock:
                delta = self.matriarca_dual.export_global_delta(
                    max_items=max_items,
                    min_importance=min_importance,
                )
            count = int(delta.get("count", 0))
            if count <= 0:
                return 0
            delta["source_node_id"] = self.identity.node_id
            delta["feromon_port"] = self.identity.feromon_port
            delta["gossip_port"] = self.identity.gossip_port
            self._lsp_v2_node.send_gossip_delta(delta)
            self.stats.gossip_rounds += 1
            self.stats.global_deltas_sent += 1
            return count
        except Exception as e:
            log.debug(f"broadcast_global_delta error: {e}")
            return 0

    def collect_feromons(self) -> List[torch.Tensor]:
        if not self.peers.count:
            return []
        return self.peers.collect_feromons()

    def get_combined_feromon(self, local_feromon: torch.Tensor, remote_weight: float = 0.3) -> torch.Tensor:
        remote = self.collect_feromons()
        if not remote:
            return local_feromon
        remote_mean = torch.stack(remote).mean(dim=0).to(local_feromon.device)
        if remote_mean.shape != local_feromon.shape:
            remote_mean = F.interpolate(
                remote_mean.unsqueeze(0).unsqueeze(0),
                size=local_feromon.shape[-1],
                mode="linear", align_corners=False,
            ).squeeze()
        combined = (1 - remote_weight) * local_feromon + remote_weight * remote_mean
        local_norm = local_feromon.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return F.normalize(combined, dim=-1) * local_norm

    def merge_remote_feromons(self, local_feromon: torch.Tensor, remote_weight: float = 0.3) -> torch.Tensor:
        return self.get_combined_feromon(local_feromon, remote_weight)

    @property
    def is_distributed(self) -> bool:
        return self.peers.count > 0

    @property
    def peer_count(self) -> int:
        return self.peers.count

    def status(self) -> dict:
        return {
            "node_id": self.identity.node_id,
            "host": self.identity.host,
            "mode": self.stats.mode,
            "peers": self.peers.count,
            "peers_list": [p.to_dict() for p in self.peers.alive_peers()],
            "stats": self.stats.summary(),
            "global_sync": {
                "attached": self.matriarca_dual is not None,
                "deltas_sent": self.stats.global_deltas_sent,
                "deltas_received": self.stats.global_deltas_received,
                "memories_received": self.stats.global_memories_received,
            },
        }
