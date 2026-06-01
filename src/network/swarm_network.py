"""
LixySwarm Network — SwarmNetwork
==================================
Abstracción principal: hace transparente single-node vs multi-node.
El LixySwarm no sabe si está solo o distribuido.

Modos:
- "local":   sin red, feromonas locales únicamente
- "lan":     descubrimiento mDNS automático en red local
- "auto":    intenta LAN, cae back a local si no hay peers

En modo local: todo funciona perfectamente, zero overhead.
Cuando aparece un peer (mDNS): automáticamente incorpora sus feromonas.
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
from .messages import FeromonMessage, GossipMessage
from .transport import FeromonUDP, GossipTCP, MDNSDiscovery

log = logging.getLogger("lixy.network")


@dataclass
class NetworkStats:
    mode: str = "local"
    peers_known: int = 0
    feromons_sent: int = 0
    feromons_received: int = 0
    gossip_rounds: int = 0
    started_at: float = 0.0

    def summary(self) -> str:
        uptime = time.time() - self.started_at if self.started_at else 0
        return (
            f"mode={self.mode} | peers={self.peers_known} | "
            f"feromon_rx={self.feromons_received} | feromon_tx={self.feromons_sent} | "
            f"gossip={self.gossip_rounds} | uptime={uptime:.0f}s"
        )


class SwarmNetwork:
    """
    Capa de red del enjambre.

    Uso:
        net = SwarmNetwork.create(swarm, mode="auto")
        net.start()

        # En forward del enjambre:
        net.broadcast_feromon(feromon_tensor, agent_id=0)
        remote_feromons = net.collect_feromons()  # vacío si no hay peers

        # Gossip de Matriarca (automático en background):
        # cada 30s sincroniza con peers si están disponibles
    """

    # ─── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        swarm=None,
        mode: str = "auto",
        feromon_port: int = 4444,
        gossip_port: int = 4445,
        checkpoint_dir: str = "checkpoints",
    ) -> "SwarmNetwork":
        """
        Crea una SwarmNetwork.
        Si swarm es None, usa una identidad anónima para testing.
        """
        if swarm is not None:
            identity = NodeIdentity.from_swarm(swarm, feromon_port=feromon_port, gossip_port=gossip_port)
        else:
            identity = NodeIdentity.generate_anonymous()
        return cls(identity=identity, mode=mode, checkpoint_dir=checkpoint_dir, swarm=swarm)

    # ─── Init ─────────────────────────────────────────────────────────────────

    def __init__(
        self,
        identity: NodeIdentity,
        mode: str = "auto",
        checkpoint_dir: str = "checkpoints",
        swarm=None,
        protocol: str = "v1",   # "v1" | "v2" — protocolo de feromonas (v2 es opt-in)
    ):
        self.identity = identity
        self.mode = mode  # "local" | "lan" | "auto"
        self.protocol = protocol  # "v1" | "v2"
        self.checkpoint_dir = Path(checkpoint_dir)
        self.swarm = swarm  # referencia al LixySwarm (para gossip de Matriarca)

        self.peers = PeerTable(self_id=identity.node_id)
        self.stats = NetworkStats(mode=mode)

        # Componentes de transporte (None en modo local)
        self._feromon_udp: Optional[FeromonUDP] = None
        self._gossip_tcp: Optional[GossipTCP] = None
        self._mdns: Optional[MDNSDiscovery] = None
        self._gossip_thread: Optional[threading.Thread] = None
        self._running = False
        # LSP v2 node (opcional, activado si protocol="v2")
        self._lsp_v2_node = None

        # Callbacks registrables
        self._on_peer_connected = None   # callback(peer) cuando llega un peer nuevo
        self._on_peer_lost = None        # callback(peer) cuando se va un peer

    def on_peer_connected(self, fn):
        """Registra callback para nuevos peers."""
        self._on_peer_connected = fn
        return fn

    def on_peer_lost(self, fn):
        self._on_peer_lost = fn
        return fn

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    def start(self):
        """Arranca la red. En modo local es casi un no-op."""
        if self._running:
            return
        self._running = True
        self.stats.started_at = time.time()

        if self.mode == "local":
            log.info(f"SwarmNetwork [local] — {self.identity}")
            return

        # Modo LAN o auto: arrancar transporte
        try:
            self._feromon_udp = FeromonUDP(self.identity, self._on_feromon_received)
            self._feromon_udp.start()

            self._gossip_tcp = GossipTCP(self.identity, self._on_gossip_received)
            self._gossip_tcp.start()

            self._mdns = MDNSDiscovery(self.identity, self._on_peer_found)
            self._mdns.start()

            # Thread de gossip periódico (cada 30s)
            self._gossip_thread = threading.Thread(
                target=self._gossip_loop, daemon=True, name="gossip-loop"
            )
            self._gossip_thread.start()

            self.stats.mode = "lan"
            log.info(f"SwarmNetwork [lan] — {self.identity}")
            print(f"🌐 SwarmNetwork activa: {self.identity}")
            print(f"   Feromonas: UDP :{self.identity.feromon_port}")
            print(f"   Gossip:    TCP :{self.identity.gossip_port}")
            print(f"   Descubrimiento: mDNS LAN automático")
            # LSP v2
            if self.protocol == "v2":
                try:
                    from src.network.lsp_v2 import LSPNodeV2
                    from src.network.lsp import LSPIdentity
                    lsp_identity = LSPIdentity.generate()
                    # Puertos v2: offset +10 para no chocar con v1
                    self._lsp_v2_node = LSPNodeV2(
                        lsp_identity,
                        feromon_port=self.identity.feromon_port + 10,
                        gossip_port=self.identity.gossip_port + 10,
                    )
                    self._lsp_v2_node.start()
                    # Callbacks de feromonas v2 alimentan la misma tabla de peers
                    @self._lsp_v2_node.on_feromon_received
                    def _on_v2_feromon(feromon, node_id_hex):
                        self.stats.feromons_received += 1
                        self.peers.update_feromon(node_id_hex[:16], feromon)
                    print(f"   🐬 LSP v2: UDP :{self.identity.feromon_port + 10} (float16, merge-on-transit)")
                except Exception as e:
                    log.warning(f"LSP v2 no pudo arrancar: {e}")

        except Exception as e:
            if self.mode == "auto":
                log.warning(f"SwarmNetwork no pudo arrancar en modo LAN ({e}), fallback a local")
                self.stats.mode = "local"
                self._feromon_udp = None
                self._gossip_tcp = None
                self._mdns = None
            else:
                raise

    def stop(self):
        self._running = False
        if self._feromon_udp:
            self._feromon_udp.stop()
        if self._gossip_tcp:
            self._gossip_tcp.stop()
        if self._mdns:
            self._mdns.stop()
        if self._lsp_v2_node:
            self._lsp_v2_node.stop()
        log.info("SwarmNetwork detenida")

    # ─── API principal ────────────────────────────────────────────────────────

    def broadcast_feromon(self, feromon: torch.Tensor, agent_id: int = 0, fitness: float = 0.5):
        """
        Envía feromona a todos los peers conocidos.
        Usa LSP v2 (binary float16) si está disponible, sino cae a v1.
        En modo local: no-op silencioso.
        """
        # Preferir LSP v2 si disponible
        if self._lsp_v2_node is not None:
            try:
                self._lsp_v2_node.send_feromon_v2(feromon.detach().cpu(), fitness=fitness, step=0)
                self.stats.feromons_sent += max(1, len(self.peers.alive_peers()))
                return
            except Exception as e:
                log.debug(f"broadcast_feromon v2 error: {e}, falling back to v1")
        # Fallback v1
        if not self._feromon_udp:
            return
        peers = self.peers.alive_peers()
        if not peers:
            return
        msg = FeromonMessage(
            node_id=self.identity.node_id,
            agent_id=agent_id,
            timestamp_ms=int(time.time() * 1000),
            feromon=feromon.detach().cpu(),
        )
        self._feromon_udp.broadcast_lan(msg, peers)
        self.stats.feromons_sent += len(peers)

    def collect_feromons(self) -> List[torch.Tensor]:
        """
        Retorna las últimas feromonas de peers activos.
        En modo local: retorna [] siempre.
        """
        if not self.peers.count:
            return []
        return self.peers.collect_feromons()

    def get_combined_feromon(
        self,
        local_feromon: torch.Tensor,
        remote_weight: float = 0.3,
    ) -> torch.Tensor:
        """
        Combina feromona local con feromonas remotas.
        Si no hay peers: retorna local_feromon sin cambios.
        Parámetro remote_weight: qué tanto influyen los peers.
        """
        remote = self.collect_feromons()
        if not remote:
            return local_feromon
        remote_mean = torch.stack(remote).mean(dim=0).to(local_feromon.device)
        # Asegurar misma dimensión
        if remote_mean.shape != local_feromon.shape:
            remote_mean = F.interpolate(
                remote_mean.unsqueeze(0).unsqueeze(0),
                size=local_feromon.shape[-1],
                mode="linear", align_corners=False,
            ).squeeze()
        combined = (1 - remote_weight) * local_feromon + remote_weight * remote_mean
        return F.normalize(combined, dim=-1) * local_feromon.norm()

    def merge_remote_feromons(
        self,
        local_feromon: torch.Tensor,
        remote_weight: float = 0.3,
    ) -> torch.Tensor:
        """Alias de get_combined_feromon para compatibilidad."""
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
        }

    # ─── Callbacks internos ───────────────────────────────────────────────────

    def _on_feromon_received(self, msg: FeromonMessage, from_addr: str):
        """Callback cuando llega una feromona de un peer."""
        self.peers.update_feromon(msg.node_id, msg.feromon)
        self.stats.feromons_received += 1
        log.debug(f"Feromona de {msg.node_id[:8]} agent={msg.agent_id}")

        # Si tenemos MatriarcaDual, actualizar banco global con la feromona remota
        if self.swarm and self.swarm.matriarca is not None:
            from src.matriarca.matriarca_legacy import MatriarcaEnriched, MatriarcaDual
            mat = self.swarm.matriarca
            # Llegar a MatriarcaDual si está disponible
            dual = None
            if isinstance(mat, MatriarcaEnriched) and isinstance(getattr(mat, '_dual', None), MatriarcaDual):
                dual = mat._dual
            elif isinstance(mat, MatriarcaDual):
                dual = mat
            if dual is not None:
                import torch
                emb = msg.feromon if isinstance(msg.feromon, torch.Tensor) else torch.tensor(msg.feromon)
                emb = emb.float()
                if emb.shape[0] != dual.global_mat.cfg.embd_dim:
                    # Proyección simple si las dimensiones no coinciden
                    emb = emb[:dual.global_mat.cfg.embd_dim].clone()
                dual.merge_global_update(
                    emb.unsqueeze(0),
                    [{"text": f"feromon@{msg.node_id[:8]} step=remote", "importance": 0.4}]
                )

    def _on_peer_found(self, peer: Peer):
        """Callback cuando mDNS descubre un nuevo peer."""
        is_new = self.peers.add(peer)
        if is_new:
            self.stats.peers_known += 1
            print(f"🌐 Nuevo peer: {peer}")
            if self._on_peer_connected:
                self._on_peer_connected(peer)
            # Ping inmediato al nuevo peer
            if self._gossip_tcp:
                ping = GossipMessage.make_ping(
                    self.identity.node_id,
                    self.identity.feromon_port,
                    self.identity.gossip_port,
                )
                self._gossip_tcp.send(peer, ping)

    def _on_gossip_received(self, msg: GossipMessage, from_addr: str) -> Optional[GossipMessage]:
        """Callback para mensajes gossip entrantes. Retorna respuesta opcional."""
        if msg.kind == "ping":
            # Responder con nuestro digest
            peer = Peer(
                node_id=msg.node_id,
                host=from_addr,
                feromon_port=msg.payload.get("feromon_port", 4444),
                gossip_port=msg.payload.get("gossip_port", 4445),
            )
            self.peers.add(peer)
            if self.swarm and self.swarm.matriarca:
                m = self.swarm.matriarca
                digest = GossipMessage.make_digest(
                    self.identity.node_id,
                    memory_count=m.memory_count,
                    newest_ts=max((meta.get("timestamp", 0) for meta in m.bank.metadata), default=0),
                    bank_hash=self._bank_hash(),
                )
                return digest

        elif msg.kind == "digest":
            # Si tienen más memorias recientes que nosotros, pedirlas
            if self.swarm and self.swarm.matriarca:
                our_newest = max(
                    (meta.get("timestamp", 0) for meta in self.swarm.matriarca.bank.metadata),
                    default=0,
                )
                their_newest = msg.payload.get("newest_ts", 0)
                if their_newest > our_newest + 60:  # tienen memorias >1min más nuevas
                    return GossipMessage.make_request(self.identity.node_id, since_ts=our_newest)

        elif msg.kind == "request":
            # Enviar memorias más nuevas que since_ts
            if self.swarm and self.swarm.matriarca:
                since = msg.payload.get("since_ts", 0)
                new_mems = [
                    {
                        "text": meta["text"],
                        "timestamp": meta["timestamp"],
                        "importance": meta["importance"],
                        # No enviamos embeddings por defecto (pesados) — solo metadatos
                        # El receptor puede re-encodificarlos si quiere
                    }
                    for meta in self.swarm.matriarca.bank.metadata
                    if meta.get("timestamp", 0) > since
                ][:50]  # máximo 50 memorias por request
                return GossipMessage.make_memories(self.identity.node_id, new_mems)

        elif msg.kind == "memories":
            # Integrar memorias recibidas (solo metadatos por ahora)
            if self.swarm and self.swarm.matriarca:
                memories = msg.payload.get("memories", [])
                import torch as _torch
                for mem in memories:
                    # Crear embedding sintético para memoria recibida
                    emb = _torch.randn(self.swarm.matriarca.cfg.embd_dim) * 0.1
                    self.swarm.matriarca.store_interaction(
                        emb,
                        text=f"[gossip:{msg.node_id[:8]}] {mem.get('text', '')[:100]}",
                        importance=mem.get("importance", 0.5) * 0.8,  # descuento por ser remota
                        auto_compress=False,
                    )
                self.stats.gossip_rounds += 1
                log.info(f"Gossip: {len(memories)} memorias de {msg.node_id[:8]}")

        return None

    def _gossip_loop(self):
        """Gossip periódico con peers (cada 30s)."""
        while self._running:
            time.sleep(30)
            if not self._gossip_tcp:
                break
            peers = self.peers.alive_peers()
            if not peers or not self.swarm or not self.swarm.matriarca:
                continue
            # Gossip con 1 peer aleatorio
            import random
            peer = random.choice(peers)
            m = self.swarm.matriarca
            digest = GossipMessage.make_digest(
                self.identity.node_id,
                memory_count=m.memory_count,
                newest_ts=max((meta.get("timestamp", 0) for meta in m.bank.metadata), default=0),
                bank_hash=self._bank_hash(),
            )
            response = self._gossip_tcp.send(peer, digest, recv_response=True)
            if response:
                self._on_gossip_received(response, peer.host)
            # Limpiar peers muertos
            self.peers.mark_dead()

    def _bank_hash(self) -> str:
        """Hash rápido del banco de memorias para detectar cambios."""
        if not self.swarm or not self.swarm.matriarca:
            return "empty"
        texts = [m["text"][:50] for m in self.swarm.matriarca.bank.metadata[-10:]]
        return hex(hash(tuple(texts)))[-8:]
