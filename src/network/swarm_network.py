"""SwarmNetwork facade.

LSP v3 is the default: persistent outbound-friendly sessions, signed frames,
anti-replay, seed bootstrap, peer exchange, and address persistence. LSP v2 is
retained only for compatibility and explicit migration testing.
"""
import os
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
from .bootstrap import PeersDB, bootstrap_network, encode_peer_list, get_seed_endpoints

log = logging.getLogger("lixy.network")


@dataclass
class NetworkStats:
    mode: str = "local"
    protocol: str = "v3"
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
            f"protocol={self.protocol} | mode={self.mode} | peers={self.peers_known} | "
            f"feromon_rx={self.feromons_received} | feromon_tx={self.feromons_sent} | "
            f"gossip={self.gossip_rounds} | global_rx={self.global_memories_received} | "
            f"uptime={uptime:.0f}s"
        )


class SwarmNetwork:
    """
    Network facade with LSP v3 by default and explicit LSP v2 compatibility.
    """

    @classmethod
    def create(cls, swarm=None, mode="auto", feromon_port=7337, gossip_port=7338,
               checkpoint_dir="checkpoints", protocol="v3", seeds=None,
               target_outbound=8, allow_private_peers=None,
               contribution_profile=None, identity_work_bits=None):
        if swarm is not None:
            identity = NodeIdentity.from_swarm(swarm, feromon_port=feromon_port, gossip_port=gossip_port)
        else:
            identity = NodeIdentity.generate_anonymous(
                feromon_port=feromon_port,
                gossip_port=gossip_port,
            )
        return cls(
            identity=identity,
            mode=mode,
            checkpoint_dir=checkpoint_dir,
            swarm=swarm,
            protocol=protocol,
            seeds=seeds,
            target_outbound=target_outbound,
            allow_private_peers=allow_private_peers,
            contribution_profile=contribution_profile,
            identity_work_bits=identity_work_bits,
        )

    def __init__(self, identity, mode="auto", checkpoint_dir="checkpoints", swarm=None,
                 protocol="v3", seeds=None, target_outbound=8,
                 allow_private_peers=None, contribution_profile=None,
                 identity_work_bits=None):
        self.identity = identity
        self.mode = mode
        self.protocol = str(protocol).lower()
        if self.protocol not in {"v2", "v3"}:
            raise ValueError("protocol must be 'v2' or 'v3'")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.swarm = swarm
        self.seeds = list(seeds) if seeds is not None else get_seed_endpoints()
        self.target_outbound = max(0, int(target_outbound))
        self.contribution_profile = (
            dict(contribution_profile) if contribution_profile is not None else None
        )
        default_work_bits = 0 if mode in {"lan", "test"} else int(
            os.environ.get("LIXYSWARM_IDENTITY_WORK_BITS", "0")
        )
        self.identity_work_bits = (
            default_work_bits if identity_work_bits is None
            else int(identity_work_bits)
        )
        if not 0 <= self.identity_work_bits <= 28:
            raise ValueError("identity_work_bits must be in [0, 28]")
        self.allow_private_peers = (
            mode in {"lan", "test"}
            if allow_private_peers is None
            else bool(allow_private_peers)
        )
        self.peers = PeerTable(self_id=identity.node_id)
        self.stats = NetworkStats(mode=mode, protocol=self.protocol)
        self.peers_db = PeersDB(str(self.checkpoint_dir / "peers.json"))
        self._lsp_v2_node = None
        self._lsp_v3_node = None
        self._running = False
        self._bootstrap_thread = None
        self._stop_event = threading.Event()
        self._global_sync_lock = threading.RLock()
        self._on_peer_connected = None
        self._on_peer_lost = None
        self.matriarca_dual = None
        self.work_coordinator = None
        self.artifact_service = None
        self.release_distributor = None
        self.useful_work_ledger = None

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

    def enable_work(self, governor, *, max_workers: int = 2):
        """Attach consent-gated typed work to the active LSP v3 node."""
        if self.protocol != "v3" or self._lsp_v3_node is None:
            raise RuntimeError("Distributed work requires an active LSP v3 node")
        if self.work_coordinator is None:
            from .work_protocol import WorkCoordinator
            self.work_coordinator = WorkCoordinator(
                self._lsp_v3_node,
                governor,
                max_workers=max_workers,
                minimum_identity_work_bits=self.identity_work_bits,
                scheduler_state_path=(
                    self.checkpoint_dir / "scheduler_history_v1.json"
                ),
                exploration_interval=int(os.environ.get(
                    "LIXYSWARM_EXPLORATION_INTERVAL", "5"
                )),
                exploration_minimum_age_s=float(os.environ.get(
                    "LIXYSWARM_EXPLORATION_MIN_AGE_S", "60"
                )),
                max_queued_offers=int(os.environ.get(
                    "LIXYSWARM_MAX_QUEUED_OFFERS", "16"
                )),
                max_offers_per_peer=int(os.environ.get(
                    "LIXYSWARM_MAX_OFFERS_PER_PEER", "2"
                )),
                max_offers_per_minute=int(os.environ.get(
                    "LIXYSWARM_MAX_OFFERS_PER_MINUTE", "12"
                )),
            )
        return self.work_coordinator

    def register_work_handler(self, operation: str, kind: str, handler):
        if self.work_coordinator is None:
            raise RuntimeError("Call enable_work() before registering handlers")
        self.work_coordinator.register_handler(operation, kind, handler)

    def submit_work(self, operation, payload, requirements, **kwargs):
        if self.work_coordinator is None:
            raise RuntimeError("Call enable_work() before submitting work")
        return self.work_coordinator.submit(
            operation, payload, requirements, **kwargs
        )

    def enable_artifacts(self, store):
        """Serve and fetch content-addressed artifacts over typed work."""
        if self.work_coordinator is None:
            raise RuntimeError("Call enable_work() before enabling artifacts")
        if self.artifact_service is None:
            from .artifact_store import ArtifactService
            self.artifact_service = ArtifactService(self.work_coordinator, store)
        return self.artifact_service

    def fetch_artifact(self, artifact_id: str, *, peer_id: str, **kwargs):
        if self.artifact_service is None:
            raise RuntimeError("Artifact service is not enabled")
        return self.artifact_service.fetch(
            artifact_id, peer_id=peer_id, **kwargs
        )

    def enable_release_distribution(
        self, registry, policy, store, *, auto_activate: bool = False
    ):
        if self.release_distributor is None:
            from src.release import ReleaseDistributor
            self.release_distributor = ReleaseDistributor(
                self, registry, policy, store, auto_activate=auto_activate
            )
            self.release_distributor.announce_active()
        return self.release_distributor

    def attach_useful_work_ledger(self, ledger):
        if self._lsp_v3_node is None:
            raise RuntimeError("useful-work ledger requires active LSP v3")
        if self.useful_work_ledger is ledger:
            return ledger
        if self.useful_work_ledger is not None:
            raise RuntimeError("a useful-work ledger is already attached")
        self.useful_work_ledger = ledger

        from src.network.useful_work import verify_useful_work_bundle

        @self._lsp_v3_node.on_useful_work_proofs
        def _proofs(value, from_node_id):
            try:
                evidence = verify_useful_work_bundle(
                    value,
                    worker_node_id=from_node_id,
                    firsthand_issuer_id=self._lsp_v3_node.identity.node_id_hex,
                )
            except (TypeError, ValueError):
                return
            self._lsp_v3_node.update_peer_useful_work(from_node_id, evidence)

        @self._lsp_v3_node.on_useful_work_credit
        def _credit(value, from_node_id):
            if value.get("issuer_node_id") != from_node_id:
                return
            try:
                ledger.add(value)
            except Exception:
                return
            self._lsp_v3_node.announce_useful_work_proofs()
        self._lsp_v3_node.set_useful_work_proof_provider(ledger.proof_bundle)
        return ledger

    def send_useful_work_credit(self, peer_id: str, credit: dict) -> bool:
        if self._lsp_v3_node is None:
            return False
        return self._lsp_v3_node.send_useful_work_credit(peer_id, credit)

    # ─── WAN / Relay ──────────────────────────────────────────────────────────

    def connect_peer(self, host: str, gossip_port: int = 7338):
        """Open a persistent v3 session or an explicit compatibility v2 peer."""
        node = self._lsp_v3_node if self.protocol == "v3" else self._lsp_v2_node
        if node is None:
            log.warning("Cannot connect to %s:%s — LSP %s not started", host, gossip_port, self.protocol)
            return False
        try:
            connected = bool(node.connect_peer(host, gossip_port))
            if self.protocol == "v2" or connected:
                self.peers_db.mark_connected(host, gossip_port)
            return connected if self.protocol == "v3" else True
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
        """Start the selected protocol; v3 is the default."""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self.stats.started_at = time.time()

        if self.mode == "local":
            log.info(f"SwarmNetwork [local] — {self.identity}")
            return

        try:
            if self.protocol == "v3":
                self._start_v3()
            else:
                self._start_v2()
        except Exception as e:
            log.warning("LSP %s could not start: %s", self.protocol, e)
            if self.mode == "auto":
                self.stats.mode = "local"
            else:
                raise

    def _start_v2(self):
        """Start the compatibility LSP v2 transport."""
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

        except Exception:
            raise

    def _start_v3(self):
        """Start persistent LSP v3 sessions and automatic peer maintenance."""
        from src.network.lsp_v3 import LSPNodeV3

        identity = self._load_or_create_lsp_identity()
        resources = dict(
            self.contribution_profile or self._local_resource_profile()
        )
        from src.network.identity_work import load_or_mine_identity_work
        resources["identity_work"] = load_or_mine_identity_work(
            self.checkpoint_dir / "identity_work_v1.json",
            identity.node_id_hex,
            bits=self.identity_work_bits,
        )
        self._lsp_v3_node = LSPNodeV3(
            identity,
            host="0.0.0.0",
            port=self.identity.gossip_port,
            advertised_host=os.environ.get("LIXYSWARM_PUBLIC_HOST"),
            seeds=self.seeds,
            address_book_path=self.checkpoint_dir / "peers_v3.json",
            target_outbound=self.target_outbound,
            allow_private=self.allow_private_peers,
            resource_profile=resources,
        )

        @self._lsp_v3_node.on_feromon_received
        def _on_v3_feromon(feromon, node_id_hex):
            self.stats.feromons_received += 1
            self.peers.update_feromon(node_id_hex, feromon)

        @self._lsp_v3_node.on_peer_connected
        def _on_v3_peer(node_id_hex, host, port, hello):
            peer = Peer(
                node_id=node_id_hex,
                host=host,
                feromon_port=port,
                gossip_port=port,
            )
            is_new = self.peers.add(peer)
            self.stats.peers_known = self._lsp_v3_node.peer_count
            self._register_runtime_node(node_id_hex, hello.get("resources", {}))
            if is_new and self._on_peer_connected:
                self._on_peer_connected(peer)
            if self.release_distributor is not None:
                try:
                    self.release_distributor.peer_connected(node_id_hex)
                except Exception:
                    pass

        @self._lsp_v3_node.on_peer_lost
        def _on_v3_peer_lost(node_id_hex):
            self.peers.remove(node_id_hex)
            self.stats.peers_known = self._lsp_v3_node.peer_count
            self._unregister_runtime_node(node_id_hex)
            if self._on_peer_lost:
                self._on_peer_lost(node_id_hex)

        @self._lsp_v3_node.on_gossip_delta_received
        def _on_v3_delta(delta, node_id_hex):
            self.stats.gossip_rounds += 1
            self.stats.global_deltas_received += 1
            if self.matriarca_dual is None:
                return
            try:
                with self._global_sync_lock:
                    merged = self.matriarca_dual.merge_global_delta(
                        delta, source_id=node_id_hex
                    )
                self.stats.global_memories_received += merged
            except Exception as exc:
                log.debug("Global Matriarca v3 merge error: %s", exc)

        self._lsp_v3_node.start()
        self.stats.mode = "p2p"
        log.info(
            "SwarmNetwork [LSP v3] — TCP:%s seeds=%s target_outbound=%s",
            self.identity.gossip_port,
            len(self.seeds),
            self.target_outbound,
        )

    def stop(self):
        self._running = False
        self._stop_event.set()
        if self.release_distributor:
            self.release_distributor.close()
            self.release_distributor = None
        if self.work_coordinator:
            self.work_coordinator.close()
            self.work_coordinator = None
        self.artifact_service = None
        if self._lsp_v3_node:
            self._lsp_v3_node.stop()
            self._lsp_v3_node = None
        if self._lsp_v2_node:
            self._lsp_v2_node.stop()
            self._lsp_v2_node = None
        if self._bootstrap_thread and self._bootstrap_thread.is_alive():
            self._bootstrap_thread.join(timeout=2.0)
        self._bootstrap_thread = None
        log.info("SwarmNetwork detenida")

    def _load_or_create_lsp_identity(self):
        from src.network.lsp import LSPIdentity
        identity_path = self.checkpoint_dir / "lsp_identity.pem"
        identity = LSPIdentity.load(str(identity_path))
        if identity is not None:
            return identity

        # Migrate a legacy per-MAC filename without retaining the MAC-derived
        # filename as the public identity convention.
        for legacy_path in sorted(self.checkpoint_dir.glob("lsp_identity_*.pem")):
            identity = LSPIdentity.load(str(legacy_path))
            if identity is not None:
                identity.save(str(identity_path))
                return identity

        identity = LSPIdentity.generate()
        identity.save(str(identity_path))
        log.info("New persistent LSP identity: %s...", identity.node_id_hex[:16])
        return identity

    def _local_resource_profile(self) -> dict:
        """Advertise bounded scheduling capabilities without operator details."""
        try:
            from src.swarm.node_manager import HardwareProfile

            hardware = HardwareProfile.from_local()
            mode = "moderate"
            if self.swarm is not None and hasattr(self.swarm, "node_manager"):
                local = self.swarm.node_manager.get_node("local")
                if local is not None:
                    hardware = local.hardware
                    mode = local.contribution_mode.value
            return {
                "mode": mode,
                "cpu_cores": max(1, min(int(hardware.cpu_cores), 1024)),
                "ram_gb": max(0.0, min(float(hardware.ram_gb), 16384.0)),
                "gpu_vram_gb": max(0.0, min(float(hardware.gpu_vram_gb), 1024.0)),
                "disk_gb": max(0.0, min(float(hardware.disk_gb), 1_000_000.0)),
                "has_gpu": bool(hardware.has_gpu),
            }
        except Exception:
            return {"mode": "relay", "cpu_cores": 1, "ram_gb": 0.0,
                    "gpu_vram_gb": 0.0, "disk_gb": 0.0, "has_gpu": False}

    def _register_runtime_node(self, node_id: str, resources: dict):
        if self.swarm is None or not hasattr(self.swarm, "node_manager"):
            return
        try:
            from src.swarm.node_manager import HardwareProfile, ContributionMode

            hardware = HardwareProfile(
                cpu_cores=max(1, min(int(resources.get("cpu_cores", 1)), 1024)),
                ram_gb=max(0.0, min(float(resources.get("ram_gb", 4.0)), 16384.0)),
                gpu_vram_gb=max(0.0, min(float(resources.get("gpu_vram_gb", 0.0)), 1024.0)),
                disk_gb=max(0.0, min(float(resources.get("disk_gb", 0.0)), 1_000_000.0)),
                has_gpu=bool(resources.get("has_gpu", False)),
            )
            self.swarm.node_manager.node_joined(node_id, hardware=hardware)
            mode_value = str(resources.get("mode", "relay")).lower()
            mode = next(
                (item for item in ContributionMode if item.value == mode_value),
                ContributionMode.RELAY,
            )
            self.swarm.node_manager.set_contribution_mode(node_id, mode)
        except Exception as exc:
            log.debug("Ignoring invalid remote resource profile: %s", exc)

    def _unregister_runtime_node(self, node_id: str):
        if self.swarm is None or not hasattr(self.swarm, "node_manager"):
            return
        try:
            self.swarm.node_manager.node_left(node_id, reason="network_disconnect")
        except Exception:
            pass

    # ─── API principal ────────────────────────────────────────────────────────

    def broadcast_feromon(self, feromon: torch.Tensor, agent_id: int = 0, fitness: float = 0.5):
        """Broadcast a compact pheromone over the active protocol."""
        try:
            if self.protocol == "v3":
                if self._lsp_v3_node is None:
                    return 0
                delivered = self._lsp_v3_node.broadcast_feromon(
                    feromon.detach().cpu(), fitness=fitness, step=0
                )
            else:
                if self._lsp_v2_node is None:
                    return 0
                self._lsp_v2_node.send_feromon_v2(
                    feromon.detach().cpu(), fitness=fitness, step=0
                )
                delivered = len(self.peers.alive_peers())
            self.stats.feromons_sent += delivered
            return delivered
        except Exception as e:
            log.debug("broadcast_feromon %s error: %s", self.protocol, e)
            return 0

    def broadcast_global_delta(self, max_items: int = 64, min_importance: float = 0.0) -> int:
        """Publica memorias globales compartibles de la MatriarcaDual adjunta."""
        if self.matriarca_dual is None:
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
            active_identity = (
                self._lsp_v3_node.identity.node_id_hex
                if self.protocol == "v3" and self._lsp_v3_node is not None
                else self.identity.node_id
            )
            delta["source_node_id"] = active_identity
            delta["feromon_port"] = self.identity.feromon_port
            delta["gossip_port"] = self.identity.gossip_port
            if self.protocol == "v3":
                if self._lsp_v3_node is None:
                    return 0
                delivered = self._lsp_v3_node.broadcast_global_delta(delta)
            else:
                if self._lsp_v2_node is None:
                    return 0
                self._lsp_v2_node.send_gossip_delta(delta)
                delivered = len(self.peers.alive_peers())
            self.stats.gossip_rounds += 1
            self.stats.global_deltas_sent += delivered
            return count if delivered else 0
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
            "protocol": self.protocol,
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
