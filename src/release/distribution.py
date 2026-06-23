"""Peer-to-peer acquisition and relay of trusted release manifests."""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from .manifest import ReleaseError, ReleaseManifest


log = logging.getLogger("lixy.release.distribution")


class ReleaseDistributor:
    def __init__(
        self,
        network,
        registry,
        policy,
        artifact_store,
        *,
        auto_activate: bool = False,
    ):
        if network._lsp_v3_node is None or network.artifact_service is None:
            raise RuntimeError("release distribution requires active v3 artifacts")
        self.network = network
        self.node = network._lsp_v3_node
        self.registry = registry
        self.policy = policy
        self.store = artifact_store
        self.auto_activate = bool(auto_activate)
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="lixy-release"
        )
        self.node.on_release_announced(self._on_announcement)

    def close(self):
        self._executor.shutdown(wait=True, cancel_futures=True)

    def announce_active(self, peer_id: str | None = None) -> bool | int:
        active = self.registry.active()
        if active is None:
            return False if peer_id else 0
        if peer_id:
            return self.node.send_release_announcement(peer_id, active.to_dict())
        return self.node.announce_release(active.to_dict())

    def peer_connected(self, peer_id: str):
        # LSP connection callbacks run on the transport event loop; schedule
        # the synchronous targeted send elsewhere to avoid loop self-deadlock.
        self._executor.submit(self.announce_active, peer_id)

    def _on_announcement(self, value: dict, from_node_id: str):
        try:
            manifest = ReleaseManifest.from_dict(value)
            self.policy.authorize(manifest)
        except ReleaseError:
            return
        with self._lock:
            if manifest.release_id in self._seen:
                return
            self._seen[manifest.release_id] = None
            while len(self._seen) > 2048:
                self._seen.popitem(last=False)
        self._executor.submit(self._acquire, manifest, from_node_id)

    def _acquire(self, manifest: ReleaseManifest, from_node_id: str):
        try:
            for artifact_id in (
                manifest.model_artifact_id,
                manifest.config_artifact_id,
                manifest.tokenizer_artifact_id,
            ):
                if artifact_id and not self.store.has(artifact_id):
                    self.network.fetch_artifact(
                        artifact_id, peer_id=from_node_id, timeout_s=300.0
                    )
            self.registry.accept(manifest, self.policy, self.store)
            if self.auto_activate:
                try:
                    self.registry.activate(manifest.release_id, self.policy, self.store)
                except ReleaseError:
                    pass
            self.node.announce_release(manifest.to_dict())
        except Exception as exc:
            log.debug("Release acquisition failed: %s", exc)
