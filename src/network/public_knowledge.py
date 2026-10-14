"""Durable public model replication on a machine's existing LSP identity.

Only this explicitly public store is served. No tensor engine, personal bank,
conversation, dataset directory, or peer-supplied code is opened here.
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

from src.contribution.policy import ContributionPolicy
from src.contribution.resource_governor import ResourceGovernor
from src.release import ReleaseError, ReleaseManifest, ReleaseRegistry
from src.release.history import ReleaseHistoryService, fetch_release_history
from src.release.reference_catalog import REFERENCE_CATALOG
from .artifact_store import ArtifactError, ArtifactService, ArtifactStore
from .telemetry import atomic_json
from .work_protocol import WorkCoordinator

log = logging.getLogger("lixy.knowledge")
MAX_PUBLIC_STORE = 512 * 1024**2


@dataclass
class KnowledgeSettings:
    version: int = 1
    enabled: bool = False
    consented_at: float = 0

    def validate(self):
        if type(self.version) is not int or self.version != 1 or type(self.enabled) is not bool:
            raise ValueError("Invalid public knowledge settings")
        if type(self.consented_at) not in (int, float) or not math.isfinite(self.consented_at) or self.consented_at < 0:
            raise ValueError("Invalid public storage consent")
        if self.enabled and self.consented_at <= 0:
            raise ValueError("Public storage requires local consent")

    def save(self, directory):
        self.validate()
        atomic_json(Path(directory) / "public-knowledge.json", asdict(self))

    @classmethod
    def load(cls, directory):
        path = Path(directory) / "public-knowledge.json"
        if not path.exists():
            return cls()
        if path.stat().st_size > 1024:
            raise ValueError("Public knowledge settings exceed their bound")
        value = json.loads(path.read_text())
        if set(value) != {"version", "enabled", "consented_at"}:
            raise ValueError("Invalid public knowledge settings schema")
        settings = cls(**value); settings.validate()
        return settings


class BandwidthBudget:
    """Shared interruptible budget for encoded artifact chunks, not CPU work."""
    def __init__(self, rate):
        self.rate = rate
        self.lock = threading.Lock()
        self.tokens = 0.0
        self.updated = time.monotonic()

    def wait(self, size, *, cancel_event=None, deadline=None):
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise ArtifactError("Artifact transfer paused or cancelled")
            if deadline is not None and time.time() >= deadline:
                raise ArtifactError("Artifact bandwidth deadline exceeded")
            with self.lock:
                now = time.monotonic()
                rate = max(0.0, min(float(self.rate()), 5.0)) * 1_000_000 / 8
                self.tokens = min(max(size, rate * 0.25), self.tokens + (now-self.updated) * rate)
                self.updated = now
                if rate <= 0:
                    raise ArtifactError("Artifact bandwidth is disabled")
                if self.tokens >= size:
                    self.tokens -= size
                    return
                delay = min(0.05, (size-self.tokens)/rate)
            if cancel_event is not None:
                cancel_event.wait(delay)
            else:
                time.sleep(delay)


class PublicKnowledgeService:
    def __init__(self, node, directory, resources, *, catalog=REFERENCE_CATALOG):
        self.node, self.directory, self.catalog = node, Path(directory), catalog
        self.resources=resources
        self.release_lock=threading.RLock()
        self.catalog.policy.validate()
        root = self.directory / "public"
        self.store = ArtifactStore(root / "artifacts", max_artifact_bytes=32*1024**2,
                                   max_total_bytes=MAX_PUBLIC_STORE)
        self.registry = ReleaseRegistry(root / "releases")
        hardware = SimpleNamespace(**resources, disk_gb=MAX_PUBLIC_STORE/1024**3)
        self.governor = ResourceGovernor(ContributionPolicy.for_mode("disabled"),
            hardware=hardware, storage_path=root)
        self.work = WorkCoordinator(node, self.governor, max_workers=2,
            max_offers_per_peer=4, max_offers_per_minute=180)
        self.artifacts = ArtifactService(self.work, self.store)
        history = ReleaseHistoryService(self.work, self.registry, catalog.policy)
        self.cancel = threading.Event(); self.cancel.set()
        self.policy = ContributionPolicy.for_mode("disabled")
        self.lock = threading.RLock()
        self.thread = None
        self.next_attempt = 0.0
        self.allowed = False
        self.closed = False
        self.complete = False
        self.desired_release_id=catalog.release_id
        self.desired_sequence=-1
        self.head_path=root/"catalog-head.json"
        if catalog.allow_signed_updates and self.head_path.exists():
            head=ReleaseManifest.from_dict(json.loads(self.head_path.read_text()))
            catalog.policy.authorize(head)
            self.desired_release_id=head.release_id; self.desired_sequence=head.sequence
        self._status = {"state":"disabled", "family":catalog.family, "release_id":None,
                        "model_id":None, "sequence":None, "error":None, "versions_done":0}
        self.budget = BandwidthBudget(lambda: self.policy.max_bandwidth_mbps if self.allowed else 0)
        self.artifacts.receive_budget = self.budget
        self.work.register_handler("artifact.describe.v1", "artifact", self._guard(self.artifacts._describe),permit_cached=self._permit_cached)
        self.work.register_handler("artifact.read-chunk.v1", "artifact", self._guard(self._chunk),permit_cached=self._permit_cached)
        self.work.register_handler("release.history.v1", "artifact", self._guard(history._page),permit_cached=self._permit_cached)
        node.capabilities["public_artifacts_v1"] = True
        node.on_release_announced(self._on_release)
        node.on_peer_connected(self._announce_current)
        self.restore()

    def _announce_current(self,peer_id,*_):
        if not self.allowed or not self.complete: return
        try:
            active=self.registry.active()
            if active: self.node.send_release_announcement(peer_id,active.to_dict())
        except (OSError,ValueError):
            pass

    def _on_release(self,value,provider_id):
        if not self.catalog.allow_signed_updates: return
        try:
            manifest=ReleaseManifest.from_dict(value)
            self.catalog.policy.authorize(manifest)
            with self.lock:
                active=self.registry.active()
                baseline=max(self.desired_sequence,active.sequence if active else -1)
                if manifest.sequence<=baseline: return
                if manifest.sequence>=self.catalog.max_chain_length: return
                atomic_json(self.head_path,manifest.to_dict())
                self.desired_release_id=manifest.release_id; self.desired_sequence=manifest.sequence
                self.complete=False; self.next_attempt=0
        except (OSError,ValueError,TypeError):
            return

    def promoted(self,manifest):
        self.catalog.policy.authorize(manifest)
        atomic_json(self.head_path,manifest.to_dict())
        self.desired_release_id=manifest.release_id; self.desired_sequence=manifest.sequence
        self.restore()
        self.node.announce_release(manifest.to_dict())

    def _set_status(self, **values):
        with self.lock:
            self._status.update(values)

    def _guard(self, handler):
        def guarded(payload, execution):
            execution.raise_if_cancelled()
            if not self.allowed:
                raise ArtifactError("Public sharing is disabled or paused")
            return handler(payload, execution)
        return guarded

    def _chunk(self, payload, execution):
        value = self.artifacts._read_chunk(payload, execution)
        self.budget.wait(len(value["data"])+2048, cancel_event=self.cancel, deadline=execution.deadline)
        execution.raise_if_cancelled()
        return value

    def _permit_cached(self, execution, output):
        if not self.allowed: return False
        if execution.operation=="artifact.read-chunk.v1":
            self.budget.wait(len(output["data"])+2048,cancel_event=self.cancel,deadline=execution.deadline)
        return self.allowed

    def restore(self):
        try:
            active = self.registry.active()
            if active is None:
                return
            active = self.registry.verify_active(self.catalog.policy, self.store)
            if self.catalog.allow_signed_updates and active.sequence>=self.desired_sequence:
                self.desired_release_id=active.release_id; self.desired_sequence=active.sequence
                atomic_json(self.head_path,active.to_dict())
            self.complete = active.release_id == self.desired_release_id
            self._set_status(release_id=active.release_id, model_id=active.model_artifact_id,
                             sequence=active.sequence, versions_done=active.sequence+1)
        except (OSError, ValueError) as exc:
            self._set_status(error=type(exc).__name__)

    def snapshot(self):
        with self.lock:
            return {**self._status, "stored_bytes":self.store.total_bytes(),
                    "sharing":self.allowed, "compute_ready":False}

    def update(self, settings, sharing):
        enabled = sharing.enabled and settings.connected
        self.policy = settings.policy
        limit = min(MAX_PUBLIC_STORE, int(settings.policy.max_disk_gb * 1024**3))
        self.store.max_total_bytes = max(1, limit)
        allowed = (enabled and not settings.paused and settings.policy.mode != "disabled"
                   and settings.policy.max_bandwidth_mbps > 0 and limit >= self.store.total_bytes() and limit > 0)
        self.allowed = allowed
        with self.governor._lock:
            self.governor.policy = settings.policy if settings.connected and not settings.paused else ContributionPolicy.for_mode("disabled")
        if not allowed:
            self.cancel.set()
            self._set_status(state="paused" if sharing.enabled else "disabled")
            return
        # Never replace a live acquisition's cancellation token on resume.
        if self.thread is not None and self.thread.is_alive():
            return
        self.cancel = threading.Event()
        if self.complete:
            self._set_status(state="available", error=None)
            return
        if time.monotonic() >= self.next_attempt and self.node.peer_count:
            self._set_status(state="discovering", error=None)
            self.thread = threading.Thread(target=self._acquire, args=(self.cancel,),
                daemon=True, name="lixy-public-acquire")
            self.thread.start()

    def _acquire(self, cancellation):
        try:
            candidates = [p["node_id"] for p in self.node.peers()
                          if p.get("capabilities", {}).get("public_artifacts_v1")]
            if not candidates:
                self._set_status(state="pending")
                return
            for peer in candidates[:8]:
                if cancellation.is_set():
                    return
                try:
                    target=self.desired_release_id
                    chain = fetch_release_history(self.work, release_id=target,
                        peer_id=peer, policy=self.catalog.policy, max_releases=self.catalog.max_chain_length,
                        timeout_s=20, cancel_event=cancellation)
                    self._set_status(state="downloading")
                    for release in chain:
                        for identifier in (release.model_artifact_id, release.config_artifact_id, release.tokenizer_artifact_id):
                            if identifier:
                                self.artifacts.fetch(identifier, peer_id=peer, timeout_s=60, cancel_event=cancellation)
                        if cancellation.is_set():
                            return
                        with self.release_lock:
                            self.registry.accept(release, self.catalog.policy, self.store)
                            active = self.registry.active()
                            if active is None or release.sequence > active.sequence:
                                self.registry.activate(release.release_id, self.catalog.policy, self.store)
                        self._set_status(versions_done=release.sequence+1)
                    self.restore()
                    self._set_status(state="available", error=None)
                    log.info("Public model acquired family=%s sequence=%s provider=%s", self.catalog.family,
                             self.registry.active().sequence, peer[:16])
                    self.node.announce_release(self.registry.active().to_dict())
                    return
                except (OSError, ValueError, RuntimeError) as exc:
                    if not cancellation.is_set():
                        self._set_status(error=type(exc).__name__, state="pending")
        finally:
            self.next_attempt = time.monotonic()+10

    def close(self):
        self.closed = True; self.allowed = False; self.cancel.set()
        self.work.close()
        if self.thread is not None:
            self.thread.join(timeout=10)


def import_public_run(source, directory, *, catalog=REFERENCE_CATALOG):
    """Explicit local migration of this pilot's signed public release chain."""
    source = Path(source)
    previous = ReleaseRegistry(source / "releases")
    source_store = ArtifactStore(source / "artifacts", max_artifact_bytes=32*1024**2,
                                 max_total_bytes=MAX_PUBLIC_STORE)
    cursor, chain = catalog.release_id, []
    while cursor:
        release = ReleaseManifest.load(previous.manifests / f"{cursor}.json")
        if release.release_id != cursor:
            raise ReleaseError("Public import ID does not match its manifest")
        catalog.policy.authorize(release)
        chain.append(release)
        if len(chain) > catalog.max_chain_length:
            raise ReleaseError("Public import exceeds catalog history bound")
        cursor = release.previous_release_id
    root = Path(directory) / "public"
    store = ArtifactStore(root / "artifacts", max_artifact_bytes=32*1024**2,
                          max_total_bytes=MAX_PUBLIC_STORE)
    registry = ReleaseRegistry(root / "releases")
    for release in reversed(chain):
        for identifier in (release.model_artifact_id, release.config_artifact_id, release.tokenizer_artifact_id):
            if identifier:
                manifest = source_store.manifest(identifier)
                if not source_store.has(identifier):
                    raise ArtifactError("Public import source failed content verification")
                imported = store.import_file(source_store.object_path(identifier), kind=manifest.kind,
                                            media_type=manifest.media_type)
                if imported.artifact_id != identifier:
                    raise ArtifactError("Public import changed artifact content")
        registry.accept(release, catalog.policy, store)
        active = registry.active()
        if active is None or release.sequence > active.sequence:
            registry.activate(release.release_id, catalog.policy, store)
    return registry.verify_active(catalog.policy, store)
