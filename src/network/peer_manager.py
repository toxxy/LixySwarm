"""Persistent peer discovery and address management for LSP v3.

The seed is only an initial address source. Once peers exchange addresses, the
manager keeps a target number of outbound sessions without consulting a
privileged coordinator.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import logging
import math
import re
import socket
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional


log = logging.getLogger("lixy.network.peers")

MAX_KNOWN_PEERS = 2048
MAX_PEER_AGE_S = 30 * 24 * 60 * 60
MAX_REPUTATION_ENTRIES = 4096
_HOSTNAME_RE = re.compile(
    r"^(?=.{1,253}$)(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)*"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$"
)


def validate_peer_address(host: str, port: int, *, allow_private: bool) -> bool:
    """Reject malformed, unroutable, and unsafe peer advertisements."""
    if not isinstance(host, str) or not host or len(host) > 253:
        return False
    if not isinstance(port, int) or not 0 < port < 65536:
        return False
    if host.lower() == "localhost":
        return allow_private

    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return bool(_HOSTNAME_RE.fullmatch(host))

    if address.is_unspecified or address.is_multicast or address.is_link_local:
        return False
    if address.is_loopback or address.is_private or address.is_reserved:
        return allow_private
    return True


def network_group(host: str) -> str:
    """Coarse address group used to prefer independent network paths."""
    try:
        address = ipaddress.ip_address(host)
        if address.version == 4:
            octets = address.packed
            return f"v4:{octets[0]:02x}{octets[1]:02x}"
        return f"v6:{address.packed[:4].hex()}"
    except ValueError:
        labels = host.lower().rstrip(".").split(".")
        return "dns:" + ".".join(labels[-2:])


@dataclass
class ReputationEntry:
    score: float = 0.0
    updated_at: float = 0.0
    banned_until: float = 0.0
    ban_count: int = 0


class PeerReputation:
    """Local-only decaying misbehavior scores with hashed persistent keys."""

    BAN_THRESHOLD = 100.0
    DECAY_PER_SECOND = 1.0 / 60.0
    BASE_BAN_SECONDS = 60 * 60

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._entries: dict[str, ReputationEntry] = {}
        self._lock = threading.RLock()
        self._load()

    @staticmethod
    def _key(kind: str, value: str) -> str:
        return hashlib.sha256(f"{kind}:{value}".encode("utf-8")).hexdigest()

    def _load(self):
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text())
            if payload.get("version") != 1:
                return
            for key, value in list(payload.get("entries", {}).items())[
                :MAX_REPUTATION_ENTRIES
            ]:
                if re.fullmatch(r"[0-9a-f]{64}", key) and isinstance(value, dict):
                    entry = ReputationEntry(
                        score=float(value.get("score", 0.0)),
                        updated_at=float(value.get("updated_at", 0.0)),
                        banned_until=float(value.get("banned_until", 0.0)),
                        ban_count=int(value.get("ban_count", 0)),
                    )
                    if not all(math.isfinite(item) for item in (
                        entry.score, entry.updated_at, entry.banned_until
                    )):
                        continue
                    entry.score = min(max(entry.score, 0.0), 1000.0)
                    entry.ban_count = min(max(entry.ban_count, 0), 16)
                    self._entries[key] = entry
        except Exception as exc:
            log.warning("Ignoring invalid peer reputation file: %s", exc)

    def _decay(self, entry: ReputationEntry, now: float):
        if entry.updated_at > 0:
            elapsed = max(0.0, now - entry.updated_at)
            entry.score = max(0.0, entry.score - elapsed * self.DECAY_PER_SECOND)
        entry.updated_at = now

    def report(self, kind: str, value: str, points: float = 25.0) -> bool:
        key = self._key(kind, value)
        now = time.time()
        with self._lock:
            entry = self._entries.setdefault(key, ReputationEntry(updated_at=now))
            self._decay(entry, now)
            entry.score = min(1000.0, entry.score + max(0.0, float(points)))
            if entry.score >= self.BAN_THRESHOLD:
                entry.ban_count = min(entry.ban_count + 1, 16)
                duration = min(
                    7 * 24 * 60 * 60,
                    self.BASE_BAN_SECONDS * (2 ** (entry.ban_count - 1)),
                )
                entry.banned_until = max(entry.banned_until, now + duration)
                entry.score = 0.0
            self._trim_and_save_locked(now)
            return entry.banned_until > now

    def reward(self, kind: str, value: str, points: float = 5.0):
        key = self._key(kind, value)
        now = time.time()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            self._decay(entry, now)
            entry.score = max(0.0, entry.score - max(0.0, float(points)))
            self._trim_and_save_locked(now)

    def is_banned(self, kind: str, value: str) -> bool:
        key = self._key(kind, value)
        now = time.time()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return False
            self._decay(entry, now)
            return entry.banned_until > now

    def _trim_and_save_locked(self, now: float):
        removable = [
            key for key, entry in self._entries.items()
            if entry.banned_until <= now and entry.score <= 0 and entry.ban_count <= 0
        ]
        for key in removable:
            self._entries.pop(key, None)
        if len(self._entries) > MAX_REPUTATION_ENTRIES:
            ordered = sorted(
                self._entries.items(),
                key=lambda item: (item[1].banned_until, item[1].score, item[1].updated_at),
                reverse=True,
            )[:MAX_REPUTATION_ENTRIES]
            self._entries = dict(ordered)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps({
            "version": 1,
            "updated_at": now,
            "entries": {
                key: asdict(entry) for key, entry in self._entries.items()
            },
        }, sort_keys=True, separators=(",", ":")))
        temporary.replace(self.path)
        try:
            self.path.chmod(0o600)
        except OSError:
            pass


@dataclass
class PeerAddress:
    host: str
    port: int
    node_id: str = ""
    source: str = "peer"
    first_seen: float = 0.0
    last_seen: float = 0.0
    last_attempt: float = 0.0
    failures: int = 0
    successes: int = 0

    def __post_init__(self):
        now = time.time()
        if not self.first_seen:
            self.first_seen = now
        if not self.last_seen:
            self.last_seen = now

    @property
    def key(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def retry_after(self) -> float:
        if self.failures <= 0:
            return 0.0
        return self.last_attempt + min(300.0, 2.0 ** min(self.failures, 8))


class AddressBook:
    """Thread-safe, atomic, bounded peer address persistence."""

    def __init__(self, path: str | Path, *, allow_private: bool = False):
        self.path = Path(path)
        self.allow_private = allow_private
        self._entries: dict[str, PeerAddress] = {}
        self._lock = threading.RLock()
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text())
            now = time.time()
            for item in payload.get("peers", []):
                entry = PeerAddress(**{
                    key: value for key, value in item.items()
                    if key in PeerAddress.__dataclass_fields__
                })
                if now - entry.last_seen > MAX_PEER_AGE_S:
                    continue
                if validate_peer_address(
                    entry.host, entry.port, allow_private=self.allow_private
                ):
                    self._entries[entry.key] = entry
        except Exception as exc:
            log.warning("Ignoring invalid LSP v3 address book: %s", exc)

    def _save_locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(
            self._entries.values(),
            key=lambda item: (item.successes - item.failures, item.last_seen),
            reverse=True,
        )[:MAX_KNOWN_PEERS]
        payload = {
            "version": 1,
            "updated": time.time(),
            "peers": [asdict(item) for item in ordered],
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
        temporary.replace(self.path)

    def add(
        self,
        host: str,
        port: int,
        *,
        node_id: str = "",
        source: str = "peer",
    ) -> bool:
        if not validate_peer_address(host, port, allow_private=self.allow_private):
            return False
        with self._lock:
            key = f"{host}:{port}"
            existing = self._entries.get(key)
            if existing:
                existing.last_seen = time.time()
                existing.node_id = node_id or existing.node_id
                if existing.source != "seed":
                    existing.source = source
            else:
                self._entries[key] = PeerAddress(
                    host=host,
                    port=port,
                    node_id=node_id,
                    source=source,
                )
            self._trim_locked()
            self._save_locked()
        return True

    def add_many(self, peers: Iterable[dict], *, source: str = "peer") -> int:
        added = 0
        with self._lock:
            for peer in peers:
                if not isinstance(peer, dict):
                    continue
                host = peer.get("host")
                try:
                    port = int(peer.get("port", 0))
                except (TypeError, ValueError):
                    continue
                # Public peer exchange accepts numeric addresses only. DNS
                # names are reserved for locally configured seed endpoints so
                # an untrusted peer cannot trigger DNS rebinding/SSRF.
                if not self.allow_private:
                    try:
                        ipaddress.ip_address(host)
                    except (TypeError, ValueError):
                        continue
                if not validate_peer_address(
                    host, port, allow_private=self.allow_private
                ):
                    continue
                key = f"{host}:{port}"
                node_id = str(peer.get("node_id", ""))[:64]
                existing = self._entries.get(key)
                if existing:
                    existing.last_seen = time.time()
                    existing.node_id = node_id or existing.node_id
                    if existing.source != "seed":
                        existing.source = source
                else:
                    self._entries[key] = PeerAddress(
                        host=host,
                        port=port,
                        node_id=node_id,
                        source=source,
                    )
                added += 1
            if added:
                self._trim_locked()
                self._save_locked()
        return added

    def mark_attempt(self, host: str, port: int, *, success: bool, node_id: str = ""):
        with self._lock:
            entry = self._entries.get(f"{host}:{port}")
            if not entry:
                return
            entry.last_attempt = time.time()
            if success:
                entry.successes += 1
                entry.failures = 0
                entry.last_seen = time.time()
                entry.node_id = node_id or entry.node_id
            else:
                entry.failures = min(entry.failures + 1, 32)
            self._save_locked()

    def candidates(
        self,
        *,
        excluded_node_ids: set[str],
        excluded_network_groups: Optional[set[str]] = None,
        limit: int = 32,
    ) -> list[PeerAddress]:
        now = time.time()
        with self._lock:
            candidates = [
                PeerAddress(**asdict(entry))
                for entry in self._entries.values()
                if entry.node_id not in excluded_node_ids
                and now >= entry.retry_after
            ]
        candidates.sort(
            key=lambda item: (
                item.successes - item.failures,
                item.source != "seed",
                item.last_seen,
            ),
            reverse=True,
        )
        # Prefer one candidate per coarse network group before filling from a
        # repeated group. This is a preference, not a hard ban, so small/LAN
        # networks can still reach the outbound target.
        diverse = []
        repeated = []
        seen_groups = set(excluded_network_groups or set())
        for candidate in candidates:
            group = network_group(candidate.host)
            if group in seen_groups:
                repeated.append(candidate)
            else:
                seen_groups.add(group)
                diverse.append(candidate)
        return (diverse + repeated)[:limit]

    def export(self, limit: int = 100) -> list[dict]:
        with self._lock:
            entries = sorted(
                self._entries.values(), key=lambda item: item.last_seen, reverse=True
            )[:limit]
            return [
                {
                    "host": item.host,
                    "port": item.port,
                    "node_id": item.node_id,
                    "last_seen": item.last_seen,
                }
                for item in entries
            ]

    def _trim_locked(self):
        if len(self._entries) <= MAX_KNOWN_PEERS:
            return
        ordered = sorted(
            self._entries.values(),
            key=lambda item: (item.successes - item.failures, item.last_seen),
            reverse=True,
        )[:MAX_KNOWN_PEERS]
        self._entries = {item.key: item for item in ordered}

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._entries)


async def resolve_seed(host: str, port: int) -> list[tuple[str, int]]:
    """Resolve all A/AAAA results for a DNS seed without blocking the event loop."""
    loop = asyncio.get_running_loop()
    try:
        records = await loop.getaddrinfo(
            host,
            port,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
        )
    except OSError:
        return []
    result: list[tuple[str, int]] = []
    for _, _, _, _, sockaddr in records:
        address = (sockaddr[0], int(sockaddr[1]))
        if address not in result:
            result.append(address)
    return result


class PeerManager:
    """Maintain outbound connectivity after bootstrap seeds disappear."""

    def __init__(
        self,
        node,
        *,
        address_book: AddressBook,
        seeds: Optional[list[tuple[str, int]]] = None,
        target_outbound: int = 8,
        maintenance_interval: float = 2.0,
    ):
        self.node = node
        self.address_book = address_book
        self.seeds = list(seeds or [])
        self.target_outbound = max(0, int(target_outbound))
        self.maintenance_interval = max(0.1, float(maintenance_interval))
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._wake = asyncio.Event()

    async def start(self):
        if self._running:
            return
        self._running = True
        for host, port in self.seeds:
            # Preserve DNS names as retryable seed entries and also cache every
            # resolved address. Multiple records provide redundant bootstrap.
            self.address_book.add(host, port, source="seed")
            for resolved_host, resolved_port in await resolve_seed(host, port):
                self.address_book.add(resolved_host, resolved_port, source="seed")
        self._task = asyncio.create_task(self._maintain(), name="lsp-v3-peer-manager")

    async def stop(self):
        self._running = False
        self._wake.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def add_discovered(self, peers: Iterable[dict]) -> int:
        added = self.address_book.add_many(peers, source="peer")
        if added:
            self._wake.set()
        return added

    def connected(
        self,
        node_id: str,
        host: str,
        port: int,
        *,
        outbound: bool,
    ):
        self.address_book.add(host, port, node_id=node_id, source="peer")
        self.address_book.mark_attempt(host, port, success=True, node_id=node_id)
        self._wake.set()

    def disconnected(self):
        self._wake.set()

    def addresses_for_exchange(self, limit: int = 100) -> list[dict]:
        return self.address_book.export(limit=limit)

    async def _maintain(self):
        while self._running:
            try:
                deficit = self.target_outbound - self.node.outbound_count
                if deficit > 0:
                    excluded = self.node.connected_node_ids
                    candidates = self.address_book.candidates(
                        excluded_node_ids=excluded,
                        excluded_network_groups=self.node.connected_network_groups,
                        limit=max(deficit * 4, 8),
                    )
                    for candidate in candidates:
                        if not self._running or self.node.outbound_count >= self.target_outbound:
                            break
                        if self.node.is_address_connected(candidate.host, candidate.port):
                            continue
                        success = await self.node._connect_address(
                            candidate.host, candidate.port
                        )
                        self.address_book.mark_attempt(
                            candidate.host,
                            candidate.port,
                            success=success,
                        )

                self._wake.clear()
                try:
                    await asyncio.wait_for(
                        self._wake.wait(), timeout=self.maintenance_interval
                    )
                except asyncio.TimeoutError:
                    pass
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.debug("Peer maintenance error: %s", exc)
                await asyncio.sleep(self.maintenance_interval)
