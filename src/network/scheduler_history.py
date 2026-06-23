"""Private requester-local history for fair, identity-aged work exploration."""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from pathlib import Path


_NODE_ID_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_SCHEDULER_PEERS = 4096
MAX_COUNTER = 2 ** 53 - 1


class SchedulerHistory:
    """Persist only pseudonymous IDs and bounded local selection counters."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        exploration_interval: int = 5,
        minimum_age_s: float = 60.0,
    ):
        self.path = Path(path) if path is not None else None
        self.exploration_interval = max(0, min(int(exploration_interval), 10_000))
        self.minimum_age_s = max(0.0, min(float(minimum_age_s), 7 * 24 * 3600))
        self.dispatch_count = 0
        self._peers: dict[str, dict] = {}
        self._dirty = 0
        self._last_save = time.monotonic()
        self._lock = threading.RLock()
        self._load()

    def observe(self, node_ids: list[str], *, now: float | None = None):
        current = time.time()
        try:
            observed_at = current if now is None else float(now)
        except (TypeError, ValueError, OverflowError):
            observed_at = current
        if not math.isfinite(observed_at) or observed_at <= 0 or observed_at > current:
            observed_at = current
        with self._lock:
            for node_id in node_ids:
                if not _NODE_ID_RE.fullmatch(str(node_id)):
                    continue
                if node_id not in self._peers:
                    self._peers[node_id] = {
                        "first_seen_at": observed_at,
                        "last_selected_at": 0.0,
                        "selections": 0,
                    }
                    self._dirty += 1
            self._checkpoint_if_due()

    def exploration_due(self) -> bool:
        with self._lock:
            return (
                self.exploration_interval > 0
                and (self.dispatch_count + 1) % self.exploration_interval == 0
            )

    def is_aged(self, node_id: str, *, now: float | None = None) -> bool:
        current = time.time() if now is None else float(now)
        with self._lock:
            entry = self._peers.get(node_id)
            return bool(
                entry
                and current - float(entry["first_seen_at"]) >= self.minimum_age_s
            )

    def least_selected(self, node_ids: list[str]) -> str | None:
        with self._lock:
            eligible = [node_id for node_id in node_ids if node_id in self._peers]
            if not eligible:
                return None
            return min(eligible, key=lambda node_id: (
                int(self._peers[node_id]["selections"]),
                float(self._peers[node_id]["last_selected_at"]),
                node_id,
            ))

    def record_dispatch(
        self, node_ids: list[str], *, selected_at: float | None = None
    ):
        timestamp = time.time() if selected_at is None else float(selected_at)
        with self._lock:
            self.dispatch_count = min(self.dispatch_count + 1, MAX_COUNTER)
            for node_id in node_ids:
                entry = self._peers.get(node_id)
                if entry is None:
                    continue
                entry["selections"] = min(int(entry["selections"]) + 1, MAX_COUNTER)
                entry["last_selected_at"] = timestamp
            self._dirty += 1
            self._checkpoint_if_due()

    def close(self):
        with self._lock:
            if self._dirty:
                self._save()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "dispatch_count": self.dispatch_count,
                "known_identities": len(self._peers),
            }

    def _load(self):
        if self.path is None or not self.path.exists():
            return
        try:
            value = json.loads(self.path.read_text())
            if not isinstance(value, dict) or value.get("version") != 1:
                return
            dispatch_count = int(value.get("dispatch_count", 0))
            peers = value.get("peers", {})
            if not isinstance(peers, dict):
                return
            loaded = {}
            for node_id, entry in peers.items():
                if not _NODE_ID_RE.fullmatch(str(node_id)) or not isinstance(entry, dict):
                    continue
                first_seen = float(entry.get("first_seen_at", 0))
                last_selected = float(entry.get("last_selected_at", 0))
                selections = int(entry.get("selections", 0))
                if (
                    first_seen <= 0
                    or last_selected < 0
                    or not math.isfinite(first_seen)
                    or not math.isfinite(last_selected)
                    or not 0 <= selections <= MAX_COUNTER
                ):
                    continue
                loaded[node_id] = {
                    "first_seen_at": first_seen,
                    "last_selected_at": last_selected,
                    "selections": selections,
                }
            self.dispatch_count = max(0, min(dispatch_count, MAX_COUNTER))
            self._peers = self._bounded_peers(loaded)
            os.chmod(self.path, 0o600)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            self.dispatch_count = 0
            self._peers = {}

    def _checkpoint_if_due(self):
        if self.path is None or self._dirty == 0:
            return
        if self._dirty >= 32 or time.monotonic() - self._last_save >= 5.0:
            self._save()

    def _save(self):
        if self.path is None:
            self._dirty = 0
            return
        self._peers = self._bounded_peers(self._peers)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps({
            "version": 1,
            "dispatch_count": self.dispatch_count,
            "peers": self._peers,
        }, sort_keys=True, separators=(",", ":")))
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.path)
        self._dirty = 0
        self._last_save = time.monotonic()

    @staticmethod
    def _bounded_peers(peers: dict[str, dict]) -> dict[str, dict]:
        if len(peers) <= MAX_SCHEDULER_PEERS:
            return peers
        ordered = sorted(
            peers.items(),
            key=lambda item: (
                float(item[1].get("last_selected_at", 0)),
                float(item[1].get("first_seen_at", 0)),
            ),
            reverse=True,
        )
        return dict(ordered[:MAX_SCHEDULER_PEERS])
