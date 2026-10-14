"""Signed public node status, bounded and free of addresses or personal memory.

Signatures authenticate the publisher, not its claimed hardware or work.
"""
from __future__ import annotations

import base64
import json
import math
import os
import re
import secrets
import tempfile
import threading
import time
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

MAX_STATUS_BYTES = 16 * 1024
FRESH_SECONDS = 60
MAX_STATUS_SKEW_MS = 60 * 1000
MAX_STORED_NODES = 1024
NODE_ID = re.compile(r"^[0-9a-f]{64}$")
PROCESS_INSTANCE_ID = secrets.token_hex(16)
STATES = {"connecting", "connected", "preparing", "available", "working", "paused", "disconnected"}
ROLES = {"participant", "seed", "relay"}


def canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
        # Windows readers and scanners may temporarily deny replacement.
        # Preserve the previous complete record and retry the atomic rename.
        deadline = time.monotonic() + 1.0
        while True:
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.02)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _number(value, minimum, maximum, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Invalid telemetry number")
    if not minimum <= value <= maximum or not math.isfinite(value):
        raise ValueError("Telemetry number out of range")
    if integer and not isinstance(value, int):
        raise ValueError("Telemetry integer required")


def validate_status(status: dict) -> None:
    fields={
        "schema", "node_id", "sequence", "timestamp_ms", "role", "state",
        "uptime_s", "resources", "jobs", "links",
    }
    if not isinstance(status,dict) or set(status) not in (fields,fields|{'process_instance_id'}):
        raise ValueError("Invalid public telemetry schema")
    if 'process_instance_id' in status and (not isinstance(status['process_instance_id'],str)
            or not re.fullmatch(r'[0-9a-f]{32}',status['process_instance_id'])):
        raise ValueError('Invalid telemetry process instance')
    if status["schema"] != 1 or isinstance(status["schema"], bool):
        raise ValueError("Unsupported telemetry schema")
    if not isinstance(status["node_id"], str) or not NODE_ID.fullmatch(status["node_id"]):
        raise ValueError("Invalid telemetry identity")
    if not isinstance(status["role"], str) or not isinstance(status["state"], str) or status["role"] not in ROLES or status["state"] not in STATES:
        raise ValueError("Invalid telemetry role or state")
    _number(status["sequence"], 1, 2**63 - 1, integer=True)
    _number(status["timestamp_ms"], 1, 2**63 - 1, integer=True)
    _number(status["uptime_s"], 0, 10**10)
    resources = status["resources"]
    if not isinstance(resources, dict) or set(resources) != {"cpu_cores", "ram_gb", "gpu_vram_gb", "has_gpu"}:
        raise ValueError("Invalid telemetry resources")
    _number(resources["cpu_cores"], 0, 65536, integer=True)
    for name in ("ram_gb", "gpu_vram_gb"):
        _number(resources[name], 0, 10**6)
    if not isinstance(resources["has_gpu"], bool):
        raise ValueError("Invalid telemetry GPU flag")
    jobs = status["jobs"]
    if not isinstance(jobs, dict) or set(jobs) != {"active", "completed"}:
        raise ValueError("Invalid telemetry work counters")
    for value in jobs.values():
        _number(value, 0, 10**12, integer=True)
    links = status["links"]
    if not isinstance(links, list) or len(links) > 64:
        raise ValueError("Too many telemetry links")
    seen = set()
    for link in links:
        if not isinstance(link, dict) or set(link) != {"node_id", "transport"}:
            raise ValueError("Invalid telemetry link")
        peer_id = link["node_id"]
        if not isinstance(peer_id, str) or not NODE_ID.fullmatch(peer_id) or peer_id in seen:
            raise ValueError("Invalid telemetry peer identity")
        if peer_id == status["node_id"] or not isinstance(link["transport"], str) or link["transport"] not in {"direct", "relay"}:
            raise ValueError("Invalid telemetry transport")
        seen.add(peer_id)


def sign_status(identity, status: dict) -> dict:
    validate_status(status)
    if status["node_id"] != identity.node_id_hex:
        raise ValueError("Telemetry signer identity mismatch")
    return {"status": status, "signature": base64.b64encode(identity.sign(canonical(status))).decode("ascii")}


def verify_status(envelope: dict, *, sender_id=None, now_ms=None) -> dict:
    if not isinstance(envelope, dict) or set(envelope) != {"status", "signature"}:
        raise ValueError("Invalid telemetry envelope")
    if len(canonical(envelope)) > MAX_STATUS_BYTES:
        raise ValueError("Telemetry exceeds maximum size")
    status = envelope["status"]
    validate_status(status)
    if sender_id is not None and sender_id != status["node_id"]:
        raise ValueError("Telemetry sender identity mismatch")
    if now_ms is not None and abs(now_ms - status["timestamp_ms"]) > MAX_STATUS_SKEW_MS:
        raise ValueError("Expired telemetry timestamp")
    try:
        signature = base64.b64decode(envelope["signature"], validate=True)
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(status["node_id"])).verify(signature, canonical(status))
    except (InvalidSignature, TypeError, ValueError) as exc:
        raise ValueError("Invalid telemetry signature") from exc
    return status


def build_status(node, *, sequence: int, role: str, state: str, uptime_s: float, resources: dict) -> dict:
    return {
        "schema": 1, "node_id": node.identity.node_id_hex,
        "sequence": sequence, "timestamp_ms": int(time.time() * 1000),
        "role": role, "state": state, "uptime_s": round(uptime_s, 1),
        'process_instance_id':PROCESS_INSTANCE_ID,
        "resources": resources, "jobs": {"active": 0, "completed": 0},
        "links": [{"node_id": peer["node_id"], "transport": peer["transport"]} for peer in node.peers()[:64]],
    }


class NodeStatusStore:
    """Single collector, atomic files per identity, bounded disk usage."""

    def __init__(self, directory: str | Path, *, max_nodes=MAX_STORED_NODES):
        self.directory = Path(directory)
        self.max_nodes = max_nodes
        self._lock = threading.RLock()

    def _read(self, node_id):
        try:
            path = self.directory / f"{node_id}.json"
            if path.stat().st_size > MAX_STATUS_BYTES + 1024:
                return {}
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return {}

    def sequence(self, node_id):
        if not NODE_ID.fullmatch(node_id):
            raise ValueError("Invalid telemetry identity")
        return self._read(node_id).get("envelope", {}).get("status", {}).get("sequence", 0)

    def save(self, envelope: dict, sender_id: str, *, now=None, local=False) -> dict:
        received = time.time() if now is None else now
        status = verify_status(envelope, sender_id=sender_id, now_ms=int(received * 1000))
        with self._lock:
            previous = self._read(sender_id)
            if status["sequence"] <= previous.get("envelope", {}).get("status", {}).get("sequence", 0):
                raise ValueError("Replayed telemetry sequence")
            if not local and previous.get("connected_observed") and received - previous["received_at"] < 2:
                raise ValueError("Telemetry publication is too frequent")
            self.directory.mkdir(parents=True, exist_ok=True)
            if not previous:
                paths = list(self.directory.glob("*.json"))
                if len(paths) >= self.max_nodes:
                    expired = [p for p in paths if received - p.stat().st_mtime > 86400]
                    if not expired:
                        raise ValueError("Telemetry store capacity reached")
                    min(expired, key=lambda p: p.stat().st_mtime).unlink()
            record = {"envelope": envelope, "received_at": received, "connected_observed": True, "local": local}
            atomic_json(self.directory / f"{sender_id}.json", record)
            return record

    def disconnected(self, node_id):
        if not NODE_ID.fullmatch(node_id):
            raise ValueError("Invalid telemetry identity")
        with self._lock:
            record = self._read(node_id)
            if record:
                record["connected_observed"] = False
                atomic_json(self.directory / f"{node_id}.json", record)

    def reset_connections(self):
        for path in self.directory.glob("*.json"):
            if NODE_ID.fullmatch(path.stem):
                self.disconnected(path.stem)

    def nodes(self, *, now=None) -> list[dict]:
        current = time.time() if now is None else now
        nodes = []
        for path in sorted(self.directory.glob("*.json"))[:self.max_nodes]:
            if not NODE_ID.fullmatch(path.stem):
                continue
            record = self._read(path.stem)
            try:
                status = verify_status(record["envelope"], sender_id=path.stem)
                age = max(0, current - record["received_at"])
                active = age <= FRESH_SECONDS and record["connected_observed"] and status["state"] != "disconnected"
                nodes.append({
                    **status, "state": status["state"] if active else "disconnected",
                    "active": bool(active), "stale": age > FRESH_SECONDS, "age_s": round(age, 1),
                    "signature_verified": True, "capacity_verified": False,
                    "work_counters_verified": False, "connected_observed": bool(record["connected_observed"]),
                    "self": bool(record.get("local", False)),
                })
            except (KeyError, TypeError, ValueError):
                continue
        return nodes
