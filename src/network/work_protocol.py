"""Typed distributed work over LSP v3.

Peers exchange declarative, content-bounded work units. Workers execute only
locally registered handlers; offers cannot contain executable code or shell
commands. ResourceGovernor consent and leases gate every execution.
"""

from __future__ import annotations

import hashlib
import base64
import json
import os
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Optional

from src.contribution import ResourceGovernor, ResourceRequirements


MAX_WORK_JSON_SIZE = 256 * 1024
MAX_WORK_LIFETIME_S = 60 * 60
MAX_COMPLETED_CACHE = 2048
_OPERATION_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,127}$")
_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_PAYLOAD_KEYS = {"code", "script", "command", "shell", "executable"}


class WorkProtocolError(ValueError):
    pass


def _canonical_json(value: dict) -> bytes:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise WorkProtocolError("Work value is not canonical JSON") from exc
    if len(encoded) > MAX_WORK_JSON_SIZE:
        raise WorkProtocolError("Work value exceeds 256 KiB")
    return encoded


def _validate_declarative_payload(payload: dict):
    """Bound nesting and reject executable-looking fields at every depth."""
    stack = [(payload, 0)]
    seen = 0
    while stack:
        value, depth = stack.pop()
        seen += 1
        if seen > 10_000 or depth > 16:
            raise WorkProtocolError("Work payload structure is too complex")
        if isinstance(value, dict):
            for key, child in value.items():
                if not isinstance(key, str):
                    raise WorkProtocolError("Work payload keys must be strings")
                if key.lower() in _FORBIDDEN_PAYLOAD_KEYS:
                    raise WorkProtocolError("Executable work payloads are forbidden")
                stack.append((child, depth + 1))
        elif isinstance(value, list):
            stack.extend((child, depth + 1) for child in value)


@dataclass(frozen=True)
class WorkUnit:
    job_id: str
    origin_node_id: str
    operation: str
    kind: str
    requirements: dict
    payload: dict
    created_at: float
    deadline: float
    nonce: str

    @classmethod
    def create(
        cls,
        *,
        origin_node_id: str,
        operation: str,
        kind: str,
        requirements: ResourceRequirements,
        payload: dict,
        timeout_s: float = 60.0,
    ) -> "WorkUnit":
        now = time.time()
        unsigned = {
            "origin_node_id": origin_node_id,
            "operation": operation,
            "kind": kind,
            "requirements": asdict(requirements),
            "payload": payload,
            "created_at": now,
            "deadline": now + min(max(float(timeout_s), 1.0), MAX_WORK_LIFETIME_S),
            "nonce": os.urandom(16).hex(),
        }
        cls._validate_unsigned(unsigned)
        job_id = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
        return cls(job_id=job_id, **unsigned)

    @classmethod
    def from_dict(cls, value: dict) -> "WorkUnit":
        if not isinstance(value, dict):
            raise WorkProtocolError("Work offer must be an object")
        required = {
            "job_id", "origin_node_id", "operation", "kind", "requirements",
            "payload", "created_at", "deadline", "nonce",
        }
        if set(value) != required:
            raise WorkProtocolError("Work offer fields do not match the schema")
        unsigned = {key: value[key] for key in required if key != "job_id"}
        cls._validate_unsigned(unsigned)
        expected = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
        if value["job_id"] != expected:
            raise WorkProtocolError("Work job ID does not match its content")
        return cls(**value)

    @staticmethod
    def _validate_unsigned(value: dict):
        if not _HEX_64_RE.fullmatch(str(value.get("origin_node_id", ""))):
            raise WorkProtocolError("Invalid work origin identity")
        if not _OPERATION_RE.fullmatch(str(value.get("operation", ""))):
            raise WorkProtocolError("Invalid work operation")
        requirements_value = value.get("requirements")
        if not isinstance(requirements_value, dict):
            raise WorkProtocolError("Work requirements must be an object")
        try:
            requirements = ResourceRequirements(**requirements_value)
            requirements.validate()
        except (TypeError, ValueError) as exc:
            raise WorkProtocolError("Invalid work requirements") from exc
        if value.get("kind") != requirements.kind:
            raise WorkProtocolError("Work kind and requirements disagree")
        payload = value.get("payload")
        if not isinstance(payload, dict):
            raise WorkProtocolError("Work payload must be an object")
        _validate_declarative_payload(payload)
        created_at = float(value.get("created_at", 0))
        deadline = float(value.get("deadline", 0))
        now = time.time()
        if created_at > now + 60 or deadline <= now or deadline - created_at > MAX_WORK_LIFETIME_S:
            raise WorkProtocolError("Work deadline is invalid or expired")
        nonce = str(value.get("nonce", ""))
        if len(nonce) != 32 or any(char not in "0123456789abcdef" for char in nonce):
            raise WorkProtocolError("Invalid work nonce")
        _canonical_json(value)

    def to_dict(self) -> dict:
        return asdict(self)

    def resource_requirements(self) -> ResourceRequirements:
        return ResourceRequirements(**self.requirements)


@dataclass(frozen=True)
class ResultReceipt:
    version: int
    job_id: str
    worker_node_id: str
    requester_node_id: str
    result_digest: str
    finished_at: float
    signature: str

    @staticmethod
    def _body(result: "WorkResult", worker_node_id: str, requester_node_id: str) -> dict:
        result_body = {
            "job_id": result.job_id,
            "status": result.status,
            "output": result.output,
            "error": result.error,
            "finished_at": result.finished_at,
        }
        return {
            "version": 1,
            "job_id": result.job_id,
            "worker_node_id": worker_node_id,
            "requester_node_id": requester_node_id,
            "result_digest": hashlib.sha256(_canonical_json(result_body)).hexdigest(),
            "finished_at": result.finished_at,
        }

    @classmethod
    def create(cls, result: "WorkResult", identity, requester_node_id: str):
        body = cls._body(
            result, identity.node_id_hex, requester_node_id
        )
        signature = identity.sign(
            b"LixySwarm work result receipt v1\x00" + _canonical_json(body)
        )
        return cls(
            **body,
            signature=base64.b64encode(signature).decode("ascii"),
        )

    @classmethod
    def from_dict(cls, value: dict) -> "ResultReceipt":
        if not isinstance(value, dict) or set(value) != {
            "version", "job_id", "worker_node_id", "requester_node_id",
            "result_digest", "finished_at", "signature",
        }:
            raise WorkProtocolError("Result receipt fields do not match the schema")
        try:
            receipt = cls(**value)
        except TypeError as exc:
            raise WorkProtocolError("Invalid result receipt") from exc
        for identifier in (
            receipt.job_id, receipt.worker_node_id,
            receipt.requester_node_id, receipt.result_digest,
        ):
            if not _HEX_64_RE.fullmatch(str(identifier)):
                raise WorkProtocolError("Invalid result receipt identifier")
        if receipt.version != 1:
            raise WorkProtocolError("Unsupported result receipt version")
        try:
            signature = base64.b64decode(receipt.signature, validate=True)
        except (TypeError, ValueError) as exc:
            raise WorkProtocolError("Invalid result receipt signature encoding") from exc
        if len(signature) != 64:
            raise WorkProtocolError("Invalid result receipt signature length")
        return receipt

    def verify(
        self,
        result: "WorkResult",
        *,
        expected_worker_id: str,
        expected_requester_id: str,
    ) -> bool:
        if (
            self.worker_node_id != expected_worker_id
            or self.requester_node_id != expected_requester_id
        ):
            return False
        expected = self._body(result, expected_worker_id, expected_requester_id)
        body = {
            key: getattr(self, key) for key in expected
        }
        if body != expected:
            return False
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            from cryptography.exceptions import InvalidSignature
            Ed25519PublicKey.from_public_bytes(
                bytes.fromhex(expected_worker_id)
            ).verify(
                base64.b64decode(self.signature, validate=True),
                b"LixySwarm work result receipt v1\x00" + _canonical_json(body),
            )
            return True
        except (ValueError, InvalidSignature):
            return False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class WorkResult:
    job_id: str
    status: str
    output: dict = field(default_factory=dict)
    error: str = ""
    finished_at: float = field(default_factory=time.time)
    receipt: dict = field(default_factory=dict)

    def validate(self):
        if not _HEX_64_RE.fullmatch(self.job_id):
            raise WorkProtocolError("Invalid result job ID")
        if self.status not in {"ok", "rejected", "error"}:
            raise WorkProtocolError("Invalid work result status")
        if (
            not isinstance(self.output, dict)
            or not isinstance(self.error, str)
            or not isinstance(self.receipt, dict)
        ):
            raise WorkProtocolError("Invalid work result body")
        if self.receipt:
            ResultReceipt.from_dict(self.receipt)
        _canonical_json(asdict(self))

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "WorkResult":
        if not isinstance(value, dict):
            raise WorkProtocolError("Work result must be an object")
        try:
            result = cls(**value)
        except TypeError as exc:
            raise WorkProtocolError("Work result fields do not match the schema") from exc
        result.validate()
        return result


@dataclass
class _PendingWork:
    expected_peer: str
    event: threading.Event = field(default_factory=threading.Event)
    result: Optional[WorkResult] = None


class WorkCoordinator:
    """Submit and execute allowlisted inference/training operations."""

    def __init__(
        self,
        node,
        governor: ResourceGovernor,
        *,
        max_workers: int = 2,
    ):
        self.node = node
        self.governor = governor
        self._handlers: dict[str, tuple[str, Callable]] = {}
        self._pending: dict[str, _PendingWork] = {}
        self._inflight: set[str] = set()
        self._completed: OrderedDict[str, WorkResult] = OrderedDict()
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, min(int(max_workers), 64)),
            thread_name_prefix="lixy-work",
        )
        node.on_work_offer_received(self._on_offer)
        node.on_work_result_received(self._on_result)

    def register_handler(self, operation: str, kind: str, handler: Callable):
        if not _OPERATION_RE.fullmatch(operation):
            raise ValueError("Invalid work operation")
        if kind not in {"inference", "training", "memory", "artifact"}:
            raise ValueError("Invalid handler kind")
        if not callable(handler):
            raise TypeError("Work handler must be callable")
        self._handlers[operation] = (kind, handler)

    def submit(
        self,
        operation: str,
        payload: dict,
        requirements: ResourceRequirements,
        *,
        peer_id: Optional[str] = None,
        timeout_s: float = 60.0,
    ) -> WorkResult:
        if not isinstance(payload, dict):
            raise WorkProtocolError("Work payload must be an object")
        requirements.validate()
        required_model_id = str(payload.get("model_artifact_id", "")) or None
        selected = peer_id or self._select_peer(
            requirements, required_model_id=required_model_id
        )
        if not selected:
            raise RuntimeError(f"No peer advertises {requirements.kind} capacity")
        work = WorkUnit.create(
            origin_node_id=self.node.identity.node_id_hex,
            operation=operation,
            kind=requirements.kind,
            requirements=requirements,
            payload=payload,
            timeout_s=timeout_s,
        )
        pending = _PendingWork(expected_peer=selected)
        with self._lock:
            self._pending[work.job_id] = pending
        if not self.node.send_work_offer(selected, work.to_dict()):
            with self._lock:
                self._pending.pop(work.job_id, None)
            raise ConnectionError("Failed to send work offer")
        if not pending.event.wait(timeout=max(1.0, timeout_s)):
            with self._lock:
                self._pending.pop(work.job_id, None)
            return WorkResult(job_id=work.job_id, status="error", error="work_timeout")
        return pending.result or WorkResult(
            job_id=work.job_id, status="error", error="missing_work_result"
        )

    def close(self):
        self._executor.shutdown(wait=True, cancel_futures=True)

    def select_peer(
        self,
        requirements: ResourceRequirements,
        *,
        required_model_id: Optional[str] = None,
    ) -> Optional[str]:
        requirements.validate()
        return self._select_peer(
            requirements, required_model_id=required_model_id
        )

    def select_peers(
        self,
        requirements: ResourceRequirements,
        *,
        required_model_id: Optional[str] = None,
        limit: int = 3,
    ) -> list[str]:
        requirements.validate()
        return [
            str(peer["node_id"])
            for peer in self._eligible_peers(
                requirements, required_model_id=required_model_id
            )[:max(0, int(limit))]
        ]

    def _select_peer(
        self,
        requirements: ResourceRequirements,
        *,
        required_model_id: Optional[str] = None,
    ) -> Optional[str]:
        candidates = self._eligible_peers(
            requirements, required_model_id=required_model_id
        )
        return str(candidates[0]["node_id"]) if candidates else None

    def _eligible_peers(
        self,
        requirements: ResourceRequirements,
        *,
        required_model_id: Optional[str] = None,
    ) -> list[dict]:
        candidates = []
        for peer in self.node.peers():
            resources = peer.get("resources", {})
            if not isinstance(resources, dict):
                continue
            work = resources.get("work", {}) if isinstance(resources, dict) else {}
            if not isinstance(work, dict) or work.get(requirements.kind) is not True:
                continue
            try:
                has_gpu = resources.get("has_gpu", False) is True
                ram_gb = float(resources.get("ram_gb", 0.0))
                disk_gb = float(resources.get("disk_gb", 0.0))
                cpu_cores = int(resources.get("cpu_cores", 0))
                gpu_vram = float(resources.get("gpu_vram_gb", 0.0))
            except (TypeError, ValueError, OverflowError):
                continue
            if not all(map(lambda value: value >= 0, (ram_gb, disk_gb, cpu_cores, gpu_vram))):
                continue
            if requirements.gpu_required and not has_gpu:
                continue
            if ram_gb < requirements.ram_gb:
                continue
            if disk_gb < requirements.disk_gb:
                continue
            if cpu_cores < requirements.cpu_slots:
                continue
            if required_model_id is not None:
                models = resources.get("models", [])
                if not isinstance(models, list) or required_model_id not in models:
                    continue
            candidates.append(peer)
        candidates.sort(
            key=lambda peer: (
                float(peer.get("resources", {}).get("gpu_vram_gb", 0.0)),
                int(peer.get("resources", {}).get("cpu_cores", 1)),
            ),
            reverse=True,
        )
        return candidates

    def _on_offer(self, value: dict, from_node_id: str):
        self._executor.submit(self._execute_offer, value, from_node_id)

    def _execute_offer(self, value: dict, from_node_id: str):
        job_id = str(value.get("job_id", "")) if isinstance(value, dict) else ""
        try:
            work = WorkUnit.from_dict(value)
            job_id = work.job_id
            if work.origin_node_id != from_node_id:
                raise WorkProtocolError("Work origin does not match the signed peer")
        except WorkProtocolError as exc:
            if _HEX_64_RE.fullmatch(job_id):
                self._send_result(from_node_id, WorkResult(
                    job_id=job_id, status="rejected", error=str(exc)[:160]
                ))
            return

        with self._lock:
            cached = self._completed.get(work.job_id)
            if cached:
                self.node.send_work_result(from_node_id, cached.to_dict())
                return
            if work.job_id in self._inflight:
                result = WorkResult(
                    job_id=work.job_id, status="rejected", error="already_running"
                )
                self._send_result(from_node_id, result)
                return
            self._inflight.add(work.job_id)

        handler_entry = self._handlers.get(work.operation)
        if not handler_entry or handler_entry[0] != work.kind:
            result = WorkResult(
                job_id=work.job_id, status="rejected", error="operation_not_allowed"
            )
            self._finish(work.job_id, from_node_id, result)
            return

        lease, reason = self.governor.acquire(work.resource_requirements())
        if lease is None:
            result = WorkResult(job_id=work.job_id, status="rejected", error=reason)
            self._finish(work.job_id, from_node_id, result)
            return

        try:
            with lease:
                output = handler_entry[1](dict(work.payload), work)
            if not isinstance(output, dict):
                raise TypeError("handler output must be a JSON object")
            result = WorkResult(job_id=work.job_id, status="ok", output=output)
            result.validate()
        except Exception as exc:
            result = WorkResult(
                job_id=work.job_id,
                status="error",
                error=f"handler_failed:{type(exc).__name__}",
            )
        self._finish(work.job_id, from_node_id, result)

    def _finish(self, job_id: str, peer_id: str, result: WorkResult):
        result = self._signed_result(result, peer_id)
        with self._lock:
            self._inflight.discard(job_id)
            self._completed[job_id] = result
            self._completed.move_to_end(job_id)
            while len(self._completed) > MAX_COMPLETED_CACHE:
                self._completed.popitem(last=False)
        try:
            self.node.send_work_result(peer_id, result.to_dict())
        except Exception:
            pass

    def _signed_result(self, result: WorkResult, requester_id: str) -> WorkResult:
        receipt = ResultReceipt.create(
            result, self.node.identity, requester_id
        )
        return replace(result, receipt=receipt.to_dict())

    def _send_result(self, peer_id: str, result: WorkResult):
        signed = self._signed_result(result, peer_id)
        try:
            self.node.send_work_result(peer_id, signed.to_dict())
        except Exception:
            pass

    def _on_result(self, value: dict, from_node_id: str):
        try:
            result = WorkResult.from_dict(value)
        except WorkProtocolError:
            return
        try:
            receipt = ResultReceipt.from_dict(result.receipt)
        except WorkProtocolError:
            return
        if not receipt.verify(
            result,
            expected_worker_id=from_node_id,
            expected_requester_id=self.node.identity.node_id_hex,
        ):
            return
        with self._lock:
            pending = self._pending.get(result.job_id)
            if pending is None or pending.expected_peer != from_node_id:
                return
            pending.result = result
            self._pending.pop(result.job_id, None)
            pending.event.set()
