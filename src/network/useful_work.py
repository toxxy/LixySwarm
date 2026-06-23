"""Requester-signed credits for validated useful training contributions."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import threading
from dataclasses import asdict, dataclass
from pathlib import Path


def _canonical(value: dict) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


@dataclass(frozen=True)
class UsefulWorkCredit:
    version: int
    credit_id: str
    issuer_node_id: str
    worker_node_id: str
    job_id: str
    result_digest: str
    gradient_artifact_id: str
    aggregate_artifact_id: str
    model_artifact_id: str
    dataset_artifact_id: str
    token_count: int
    issued_at: float
    receipt: dict
    signature: str

    def body(self) -> dict:
        return {
            key: getattr(self, key) for key in (
                "version", "issuer_node_id", "worker_node_id", "job_id",
                "result_digest", "gradient_artifact_id", "aggregate_artifact_id",
                "model_artifact_id", "dataset_artifact_id", "token_count",
                "issued_at", "receipt",
            )
        }

    def claim(self) -> dict:
        """Stable contribution identity; aggregation retries cannot mint more credit."""
        return {
            key: getattr(self, key) for key in (
                "version", "issuer_node_id", "worker_node_id", "job_id",
                "result_digest", "gradient_artifact_id", "model_artifact_id",
                "dataset_artifact_id", "token_count",
            )
        }

    @classmethod
    def issue(
        cls, identity, *, worker_node_id: str, receipt: dict,
        gradient_artifact_id: str, aggregate_artifact_id: str,
        model_artifact_id: str, dataset_artifact_id: str, token_count: int,
    ):
        if receipt.get("worker_node_id") != worker_node_id:
            raise ValueError("credit receipt worker mismatch")
        if receipt.get("requester_node_id") != identity.node_id_hex:
            raise ValueError("credit issuer did not request the work")
        _verify_worker_receipt(receipt)
        body = {
            "version": 1,
            "issuer_node_id": identity.node_id_hex,
            "worker_node_id": worker_node_id,
            "job_id": receipt["job_id"],
            "result_digest": receipt["result_digest"],
            "gradient_artifact_id": gradient_artifact_id,
            "aggregate_artifact_id": aggregate_artifact_id,
            "model_artifact_id": model_artifact_id,
            "dataset_artifact_id": dataset_artifact_id,
            "token_count": int(token_count),
            "issued_at": float(receipt["finished_at"]),
            "receipt": dict(receipt),
        }
        credit_id = hashlib.sha256(_canonical({
            key: body[key] for key in (
                "version", "issuer_node_id", "worker_node_id", "job_id",
                "result_digest", "gradient_artifact_id", "model_artifact_id",
                "dataset_artifact_id", "token_count",
            )
        })).hexdigest()
        signature = identity.sign(
            b"LixySwarm useful training credit v1\x00" + _canonical(body)
        )
        return cls(
            credit_id=credit_id,
            signature=base64.b64encode(signature).decode(),
            **body,
        )

    def verify(self) -> bool:
        hex_fields = (
            self.credit_id, self.issuer_node_id, self.worker_node_id,
            self.job_id, self.result_digest, self.gradient_artifact_id,
            self.aggregate_artifact_id, self.model_artifact_id,
            self.dataset_artifact_id,
        )
        if self.version != 1 or any(
            len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
            for value in hex_fields
        ):
            return False
        if not 1 <= self.token_count <= 4096 or not 0 < self.issued_at <= time.time() + 300:
            return False
        body = self.body()
        if hashlib.sha256(_canonical(self.claim())).hexdigest() != self.credit_id:
            return False
        try:
            receipt = _verify_worker_receipt(self.receipt)
            if (
                receipt["job_id"] != self.job_id
                or receipt["worker_node_id"] != self.worker_node_id
                or receipt["requester_node_id"] != self.issuer_node_id
                or receipt["result_digest"] != self.result_digest
                or float(receipt["finished_at"]) != self.issued_at
            ):
                return False
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(self.issuer_node_id)).verify(
                base64.b64decode(self.signature, validate=True),
                b"LixySwarm useful training credit v1\x00" + _canonical(body),
            )
            return True
        except Exception:
            return False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict):
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("invalid useful-work credit schema")
        credit = cls(**value)
        if not credit.verify():
            raise ValueError("invalid useful-work credit")
        return credit


def _verify_worker_receipt(value: dict) -> dict:
    """Validate the portable worker signature without needing the full result."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    required = {
        "version", "job_id", "worker_node_id", "requester_node_id",
        "result_digest", "finished_at", "signature",
    }
    if not isinstance(value, dict) or set(value) != required or value["version"] != 1:
        raise ValueError("invalid worker receipt schema")
    for key in ("job_id", "worker_node_id", "requester_node_id", "result_digest"):
        field = value[key]
        if not isinstance(field, str) or len(field) != 64 or any(
            c not in "0123456789abcdef" for c in field
        ):
            raise ValueError("invalid worker receipt identifier")
    finished_at = float(value["finished_at"])
    if not 0 < finished_at <= time.time() + 300:
        raise ValueError("invalid worker receipt time")
    body = {key: value[key] for key in (
        "version", "job_id", "worker_node_id", "requester_node_id",
        "result_digest", "finished_at",
    )}
    signature = base64.b64decode(value["signature"], validate=True)
    Ed25519PublicKey.from_public_bytes(bytes.fromhex(value["worker_node_id"])).verify(
        signature,
        b"LixySwarm work result receipt v1\x00" + _canonical(body),
    )
    return body


class UsefulWorkLedger:
    def __init__(self, path: str | Path, owner_node_id: str):
        self.path = Path(path)
        self.owner_node_id = owner_node_id
        self._credits: dict[str, UsefulWorkCredit] = {}
        self._lock = threading.RLock()
        self._load()

    def add(self, value: dict) -> bool:
        credit = UsefulWorkCredit.from_dict(value)
        if credit.worker_node_id != self.owner_node_id:
            return False
        with self._lock:
            if credit.credit_id in self._credits:
                return True
            self._credits[credit.credit_id] = credit
            self._save()
        return True

    def summary(self) -> dict:
        with self._lock:
            return {
                "validated_training_credits": len(self._credits),
                "distinct_issuers": len({
                    c.issuer_node_id for c in self._credits.values()
                }),
                "validated_tokens": sum(c.token_count for c in self._credits.values()),
            }

    def _load(self):
        if not self.path.exists():
            return
        try:
            values = json.loads(self.path.read_text()).get("credits", [])
        except (OSError, json.JSONDecodeError, AttributeError):
            return
        for value in values if isinstance(values, list) else []:
            try:
                credit = UsefulWorkCredit.from_dict(value)
                if credit.worker_node_id == self.owner_node_id:
                    self._credits[credit.credit_id] = credit
            except (TypeError, ValueError):
                continue

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps({
            "version": 1,
            "credits": [c.to_dict() for c in self._credits.values()],
        }, sort_keys=True, separators=(",", ":")))
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.path)
