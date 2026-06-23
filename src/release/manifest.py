"""Threshold-signed, locally trusted model release manifests."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path


MAX_RELEASE_JSON_BYTES = 64 * 1024
MAX_RELEASE_SIGNERS = 16
VALID_MODEL_FORMATS = {"pytorch-weights-only-v1", "safetensors-v1"}
_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")


class ReleaseError(ValueError):
    pass


def _canonical(value: dict) -> bytes:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReleaseError("release data is not canonical JSON") from exc
    if len(encoded) > MAX_RELEASE_JSON_BYTES:
        raise ReleaseError("release data exceeds 64 KiB")
    return encoded


@dataclass(frozen=True)
class ReleaseManifest:
    version: int
    release_id: str
    network_id: str
    sequence: int
    previous_release_id: str
    model_artifact_id: str
    model_format: str
    config_artifact_id: str
    tokenizer_artifact_id: str
    created_at: float
    minimum_client_version: str
    signatures: list[dict] = field(default_factory=list)

    def body(self) -> dict:
        return {
            "version": self.version,
            "network_id": self.network_id,
            "sequence": self.sequence,
            "previous_release_id": self.previous_release_id,
            "model_artifact_id": self.model_artifact_id,
            "model_format": self.model_format,
            "config_artifact_id": self.config_artifact_id,
            "tokenizer_artifact_id": self.tokenizer_artifact_id,
            "created_at": self.created_at,
            "minimum_client_version": self.minimum_client_version,
        }

    @classmethod
    def create(
        cls,
        *,
        model_artifact_id: str,
        model_format: str,
        sequence: int,
        previous_release_id: str = "",
        config_artifact_id: str = "",
        tokenizer_artifact_id: str = "",
        network_id: str = "LIXYMAIN",
        minimum_client_version: str = "0.3.0",
        created_at: float | None = None,
    ) -> "ReleaseManifest":
        body = {
            "version": 1,
            "network_id": network_id,
            "sequence": int(sequence),
            "previous_release_id": previous_release_id,
            "model_artifact_id": model_artifact_id,
            "model_format": model_format,
            "config_artifact_id": config_artifact_id,
            "tokenizer_artifact_id": tokenizer_artifact_id,
            "created_at": time.time() if created_at is None else float(created_at),
            "minimum_client_version": minimum_client_version,
        }
        release_id = hashlib.sha256(_canonical(body)).hexdigest()
        manifest = cls(release_id=release_id, signatures=[], **body)
        manifest.validate()
        return manifest

    def validate(self):
        if self.version != 1 or self.network_id != "LIXYMAIN":
            raise ReleaseError("unsupported release manifest version or network")
        if not _HEX_64_RE.fullmatch(self.release_id):
            raise ReleaseError("invalid release ID")
        if hashlib.sha256(_canonical(self.body())).hexdigest() != self.release_id:
            raise ReleaseError("release ID does not match manifest content")
        if self.sequence < 0 or self.sequence > 2 ** 63 - 1:
            raise ReleaseError("release sequence is out of range")
        if self.sequence == 0 and self.previous_release_id:
            raise ReleaseError("genesis release cannot have a predecessor")
        if self.sequence > 0 and not _HEX_64_RE.fullmatch(self.previous_release_id):
            raise ReleaseError("non-genesis release requires a predecessor")
        for artifact_id in (
            self.model_artifact_id,
            self.config_artifact_id,
            self.tokenizer_artifact_id,
        ):
            if artifact_id and not _HEX_64_RE.fullmatch(artifact_id):
                raise ReleaseError("invalid release artifact ID")
        if not self.model_artifact_id:
            raise ReleaseError("release requires a model artifact")
        if self.model_format not in VALID_MODEL_FORMATS:
            raise ReleaseError("unsafe or unsupported model format")
        if not 0 < self.created_at <= time.time() + 300:
            raise ReleaseError("release creation time is invalid")
        if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[a-z0-9.-]+)?", self.minimum_client_version):
            raise ReleaseError("minimum client version is invalid")
        if not isinstance(self.signatures, list) or len(self.signatures) > MAX_RELEASE_SIGNERS:
            raise ReleaseError("release signature list is invalid")
        seen = set()
        for item in self.signatures:
            if not isinstance(item, dict) or set(item) != {"signer_id", "signature"}:
                raise ReleaseError("release signature fields are invalid")
            signer_id = item["signer_id"]
            if not _HEX_64_RE.fullmatch(str(signer_id)) or signer_id in seen:
                raise ReleaseError("release signer identity is invalid or duplicated")
            seen.add(signer_id)
            try:
                signature = base64.b64decode(item["signature"], validate=True)
            except (TypeError, ValueError) as exc:
                raise ReleaseError("release signature encoding is invalid") from exc
            if len(signature) != 64:
                raise ReleaseError("release signature length is invalid")
        _canonical(self.to_dict(validate=False))

    def sign(self, identity) -> "ReleaseManifest":
        self.validate()
        message = b"LixySwarm release manifest v1\x00" + _canonical(self.body())
        item = {
            "signer_id": identity.node_id_hex,
            "signature": base64.b64encode(identity.sign(message)).decode("ascii"),
        }
        signatures = [
            current for current in self.signatures
            if current["signer_id"] != identity.node_id_hex
        ] + [item]
        signatures.sort(key=lambda current: current["signer_id"])
        signed = replace(self, signatures=signatures)
        signed.validate()
        return signed

    def valid_signers(self) -> set[str]:
        self.validate()
        message = b"LixySwarm release manifest v1\x00" + _canonical(self.body())
        valid = set()
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from cryptography.exceptions import InvalidSignature
        for item in self.signatures:
            try:
                Ed25519PublicKey.from_public_bytes(
                    bytes.fromhex(item["signer_id"])
                ).verify(base64.b64decode(item["signature"]), message)
                valid.add(item["signer_id"])
            except (ValueError, InvalidSignature):
                continue
        return valid

    def to_dict(self, *, validate: bool = True) -> dict:
        if validate:
            self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "ReleaseManifest":
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ReleaseError("release manifest fields do not match the schema")
        try:
            manifest = cls(**value)
        except TypeError as exc:
            raise ReleaseError("invalid release manifest") from exc
        manifest.validate()
        return manifest

    @classmethod
    def load(cls, path: str | Path) -> "ReleaseManifest":
        try:
            return cls.from_dict(json.loads(Path(path).read_text()))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReleaseError("cannot read release manifest") from exc

    def save(self, path: str | Path):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        os.replace(temporary, destination)


@dataclass(frozen=True)
class TrustPolicy:
    version: int = 1
    threshold: int = 1
    trusted_signers: tuple[str, ...] = ()
    pinned_genesis_release_id: str = ""
    revoked_release_ids: tuple[str, ...] = ()

    def validate(self):
        if self.version != 1:
            raise ReleaseError("unsupported trust policy version")
        if not 1 <= self.threshold <= len(self.trusted_signers) <= MAX_RELEASE_SIGNERS:
            raise ReleaseError("trust threshold is invalid")
        if len(set(self.trusted_signers)) != len(self.trusted_signers):
            raise ReleaseError("trusted signer is duplicated")
        for value in (*self.trusted_signers, *self.revoked_release_ids):
            if not _HEX_64_RE.fullmatch(value):
                raise ReleaseError("trust policy contains an invalid identifier")
        if self.pinned_genesis_release_id and not _HEX_64_RE.fullmatch(
            self.pinned_genesis_release_id
        ):
            raise ReleaseError("pinned genesis release ID is invalid")

    def authorize(self, manifest: ReleaseManifest) -> set[str]:
        self.validate()
        manifest.validate()
        if manifest.release_id in self.revoked_release_ids:
            raise ReleaseError("release is locally revoked")
        if (
            manifest.sequence == 0
            and self.pinned_genesis_release_id
            and manifest.release_id != self.pinned_genesis_release_id
        ):
            raise ReleaseError("release does not match the pinned genesis")
        valid = manifest.valid_signers().intersection(self.trusted_signers)
        if len(valid) < self.threshold:
            raise ReleaseError("release does not meet the local signature threshold")
        return valid

    def save(self, path: str | Path):
        self.validate()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)

    @classmethod
    def load(cls, path: str | Path) -> "TrustPolicy":
        try:
            value = json.loads(Path(path).read_text())
            policy = cls(
                version=value.get("version", 1),
                threshold=value["threshold"],
                trusted_signers=tuple(value["trusted_signers"]),
                pinned_genesis_release_id=value.get("pinned_genesis_release_id", ""),
                revoked_release_ids=tuple(value.get("revoked_release_ids", [])),
            )
            policy.validate()
            return policy
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ReleaseError("cannot read trust policy") from exc


class ReleaseRegistry:
    """Local accepted-release registry with explicit activation and rollback."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifests = self.root / "manifests"
        self.active_path = self.root / "active.json"
        self.audit_path = self.root / "audit.jsonl"
        self.manifests.mkdir(parents=True, exist_ok=True)

    def accept(self, manifest: ReleaseManifest, policy: TrustPolicy, artifact_store):
        signers = policy.authorize(manifest)
        self._verify_artifacts(manifest, artifact_store)
        destination = self.manifests / f"{manifest.release_id}.json"
        manifest.save(destination)
        self._audit("accepted", manifest.release_id, signers=sorted(signers))
        return destination

    def activate(self, release_id: str, policy: TrustPolicy, artifact_store):
        manifest = ReleaseManifest.load(self.manifests / f"{release_id}.json")
        signers = policy.authorize(manifest)
        self._verify_artifacts(manifest, artifact_store)
        active = self.active()
        if active is None:
            if manifest.sequence != 0 or manifest.previous_release_id:
                raise ReleaseError("first active release must be genesis")
        else:
            if (
                manifest.sequence != active.sequence + 1
                or manifest.previous_release_id != active.release_id
            ):
                raise ReleaseError("release does not extend the active chain")
        temporary = self.active_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({
            "release_id": manifest.release_id,
            "activated_at": time.time(),
        }, sort_keys=True))
        os.replace(temporary, self.active_path)
        self._audit("activated", manifest.release_id, signers=sorted(signers))
        return manifest

    def rollback(
        self,
        release_id: str,
        policy: TrustPolicy,
        artifact_store,
        *,
        explicit_confirmation: bool,
    ):
        if not explicit_confirmation:
            raise ReleaseError("rollback requires explicit confirmation")
        target = ReleaseManifest.load(self.manifests / f"{release_id}.json")
        policy.authorize(target)
        self._verify_artifacts(target, artifact_store)
        current = self.active()
        if current is None or target.sequence >= current.sequence:
            raise ReleaseError("rollback target must be an older accepted release")
        cursor = current
        while cursor.sequence > target.sequence:
            if not cursor.previous_release_id:
                raise ReleaseError("rollback target is not in the active release chain")
            cursor = ReleaseManifest.load(
                self.manifests / f"{cursor.previous_release_id}.json"
            )
            policy.authorize(cursor)
        if cursor.release_id != target.release_id:
            raise ReleaseError("rollback target is not in the active release chain")
        temporary = self.active_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({
            "release_id": target.release_id,
            "activated_at": time.time(),
            "rollback_from": current.release_id,
        }, sort_keys=True))
        os.replace(temporary, self.active_path)
        self._audit("rollback", target.release_id, previous=current.release_id)
        return target

    def active(self) -> ReleaseManifest | None:
        if not self.active_path.exists():
            return None
        try:
            release_id = json.loads(self.active_path.read_text())["release_id"]
            return ReleaseManifest.load(self.manifests / f"{release_id}.json")
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise ReleaseError("active release registry is invalid") from exc

    def verify_active(self, policy: TrustPolicy, artifact_store) -> ReleaseManifest:
        manifest = self.active()
        if manifest is None:
            raise ReleaseError("no active release is configured")
        policy.authorize(manifest)
        self._verify_artifacts(manifest, artifact_store)
        return manifest

    @staticmethod
    def _verify_artifacts(manifest: ReleaseManifest, artifact_store):
        model = artifact_store.manifest(manifest.model_artifact_id)
        if model.kind != "model" or not artifact_store.has(model.artifact_id):
            raise ReleaseError("release model artifact is unavailable or invalid")
        for artifact_id in (
            manifest.config_artifact_id, manifest.tokenizer_artifact_id
        ):
            if artifact_id and not artifact_store.has(artifact_id):
                raise ReleaseError("release support artifact is unavailable")

    def _audit(self, event: str, release_id: str, **details):
        self.root.mkdir(parents=True, exist_ok=True)
        record = {
            "event": event,
            "release_id": release_id,
            "timestamp": time.time(),
            **details,
        }
        with self.audit_path.open("a") as destination:
            destination.write(json.dumps(record, sort_keys=True) + "\n")
