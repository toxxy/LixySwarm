"""Content-addressed artifact storage and verified LSP v3 transfer."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor,TimeoutError as FutureTimeout
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
from weakref import WeakValueDictionary

from src.contribution import ResourceRequirements
from .artifact_codec import CHUNK_BYTES as ARTIFACT_CHUNK_BYTES, ENCODING, encode_chunk, decode_chunk


DEFAULT_MAX_ARTIFACT_BYTES = 10 * 1024 ** 3
DEFAULT_MAX_STORE_BYTES = 100 * 1024 ** 3
VALID_ARTIFACT_KINDS = {"model", "dataset", "gradient", "evaluation", "other"}
_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
_MEDIA_TYPE_RE = re.compile(r"^[a-z0-9][a-z0-9.+-]{0,63}/[a-z0-9][a-z0-9.+-]{0,63}$")


class ArtifactError(ValueError):
    pass


class _TransferLock:
    def __init__(self):self.lock=threading.Lock()


class _StoreTransfers:
    def __init__(self):
        self.guard=threading.Lock();self.entries=WeakValueDictionary()

    def entry(self,identifier):
        with self.guard:
            value=self.entries.get(identifier)
            if value is None:value=_TransferLock();self.entries[identifier]=value
            return value


_TRANSFER_STORES=WeakValueDictionary()
_TRANSFER_STORES_GUARD=threading.Lock()


def digest_file(path: str | Path) -> tuple[str, int]:
    path = Path(path)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


@dataclass(frozen=True)
class ArtifactManifest:
    artifact_id: str
    size: int
    kind: str
    media_type: str
    created_at: float

    def validate(self, *, max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES):
        if not _HEX_64_RE.fullmatch(self.artifact_id):
            raise ArtifactError("invalid artifact ID")
        if not 0 <= int(self.size) <= int(max_bytes):
            raise ArtifactError("artifact size is out of range")
        if self.kind not in VALID_ARTIFACT_KINDS:
            raise ArtifactError("invalid artifact kind")
        if not _MEDIA_TYPE_RE.fullmatch(self.media_type):
            raise ArtifactError("invalid artifact media type")
        if not 0 < float(self.created_at) <= time.time() + 60:
            raise ArtifactError("invalid artifact creation time")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict, *, max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES):
        if not isinstance(value, dict) or set(value) != {
            "artifact_id", "size", "kind", "media_type", "created_at"
        }:
            raise ArtifactError("artifact manifest fields do not match the schema")
        try:
            manifest = cls(**value)
        except TypeError as exc:
            raise ArtifactError("invalid artifact manifest") from exc
        manifest.validate(max_bytes=max_bytes)
        return manifest


class ArtifactStore:
    """A bounded store whose public identifiers never reveal source paths."""

    def __init__(
        self,
        root: str | Path,
        *,
        max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
        max_total_bytes: int = DEFAULT_MAX_STORE_BYTES,
    ):
        self.root = Path(root)
        self.objects = self.root / "objects"
        self.manifests = self.root / "manifests"
        self.incoming = self.root / "incoming"
        self.max_artifact_bytes = max(1, int(max_artifact_bytes))
        self.max_total_bytes = max(1, int(max_total_bytes))
        self._lock = threading.RLock()
        for directory in (self.objects, self.manifests, self.incoming):
            directory.mkdir(parents=True, exist_ok=True)
        # Different services/store objects must not append to the same partial
        # concurrently. Keep only active stores and transfers in this registry.
        with _TRANSFER_STORES_GUARD:
            key=str(self.root.resolve());manager=_TRANSFER_STORES.get(key)
            if manager is None:manager=_StoreTransfers();_TRANSFER_STORES[key]=manager
            self._transfers=manager

    def _object_path(self, artifact_id: str) -> Path:
        if not _HEX_64_RE.fullmatch(str(artifact_id)):
            raise ArtifactError("invalid artifact ID")
        return self.objects / artifact_id[:2] / artifact_id[2:]

    def object_path(self, artifact_id: str) -> Path:
        """Return the validated local content-addressed object path."""
        return self._object_path(artifact_id)

    def _manifest_path(self, artifact_id: str) -> Path:
        if not _HEX_64_RE.fullmatch(str(artifact_id)):
            raise ArtifactError("invalid artifact ID")
        return self.manifests / f"{artifact_id}.json"

    def total_bytes(self) -> int:
        total = 0
        for path in self.objects.glob("*/*"):
            try:
                if path.is_file() and not path.is_symlink():
                    total += path.stat().st_size
            except OSError:
                continue
        return total

    def import_file(
        self,
        source_path: str | Path,
        *,
        kind: str = "other",
        media_type: str = "application/octet-stream",
    ) -> ArtifactManifest:
        source = Path(source_path)
        if not source.is_file() or source.is_symlink():
            raise ArtifactError("artifact source must be a regular non-symlink file")
        artifact_id, size = digest_file(source)
        manifest = ArtifactManifest(
            artifact_id=artifact_id,
            size=size,
            kind=kind,
            media_type=media_type,
            created_at=time.time(),
        )
        manifest.validate(max_bytes=self.max_artifact_bytes)
        destination = self._object_path(artifact_id)
        with self._lock:
            if not destination.exists():
                self._check_capacity(size)
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.incoming / (
                    f"{artifact_id}.{os.getpid()}.{os.urandom(8).hex()}.tmp"
                )
                try:
                    with source.open("rb") as reader, temporary.open("xb") as writer:
                        while True:
                            chunk = reader.read(1024 * 1024)
                            if not chunk:
                                break
                            writer.write(chunk)
                        writer.flush()
                        os.fsync(writer.fileno())
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
            self._write_manifest(manifest)
        return manifest

    def commit_generated(
        self,
        generated_path: str | Path,
        *,
        kind: str,
        media_type: str,
    ) -> ArtifactManifest:
        """Atomically commit a generated file already inside incoming/."""
        source = Path(generated_path)
        try:
            managed = source.resolve().parent == self.incoming.resolve()
        except OSError:
            managed = False
        if not managed or not source.is_file() or source.is_symlink():
            raise ArtifactError("generated artifact must be a regular incoming file")
        artifact_id, size = digest_file(source)
        manifest = ArtifactManifest(
            artifact_id=artifact_id,
            size=size,
            kind=kind,
            media_type=media_type,
            created_at=time.time(),
        )
        manifest.validate(max_bytes=self.max_artifact_bytes)
        destination = self._object_path(artifact_id)
        with self._lock:
            if destination.exists():
                source.unlink()
            else:
                self._check_capacity(size)
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, destination)
            self._write_manifest(manifest)
        return manifest

    def manifest(self, artifact_id: str) -> ArtifactManifest:
        path = self._manifest_path(artifact_id)
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError("artifact is not available") from exc
        manifest = ArtifactManifest.from_dict(
            value, max_bytes=self.max_artifact_bytes
        )
        object_path = self._object_path(artifact_id)
        try:
            if object_path.is_symlink() or object_path.stat().st_size != manifest.size:
                raise ArtifactError("artifact object does not match its manifest")
        except OSError as exc:
            raise ArtifactError("artifact object is missing") from exc
        return manifest

    def read_chunk(self, artifact_id: str, offset: int, length: int) -> bytes:
        manifest = self.manifest(artifact_id)
        offset = int(offset)
        length = int(length)
        if offset < 0 or offset > manifest.size:
            raise ArtifactError("artifact offset is out of range")
        if not 1 <= length <= ARTIFACT_CHUNK_BYTES:
            raise ArtifactError("artifact chunk length is out of range")
        with self._object_path(artifact_id).open("rb") as source:
            source.seek(offset)
            return source.read(min(length, manifest.size - offset))

    def begin_receive(self, manifest: ArtifactManifest) -> tuple[Path, int]:
        manifest.validate(max_bytes=self.max_artifact_bytes)
        if self.has(manifest.artifact_id):
            return self._object_path(manifest.artifact_id), manifest.size
        self._check_capacity(manifest.size)
        partial = self.incoming / f"{manifest.artifact_id}.part"
        current = partial.stat().st_size if partial.exists() else 0
        if current > manifest.size:
            partial.unlink()
            current = 0
        return partial, current

    def finalize_receive(self, manifest: ArtifactManifest, partial: Path) -> Path:
        artifact_id, size = digest_file(partial)
        if artifact_id != manifest.artifact_id or size != manifest.size:
            partial.unlink(missing_ok=True)
            raise ArtifactError("downloaded artifact failed SHA-256 verification")
        destination = self._object_path(manifest.artifact_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial, destination)
        self._write_manifest(manifest)
        return destination

    def has(self, artifact_id: str) -> bool:
        try:
            manifest = self.manifest(artifact_id)
            actual_id, actual_size = digest_file(self._object_path(artifact_id))
            return actual_id == manifest.artifact_id and actual_size == manifest.size
        except ArtifactError:
            return False

    def _check_capacity(self, incoming_bytes: int):
        if incoming_bytes > self.max_artifact_bytes:
            raise ArtifactError("artifact exceeds per-object quota")
        if self.total_bytes() + incoming_bytes > self.max_total_bytes:
            raise ArtifactError("artifact store quota exceeded")

    def _write_manifest(self, manifest: ArtifactManifest):
        destination = self._manifest_path(manifest.artifact_id)
        temporary = destination.with_suffix(
            f".json.{os.getpid()}.{os.urandom(8).hex()}.tmp"
        )
        temporary.write_text(json.dumps(
            manifest.to_dict(), sort_keys=True, separators=(",", ":")
        ))
        os.replace(temporary, destination)


class ArtifactService:
    """Register artifact read operations and fetch verified objects from peers."""

    def __init__(self, coordinator, store: ArtifactStore, *, receive_budget=None, namespace="artifact",register_handlers=True,prefer_compressed=True,fetch_window=1):
        if namespace not in {"artifact", "language.artifact", "public-memory.artifact"}:
            raise ArtifactError("unsupported artifact operation namespace")
        self.coordinator = coordinator
        self.store = store
        self.receive_budget = receive_budget
        self.namespace = namespace
        if type(prefer_compressed) is not bool:
            raise ArtifactError('Invalid local artifact encoding preference')
        self.prefer_compressed = prefer_compressed
        if type(fetch_window) is not int or fetch_window not in {1,2}:
            raise ArtifactError('Artifact receive window must be one or two chunks')
        self.fetch_window=fetch_window
        node=getattr(coordinator,'node',None)
        if (isinstance(getattr(node,'capabilities',None),dict) and getattr(coordinator,'max_offers_per_peer',2)>=4
                and getattr(coordinator,'_max_workers',1)>=2):
            node.capabilities['artifact_read_window_v1']=2
        if register_handlers:
            coordinator.register_handler(f"{namespace}.describe.v1", "artifact", self._describe)
            coordinator.register_handler(f"{namespace}.read-chunk.v1", "artifact", self._read_chunk)
            coordinator.register_handler(f"{namespace}.read-chunk.v2", "artifact", self._read_encoded_chunk)

    def _describe(self, payload: dict, _work) -> dict:
        if set(payload) != {"artifact_id"}:
            raise ArtifactError("invalid artifact describe request")
        return {"manifest": self.store.manifest(payload["artifact_id"]).to_dict(),
                "chunk_encodings": ['raw', ENCODING]}

    def _read_chunk(self, payload: dict, _work) -> dict:
        if set(payload) != {"artifact_id", "offset", "length"}:
            raise ArtifactError("invalid artifact chunk request")
        chunk = self.store.read_chunk(
            payload["artifact_id"], payload["offset"], payload["length"]
        )
        return {
            "offset": int(payload["offset"]),
            "data": base64.b64encode(chunk).decode("ascii"),
            "chunk_sha256": hashlib.sha256(chunk).hexdigest(),
        }

    def _read_encoded_chunk(self, payload: dict, _work) -> dict:
        if set(payload) != {"artifact_id", "offset", "length"}:
            raise ArtifactError('Invalid encoded artifact chunk request')
        chunk = self.store.read_chunk(payload['artifact_id'], payload['offset'], payload['length'])
        return {'offset': int(payload['offset']), 'chunk_sha256': hashlib.sha256(chunk).hexdigest(),
                **encode_chunk(chunk)}

    def fetch(
        self,
        artifact_id: str,
        *,
        peer_id: str,
        timeout_s: float = 60.0,
        cancel_event: Optional[threading.Event] = None,
        expected_size: Optional[int] = None,
        progress=lambda *_:None,
        observer=None,
    ) -> Path:
        if not _HEX_64_RE.fullmatch(str(artifact_id)):
            raise ArtifactError("invalid artifact ID")
        deadline = time.monotonic() + max(
            1.0, min(float(timeout_s), 60.0 * 60.0)
        )
        entry=self.store._transfers.entry(artifact_id);fetch_lock=entry.lock
        while not fetch_lock.acquire(
            timeout=min(0.1, self._remaining_timeout(deadline, cancel_event))
        ):
            pass
        try:
            try:
                return self._fetch_locked(
                    artifact_id,
                    peer_id=peer_id,
                    deadline=deadline,
                    cancel_event=cancel_event,
                    expected_size=expected_size,
                    progress=progress,
                    observer=observer,
                )
            except ArtifactError:raise
            except ValueError as error:raise ArtifactError(str(error)) from error
        finally:
            entry.lock.release()

    def _fetch_locked(
        self,
        artifact_id: str,
        *,
        peer_id: str,
        deadline: float,
        cancel_event: Optional[threading.Event],
        expected_size: Optional[int] = None,
        progress=lambda *_:None,
        observer=None,
    ) -> Path:
        if self.store.has(artifact_id):
            if expected_size is not None and self.store.manifest(artifact_id).size!=expected_size:
                raise ArtifactError("local artifact differs from the signed dependency size")
            return self.store._object_path(artifact_id)
        describe = self.coordinator.submit(
            f"{self.namespace}.describe.v1",
            {"artifact_id": artifact_id},
            ResourceRequirements(kind="artifact"),
            peer_id=peer_id,
            timeout_s=self._remaining_timeout(deadline, cancel_event),
            cancel_event=cancel_event,
        )
        if describe.status != "ok":
            raise ArtifactError(describe.error or "artifact manifest request failed")
        manifest = ArtifactManifest.from_dict(
            describe.output.get("manifest"),
            max_bytes=self.store.max_artifact_bytes,
        )
        if manifest.artifact_id != artifact_id:
            raise ArtifactError("peer returned a different artifact")
        if expected_size is not None and manifest.size!=expected_size:
            raise ArtifactError("peer artifact size differs from the signed dependency")
        encodings = describe.output.get('chunk_encodings')
        encoded = self.prefer_compressed and isinstance(encodings, list) and 'raw' in encodings and ENCODING in encodings
        # Negotiate through peer capabilities, preserving the exact signed
        # describe.v1 response used by historical dataset transfer proofs.
        peers=getattr(getattr(self.coordinator,'node',None),'peers',None)
        peer=next((p for p in peers() if p['node_id']==peer_id),{}) if callable(peers) else {}
        limit=peer.get('capabilities',{}).get('artifact_read_window_v1',1)
        if type(limit) is not int or not 1<=limit<=64:
            raise ArtifactError('Invalid provider artifact receive window')
        window=min(self.fetch_window,limit)
        partial, offset = self.store.begin_receive(manifest)
        if offset == manifest.size:
            return self.store.finalize_receive(manifest, partial)
        mode = "ab" if offset else "wb"
        # Workers receive and verify bounded chunks; only this thread appends
        # an ordered contiguous prefix. A failure never leaves an unwritten gap.
        internal_cancel=threading.Event()
        pool=ThreadPoolExecutor(max_workers=window,thread_name_prefix='lixy-artifact') if window>1 else None
        pending=deque();requested=offset
        def request_chunk(position,length):
            request_cancel=internal_cancel if pool is not None else cancel_event or internal_cancel
            slot=getattr(self.coordinator,'artifact_request_slot',None)
            guard=slot(peer_id,deadline=deadline,cancel_event=request_cancel) if callable(slot) else nullcontext()
            with guard:
                began=time.monotonic()
                result=self.coordinator.submit(
                    f"{self.namespace}.read-chunk.v2" if encoded else f"{self.namespace}.read-chunk.v1",
                    {'artifact_id':artifact_id,'offset':position,'length':length},ResourceRequirements(kind='artifact'),
                    peer_id=peer_id,timeout_s=self._remaining_timeout(deadline,request_cancel),
                    cancel_event=request_cancel)
                if result.status!='ok':raise ArtifactError(result.error or 'artifact chunk request failed')
                if result.output.get('offset')!=position:raise ArtifactError('peer returned a chunk at the wrong offset')
                try:
                    chunk=decode_chunk(result.output,length) if encoded else base64.b64decode(result.output.get('data',''),validate=True)
                except (ValueError,TypeError) as error:raise ArtifactError('peer returned invalid chunk encoding') from error
                if len(chunk)!=length:raise ArtifactError('peer returned an invalid chunk length')
                if hashlib.sha256(chunk).hexdigest()!=result.output.get('chunk_sha256'):raise ArtifactError('artifact chunk hash does not match')
                received=time.monotonic()
                if self.receive_budget is not None:
                    # Charge every received valid block, including prefetched
                    # data that will not be committed after another block fails.
                    self.receive_budget.wait(len(result.output['data'])+2048,
                        cancel_event=internal_cancel if pool is not None else cancel_event,
                        deadline=time.time()+max(0,deadline-time.monotonic()))
                return chunk,{'rpc_seconds':received-began,'received_at':received,
                    'encoded_data_bytes':len(result.output['data']),'encoding':result.output.get('encoding','raw')}
        def fill_window():
            nonlocal requested
            while pool is not None and len(pending)<window and requested<manifest.size:
                length=min(ARTIFACT_CHUNK_BYTES,manifest.size-requested)
                pending.append(pool.submit(request_chunk,requested,length));requested+=length
        def next_chunk(position,length):
            if pool is None:return request_chunk(position,length)
            future=pending.popleft()
            while True:
                self._remaining_timeout(deadline,cancel_event)
                try:return future.result(timeout=min(.05,self._remaining_timeout(deadline,cancel_event)))
                except FutureTimeout:
                    if future.done():return future.result()
        with partial.open(mode) as destination:
            try:
                progress(offset,manifest.size);fill_window()
                while offset<manifest.size:
                    length=min(ARTIFACT_CHUNK_BYTES,manifest.size-offset)
                    chunk,stats=next_chunk(offset,length)
                    self._remaining_timeout(deadline,cancel_event)
                    destination.write(chunk);offset+=len(chunk);progress(offset,manifest.size)
                    if observer is not None:
                        received=stats.pop('received_at')
                        observer({'offset':offset,'bytes':len(chunk),**stats,
                            'receive_and_write_seconds':time.monotonic()-received,'fetch_window':window})
                    fill_window()
                destination.flush();os.fsync(destination.fileno())
            finally:
                internal_cancel.set()
                if pool is not None:
                    for future in pending:future.cancel()
                    pool.shutdown(wait=True,cancel_futures=True)
        return self.store.finalize_receive(manifest, partial)

    @staticmethod
    def _remaining_timeout(
        deadline: float,
        cancel_event: Optional[threading.Event],
    ) -> float:
        if cancel_event is not None and cancel_event.is_set():
            raise ArtifactError("artifact fetch cancelled")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ArtifactError("artifact fetch timed out")
        return remaining
