import hashlib
import threading

import pytest

from src.contribution import ContributionPolicy, ResourceGovernor
from src.network.artifact_store import ArtifactError, ArtifactService, ArtifactStore
from src.network.lsp import LSPIdentity
from src.network.lsp_v3 import LSPNodeV3
from src.network.work_protocol import WorkCoordinator
from src.swarm.node_manager import HardwareProfile


HARDWARE = HardwareProfile(
    cpu_cores=4, ram_gb=8, gpu_vram_gb=0, disk_gb=20, has_gpu=False
)


def _governor(path):
    return ResourceGovernor(
        ContributionPolicy.for_mode("relay"), hardware=HARDWARE, storage_path=path
    )


def _node(tmp_path, name, governor):
    return LSPNodeV3(
        LSPIdentity.generate(),
        host="127.0.0.1",
        port=0,
        address_book_path=tmp_path / f"{name}.json",
        target_outbound=0,
        allow_private=True,
        resource_profile=governor.advertised_profile(),
    )


def test_artifact_store_uses_content_identity_without_source_name(tmp_path):
    source = tmp_path / "operator-private-name.bin"
    source.write_bytes(b"model bytes")
    store = ArtifactStore(tmp_path / "store", max_total_bytes=1024)
    manifest = store.import_file(source, kind="model")

    assert manifest.artifact_id == hashlib.sha256(b"model bytes").hexdigest()
    assert "operator-private-name" not in str(store._object_path(manifest.artifact_id))
    assert store.read_chunk(manifest.artifact_id, 0, 32) == b"model bytes"


def test_artifact_store_rejects_symlinks_and_quota_overflow(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"12345")
    link = tmp_path / "link.bin"
    link.symlink_to(source)
    store = ArtifactStore(tmp_path / "store", max_total_bytes=4)
    with pytest.raises(ArtifactError, match="non-symlink"):
        store.import_file(link)
    with pytest.raises(ArtifactError, match="quota"):
        store.import_file(source)


def test_complete_partial_is_verified_and_committed(tmp_path):
    content = b"resumed content"
    artifact_id = hashlib.sha256(content).hexdigest()
    store = ArtifactStore(tmp_path / "store")
    from src.network.artifact_store import ArtifactManifest
    import time
    manifest = ArtifactManifest(
        artifact_id=artifact_id,
        size=len(content),
        kind="dataset",
        media_type="application/octet-stream",
        created_at=time.time(),
    )
    partial, _ = store.begin_receive(manifest)
    partial.write_bytes(content)
    partial, offset = store.begin_receive(manifest)
    assert offset == len(content)
    committed = store.finalize_receive(manifest, partial)
    assert committed.read_bytes() == content
    assert store.has(artifact_id)


def test_artifact_transfer_is_chunked_and_end_to_end_verified(tmp_path):
    provider_governor = _governor(tmp_path / "provider-state")
    requester_governor = _governor(tmp_path / "requester-state")
    provider = _node(tmp_path, "provider", provider_governor)
    requester = _node(tmp_path, "requester", requester_governor)
    provider.start()
    requester.start()
    provider_work = WorkCoordinator(provider, provider_governor)
    requester_work = WorkCoordinator(requester, requester_governor)
    provider_store = ArtifactStore(tmp_path / "provider-artifacts")
    requester_store = ArtifactStore(tmp_path / "requester-artifacts")
    ArtifactService(provider_work, provider_store)
    requester_artifacts = ArtifactService(requester_work, requester_store)
    content = bytes(range(256)) * 1000
    source = tmp_path / "dataset.bin"
    source.write_bytes(content)
    manifest = provider_store.import_file(
        source, kind="dataset", media_type="application/octet-stream"
    )
    try:
        assert requester.connect_peer("127.0.0.1", provider.port)
        downloaded = requester_artifacts.fetch(
            manifest.artifact_id,
            peer_id=provider.identity.node_id_hex,
            timeout_s=5,
        )
        assert downloaded.read_bytes() == content
        assert requester_store.manifest(manifest.artifact_id) == manifest
    finally:
        provider_work.close()
        requester_work.close()
        provider.stop()
        requester.stop()


def test_artifact_fetch_wait_is_cancellable(tmp_path):
    class Coordinator:
        def register_handler(self, _operation, _kind, _handler):
            pass

        def submit(self, *_args, **_kwargs):
            raise AssertionError("cancelled fetch must not submit work")

    service = ArtifactService(
        Coordinator(), ArtifactStore(tmp_path / "cancelled-store")
    )
    artifact_id = "ab" * 32
    occupied = threading.Lock()
    occupied.acquire()
    service._fetch_locks[artifact_id] = occupied
    cancelled = threading.Event()
    cancelled.set()
    try:
        with pytest.raises(ArtifactError, match="cancelled"):
            service.fetch(
                artifact_id,
                peer_id="cd" * 32,
                timeout_s=5,
                cancel_event=cancelled,
            )
    finally:
        occupied.release()
