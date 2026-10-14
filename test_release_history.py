from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.network.lsp import LSPIdentity
from src.network.work_protocol import ResultReceipt, WorkResult
from src.release import ReleaseError, ReleaseManifest, ReleaseRegistry, TrustPolicy
from src.release.history import ReleaseHistoryService, fetch_release_history


def fixture(tmp_path, mutate=None):
    signer, provider, consumer = (LSPIdentity.generate() for _ in range(3))
    policy = TrustPolicy(trusted_signers=(signer.node_id_hex,))
    registry = ReleaseRegistry(tmp_path / "releases")
    chain = []
    for sequence in range(5):
        manifest = ReleaseManifest.create(model_artifact_id="ab" * 32,
            model_format="pytorch-weights-only-v1", sequence=sequence,
            previous_release_id=chain[-1].release_id if chain else "").sign(signer)
        manifest.save(registry.manifests / f"{manifest.release_id}.json")
        chain.append(manifest)
    class Coordinator:
        node = SimpleNamespace(identity=consumer)
        calls = 0

        def register_handler(self, operation, kind, handler):
            self.handler = handler

        def submit(self, operation, payload, requirements, **kwargs):
            self.calls += 1
            output = self.handler(payload, SimpleNamespace(raise_if_cancelled=lambda: None))
            if mutate:
                output = mutate(output)
            result = WorkResult(job_id=f"{self.calls:064x}", status="ok", output=output)
            return replace(result, receipt=ResultReceipt.create(result, provider,
                consumer.node_id_hex).to_dict())
    coordinator = Coordinator()
    service = ReleaseHistoryService(coordinator, registry, policy)
    return coordinator, service, chain, provider, policy


def test_replicated_history_keeps_publisher_trust_separate_from_provider(tmp_path):
    coordinator, _, chain, provider, policy = fixture(tmp_path)
    acquired = fetch_release_history(coordinator, release_id=chain[-1].release_id,
                                     peer_id=provider.node_id_hex, policy=policy)
    assert acquired == chain and coordinator.calls == 3
    assert provider.node_id_hex not in policy.trusted_signers


def test_history_rejects_extra_storage_work_beyond_its_bound(tmp_path):
    coordinator, _, chain, provider, policy = fixture(tmp_path)
    with pytest.raises(ReleaseError, match="acquisition bound"):
        fetch_release_history(coordinator, release_id=chain[-1].release_id,
            peer_id=provider.node_id_hex, policy=policy, max_releases=2)


def test_provider_signature_cannot_authorize_reordered_manifests(tmp_path):
    coordinator, _, chain, provider, policy = fixture(tmp_path,
        mutate=lambda p: {**p, "releases":list(reversed(p["releases"]))})
    with pytest.raises(ReleaseError, match="expected chain"):
        fetch_release_history(coordinator, release_id=chain[-1].release_id,
            peer_id=provider.node_id_hex, policy=policy)


def test_history_requires_the_continuation_signed_into_each_release(tmp_path):
    coordinator, _, chain, provider, policy = fixture(tmp_path,
        mutate=lambda p: {**p, "next_release_id":""})
    with pytest.raises(ReleaseError, match="continuation"):
        fetch_release_history(coordinator, release_id=chain[-1].release_id,
            peer_id=provider.node_id_hex, policy=policy)


def test_history_does_not_accept_receipts_from_another_provider(tmp_path):
    coordinator, _, chain, _, policy = fixture(tmp_path)
    with pytest.raises(ReleaseError, match="receipt"):
        fetch_release_history(coordinator, release_id=chain[-1].release_id,
            peer_id=LSPIdentity.generate().node_id_hex, policy=policy)


def test_history_request_cannot_escape_the_registry(tmp_path):
    _, service, _, _, _ = fixture(tmp_path)
    with pytest.raises(ReleaseError, match="request"):
        service._page({"release_id":"../../identity.key"}, None)
