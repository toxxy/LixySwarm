from dataclasses import replace
import time

import pytest

from src.network.artifact_store import ArtifactStore
from src.network.lsp import LSPIdentity
from src.network.swarm_network import SwarmNetwork
from src.contribution import ContributionPolicy, ResourceGovernor
from src.release import ReleaseError, ReleaseManifest, ReleaseRegistry, TrustPolicy


def _signed(manifest, identities):
    for identity in identities:
        manifest = manifest.sign(identity)
    return manifest


def test_threshold_release_signatures_bind_every_field():
    signers = [LSPIdentity.generate() for _ in range(3)]
    policy = TrustPolicy(
        threshold=2,
        trusted_signers=tuple(identity.node_id_hex for identity in signers),
    )
    manifest = ReleaseManifest.create(
        model_artifact_id="ab" * 32,
        model_format="pytorch-weights-only-v1",
        sequence=0,
    )
    signed = _signed(manifest, signers[:2])
    assert policy.authorize(signed) == {
        signers[0].node_id_hex, signers[1].node_id_hex
    }
    with pytest.raises(ReleaseError, match="release ID"):
        replace(signed, model_artifact_id="cd" * 32).validate()
    with pytest.raises(ReleaseError, match="threshold"):
        policy.authorize(manifest.sign(signers[0]))


def test_release_registry_prevents_downgrade_and_requires_explicit_rollback(tmp_path):
    signer_a = LSPIdentity.generate()
    signer_b = LSPIdentity.generate()
    policy = TrustPolicy(
        threshold=2,
        trusted_signers=(signer_a.node_id_hex, signer_b.node_id_hex),
    )
    store = ArtifactStore(tmp_path / "artifacts")
    model_v0_path = tmp_path / "model-v0.bin"
    model_v0_path.write_bytes(b"trusted model v0")
    model_v0 = store.import_file(model_v0_path, kind="model")
    genesis = _signed(ReleaseManifest.create(
        model_artifact_id=model_v0.artifact_id,
        model_format="pytorch-weights-only-v1",
        sequence=0,
    ), [signer_a, signer_b])
    registry = ReleaseRegistry(tmp_path / "releases")
    registry.accept(genesis, policy, store)
    assert registry.activate(genesis.release_id, policy, store) == genesis

    model_v1_path = tmp_path / "model-v1.bin"
    model_v1_path.write_bytes(b"trusted model v1")
    model_v1 = store.import_file(model_v1_path, kind="model")
    release_v1 = _signed(ReleaseManifest.create(
        model_artifact_id=model_v1.artifact_id,
        model_format="pytorch-weights-only-v1",
        sequence=1,
        previous_release_id=genesis.release_id,
    ), [signer_a, signer_b])
    registry.accept(release_v1, policy, store)
    registry.activate(release_v1.release_id, policy, store)
    assert registry.active().release_id == release_v1.release_id

    with pytest.raises(ReleaseError, match="explicit"):
        registry.rollback(
            genesis.release_id, policy, store, explicit_confirmation=False
        )
    registry.rollback(
        genesis.release_id, policy, store, explicit_confirmation=True
    )
    assert registry.active().release_id == genesis.release_id

    unrelated = _signed(ReleaseManifest.create(
        model_artifact_id=model_v0.artifact_id,
        model_format="pytorch-weights-only-v1",
        sequence=0,
        created_at=genesis.created_at + 0.001,
    ), [signer_a, signer_b])
    registry.accept(unrelated, policy, store)
    registry.activate(release_v1.release_id, policy, store)
    with pytest.raises(ReleaseError, match="active release chain"):
        registry.rollback(
            unrelated.release_id, policy, store, explicit_confirmation=True
        )


def test_release_rejects_unsafe_model_format():
    with pytest.raises(ReleaseError, match="unsafe"):
        ReleaseManifest.create(
            model_artifact_id="ab" * 32,
            model_format="pickle-arbitrary-code",
            sequence=0,
        )


def test_trusted_release_and_model_propagate_peer_to_peer(tmp_path):
    signer = LSPIdentity.generate()
    policy = TrustPolicy(
        threshold=1, trusted_signers=(signer.node_id_hex,)
    )
    provider_store = ArtifactStore(tmp_path / "provider-artifacts")
    receiver_store = ArtifactStore(tmp_path / "receiver-artifacts")
    model_path = tmp_path / "model.bin"
    model_path.write_bytes(b"peer distributed signed model")
    model = provider_store.import_file(model_path, kind="model")
    release = ReleaseManifest.create(
        model_artifact_id=model.artifact_id,
        model_format="pytorch-weights-only-v1",
        sequence=0,
    ).sign(signer)
    provider_registry = ReleaseRegistry(tmp_path / "provider-releases")
    receiver_registry = ReleaseRegistry(tmp_path / "receiver-releases")
    provider_registry.accept(release, policy, provider_store)
    provider_registry.activate(release.release_id, policy, provider_store)

    provider_governor = ResourceGovernor(
        ContributionPolicy.for_mode("relay"),
        storage_path=tmp_path / "provider-state",
    )
    receiver_governor = ResourceGovernor(
        ContributionPolicy.for_mode("relay"),
        storage_path=tmp_path / "receiver-state",
    )
    provider = SwarmNetwork.create(
        mode="lan", gossip_port=0, target_outbound=0,
        checkpoint_dir=tmp_path / "provider-net",
        contribution_profile=provider_governor.advertised_profile(
            available_work={"artifact"}
        ),
    )
    receiver = SwarmNetwork.create(
        mode="lan", gossip_port=0, target_outbound=0,
        checkpoint_dir=tmp_path / "receiver-net",
        contribution_profile=receiver_governor.advertised_profile(
            available_work={"artifact"}
        ),
    )
    provider.start()
    receiver.start()
    provider.enable_work(provider_governor)
    receiver.enable_work(receiver_governor)
    provider.enable_artifacts(provider_store)
    receiver.enable_artifacts(receiver_store)
    provider.enable_release_distribution(
        provider_registry, policy, provider_store
    )
    receiver.enable_release_distribution(
        receiver_registry, policy, receiver_store, auto_activate=True
    )
    try:
        assert receiver.connect_peer("127.0.0.1", provider._lsp_v3_node.port)
        deadline = time.time() + 8
        while time.time() < deadline:
            active = receiver_registry.active()
            if active and receiver_store.has(model.artifact_id):
                break
            time.sleep(0.05)
        assert receiver_registry.active().release_id == release.release_id
        assert receiver_store.has(model.artifact_id)
    finally:
        receiver.stop()
        provider.stop()
