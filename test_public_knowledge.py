import json
import threading
import time
from pathlib import Path

import pytest

from src.contribution import ContributionPolicy, ResourceRequirements
from src.network.artifact_store import ArtifactError, ArtifactStore
from src.network.client_state import ClientSettings
from src.network.lsp import LSPIdentity
from src.network.lsp_v3 import LSPNodeV3
from src.network.public_knowledge import (
    BandwidthBudget, KnowledgeSettings, PublicKnowledgeService, import_public_run,
)
from src.release import ReleaseManifest, ReleaseRegistry, TrustPolicy
from src.release.reference_catalog import CatalogEntry
from src.network.work_protocol import WorkUnit

RESOURCES = {"cpu_cores":4,"ram_gb":8,"gpu_vram_gb":0,"has_gpu":False}


def run_fixture(path):
    signer = LSPIdentity.generate()
    store = ArtifactStore(path / "artifacts")
    registry = ReleaseRegistry(path / "releases")
    chain = []
    policy = TrustPolicy(trusted_signers=(signer.node_id_hex,))
    for sequence in range(2):
        model = path / f"model-{sequence}.bin"
        model.write_bytes(bytes([sequence])*200000)
        artifact = store.import_file(model, kind="model")
        release = ReleaseManifest.create(model_artifact_id=artifact.artifact_id,
            model_format="pytorch-weights-only-v1", sequence=sequence,
            previous_release_id=chain[-1].release_id if chain else "").sign(signer)
        registry.accept(release, policy, store); registry.activate(release.release_id, policy, store)
        chain.append(release)
    catalog = CatalogEntry("test-public-family", chain[-1].release_id,
        TrustPolicy(trusted_signers=(signer.node_id_hex,), pinned_genesis_release_id=chain[0].release_id))
    return catalog, chain


def node(path):
    return LSPNodeV3(LSPIdentity.generate(), host="127.0.0.1", port=0,
        address_book_path=path/"peers.json", target_outbound=0, allow_private=True)


def wait(predicate, timeout=12):
    until = time.monotonic()+timeout
    while time.monotonic()<until:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Public knowledge did not reach expected state")


def test_shared_storage_requires_separate_consent_and_rejects_unknown_fields(tmp_path):
    assert not KnowledgeSettings.load(tmp_path).enabled
    with pytest.raises(ValueError, match="consent"):
        KnowledgeSettings(enabled=True).save(tmp_path)
    KnowledgeSettings(enabled=True, consented_at=time.time()).save(tmp_path)
    assert KnowledgeSettings.load(tmp_path).enabled
    path = tmp_path/"public-knowledge.json"
    value = json.loads(path.read_text()); value["personal_file"] = "personal.enc"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="schema"):
        KnowledgeSettings.load(tmp_path)


def test_import_only_copies_signed_release_objects_and_never_other_banks(tmp_path):
    source = tmp_path/"old-run"; source.mkdir()
    catalog, chain = run_fixture(source)
    private = source/"personal.enc"; private.write_bytes(b"private-canary-public-storage-73")
    extra = ArtifactStore(source/"artifacts").import_file(private, kind="other")
    target = tmp_path/"node-state"
    final = import_public_run(source, target, catalog=catalog)
    assert final.release_id == chain[-1].release_id
    store = ArtifactStore(target/"public/artifacts")
    assert not store.has(extra.artifact_id)
    assert all(b"private-canary-public-storage-73" not in p.read_bytes()
               for p in target.rglob("*") if p.is_file())
    # Repeating migration is idempotent and cannot replay activation backwards.
    assert import_public_run(source,target,catalog=catalog).release_id == final.release_id


def test_existing_physical_identity_serves_and_fresh_node_acquires_signed_history(tmp_path):
    source = tmp_path/"old-run"; source.mkdir()
    catalog, chain = run_fixture(source)
    provider_dir, receiver_dir = tmp_path/"provider", tmp_path/"receiver"
    import_public_run(source, provider_dir, catalog=catalog)
    provider, receiver = node(provider_dir), node(receiver_dir)
    original_id = provider.identity.node_id_hex
    service = PublicKnowledgeService(provider,provider_dir,RESOURCES,catalog=catalog)
    newcomer = PublicKnowledgeService(receiver,receiver_dir,RESOURCES,catalog=catalog)
    settings = ClientSettings()
    sharing = KnowledgeSettings(enabled=True,consented_at=time.time())
    try:
        service.update(settings,sharing)
        provider.start(); receiver.start()
        assert receiver.connect_peer("127.0.0.1",provider.port)
        newcomer.update(settings,sharing)
        wait(lambda:newcomer.snapshot()["state"]=="available")
        assert newcomer.registry.active().release_id == chain[-1].release_id
        assert newcomer.store.total_bytes()==400000
        assert provider.identity.node_id_hex == original_id
        settings.paused=True; service.update(settings,sharing)
        result=newcomer.work.submit("artifact.describe.v1",{"artifact_id":chain[-1].model_artifact_id},
            ResourceRequirements(kind="artifact"),peer_id=original_id,timeout_s=3)
        assert result.status != "ok"
        assert service.store.has(chain[-1].model_artifact_id)
        settings.paused=False; service.update(settings,sharing)
    finally:
        newcomer.close(); service.close(); receiver.stop(); provider.stop()
    restored = PublicKnowledgeService(node(receiver_dir),receiver_dir,RESOURCES,catalog=catalog)
    try:
        restored.update(settings,sharing)
        assert restored.snapshot()["state"]=="available"
        assert restored.registry.active().release_id == chain[-1].release_id
    finally:
        restored.close()


def test_transfer_budget_is_interruptible_and_zero_budget_denies_work():
    with pytest.raises(ArtifactError,match="disabled"):
        BandwidthBudget(lambda:0).wait(1000)
    cancelled=threading.Event(); done=threading.Event()
    def run():
        with pytest.raises(ArtifactError,match="cancelled"):
            BandwidthBudget(lambda:0.001).wait(100000,cancel_event=cancelled)
        done.set()
    thread=threading.Thread(target=run); thread.start()
    cancelled.set(); assert done.wait(1); thread.join()


def test_lowering_disk_quota_preserves_data_and_disables_sharing(tmp_path):
    source=tmp_path/"source"; source.mkdir(); catalog,chain=run_fixture(source)
    target=tmp_path/"target"; import_public_run(source,target,catalog=catalog)
    service=PublicKnowledgeService(node(target),target,RESOURCES,catalog=catalog)
    try:
        settings=ClientSettings(); settings.policy.max_disk_gb=0.0001
        service.update(settings,KnowledgeSettings(enabled=True,consented_at=time.time()))
        assert not service.snapshot()["sharing"]
        assert service.store.has(chain[-1].model_artifact_id)
    finally:
        service.close()


def test_cached_results_cannot_bypass_withdrawn_consent(tmp_path):
    source=tmp_path/"source"; source.mkdir(); catalog,_=run_fixture(source)
    service=PublicKnowledgeService(node(tmp_path/"node"),tmp_path/"node",RESOURCES,catalog=catalog)
    try:
        settings=ClientSettings(); service.update(settings,KnowledgeSettings(enabled=True,consented_at=time.time()))
        calls=[]; outputs=[]; enabled=True
        service.node.send_work_result=lambda peer,value:outputs.append(value) or True
        service.work.register_handler("artifact.cached-test.v1","artifact",lambda payload,execution:calls.append(1) or {"value":42},
            permit_cached=lambda execution,output:enabled)
        peer=LSPIdentity.generate()
        work=WorkUnit.create(origin_node_id=peer.node_id_hex,operation="artifact.cached-test.v1",kind="artifact",
            requirements=ResourceRequirements(kind="artifact"),payload={},timeout_s=30)
        service.work._execute_offer(work.to_dict(),peer.node_id_hex)
        service.work._execute_offer(work.to_dict(),peer.node_id_hex)
        assert len(calls)==1 and outputs[-1]["status"]=="ok"
        enabled=False
        service.work._execute_offer(work.to_dict(),peer.node_id_hex)
        assert outputs[-1]["status"]=="rejected" and len(calls)==1
        enabled=True; service.governor.policy=ContributionPolicy.for_mode("disabled")
        service.work._execute_offer(work.to_dict(),peer.node_id_hex)
        assert outputs[-1]["status"]=="rejected" and len(calls)==1
    finally:
        service.close()
