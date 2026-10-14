import json
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.contribution.policy import ContributionPolicy
from src.contribution.resource_governor import ResourceRequirements
from src.network.artifact_store import ArtifactStore
from src.network.client_state import ClientSettings
from src.network.learning_coordinator import LearningCoordinator,LearningSettings
from src.network.reference_compute import EngineSettings
from src.network.lsp import LSPIdentity
from src.network.lsp_v3 import LSPNodeV3
from src.network.public_knowledge import PublicKnowledgeService,KnowledgeSettings
from src.network.work_protocol import ResultReceipt
from src.release import ReleaseManifest,ReleaseRegistry,TrustPolicy
from src.release.reference_catalog import CatalogEntry

RESOURCES={"cpu_cores":4,"ram_gb":8,"has_gpu":False,"gpu_vram_gb":0}


def setup(tmp_path):
    signer=LSPIdentity.generate(); signer.save(str(tmp_path/"publisher.key"))
    source=tmp_path/"base.bin"; source.write_bytes(b"public base")
    store=ArtifactStore(tmp_path/"node/public/artifacts"); artifact=store.import_file(source,kind="model")
    genesis=ReleaseManifest.create(model_artifact_id=artifact.artifact_id,model_format="pytorch-weights-only-v1",sequence=0).sign(signer)
    policy=TrustPolicy(trusted_signers=(signer.node_id_hex,),pinned_genesis_release_id=genesis.release_id)
    registry=ReleaseRegistry(tmp_path/"node/public/releases"); registry.accept(genesis,policy,store); registry.activate(genesis.release_id,policy,store)
    node=LSPNodeV3(LSPIdentity.generate(),host="127.0.0.1",port=0,address_book_path=tmp_path/"peers.json",target_outbound=0,allow_private=True)
    catalog=CatalogEntry("test",genesis.release_id,policy,allow_signed_updates=True)
    knowledge=PublicKnowledgeService(node,tmp_path/"node",RESOURCES,catalog=catalog)
    knowledge.update(ClientSettings(),KnowledgeSettings(enabled=True,consented_at=time.time()))
    compute=SimpleNamespace(ready=True,settings=ClientSettings(policy=ContributionPolicy.for_mode("balanced")),engine=EngineSettings())
    return signer,node,knowledge,LearningCoordinator(knowledge,compute,tmp_path/"node")


def test_local_work_receipt_uses_the_existing_physical_identity(tmp_path):
    signer,node,knowledge,coordinator=setup(tmp_path)
    try:
        knowledge.work.register_handler("artifact.local-test.v1","artifact",lambda payload,execution:{"value":42})
        result=knowledge.work.execute_local("artifact.local-test.v1",{},ResourceRequirements(kind="artifact"))
        assert result.status=="ok"
        assert ResultReceipt.from_dict(result.receipt).verify(result,expected_worker_id=node.identity.node_id_hex,
            expected_requester_id=node.identity.node_id_hex)
    finally:
        coordinator.close(); knowledge.close()


def test_pending_promotion_recovers_once_and_keeps_cursor_and_chain(tmp_path):
    signer,node,knowledge,coordinator=setup(tmp_path)
    try:
        source=tmp_path/"improved.bin"; source.write_bytes(b"public improved")
        artifact=knowledge.store.import_file(source,kind="model")
        base=knowledge.registry.active()
        release=ReleaseManifest.create(model_artifact_id=artifact.artifact_id,model_format="pytorch-weights-only-v1",
            sequence=1,previous_release_id=base.release_id).sign(signer)
        coordinator._save(status="promoting",release=release.to_dict())
        restored=LearningCoordinator(knowledge,coordinator.compute,tmp_path/"node")
        restored._recover_promotion()
        assert knowledge.registry.active().release_id==release.release_id
        assert restored.ledger["round"]==1 and restored.ledger["cursor"]==65
        restored._recover_promotion()
        assert restored.ledger["round"]==1 and knowledge.complete
        again=PublicKnowledgeService(node,tmp_path/"node",RESOURCES,catalog=knowledge.catalog)
        assert again.desired_release_id==release.release_id and again.complete
        again.close()
    finally:
        coordinator.close(); knowledge.close()


def test_untrusted_catalog_update_cannot_change_the_desired_version(tmp_path):
    signer,node,knowledge,coordinator=setup(tmp_path)
    try:
        base=knowledge.registry.active()
        forged=ReleaseManifest.create(model_artifact_id="ab"*32,model_format="pytorch-weights-only-v1",
            sequence=1,previous_release_id=base.release_id).sign(LSPIdentity.generate())
        knowledge._on_release(forged.to_dict(),LSPIdentity.generate().node_id_hex)
        assert knowledge.desired_release_id==base.release_id and knowledge.complete
    finally:
        coordinator.close(); knowledge.close()


def test_learning_controls_reject_duplicate_workers_and_unbounded_runs(tmp_path):
    worker=LSPIdentity.generate().node_id_hex
    with pytest.raises(ValueError): LearningSettings(enabled=True,cycles=11,workers=[worker],signing_key_file=str(tmp_path/"key")).save(tmp_path)
    with pytest.raises(ValueError): LearningSettings(enabled=True,workers=[worker,worker],signing_key_file=str(tmp_path/"key")).save(tmp_path)


def test_cpu_engine_on_gpu_hardware_gets_cpu_work_and_remote_permissions_apply(tmp_path,monkeypatch):
    _,node,knowledge,coordinator=setup(tmp_path)
    try:
        knowledge.resources={**RESOURCES,"has_gpu":True,"gpu_vram_gb":32}
        coordinator.compute.engine=EngineSettings(devices=["cpu"])
        assert not coordinator._training_requirements(node.identity.node_id_hex).gpu_required
        coordinator.compute.engine=EngineSettings(devices=["cpu","cuda"])
        assert coordinator._training_requirements(node.identity.node_id_hex).gpu_required
        remote="a"*64
        monkeypatch.setattr(node,"peers",lambda:[{"node_id":remote,"resources":{"has_gpu":False}}])
        assert not coordinator._training_requirements(remote).gpu_required
    finally:
        coordinator.close(); knowledge.close()
