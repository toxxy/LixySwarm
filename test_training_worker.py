from types import SimpleNamespace

import numpy as np
import torch

from src.contribution import ContributionPolicy, ResourceGovernor, ResourceRequirements
from src.network.artifact_store import ArtifactService, ArtifactStore
from src.network.lsp import LSPIdentity
from src.network.lsp_v3 import LSPNodeV3
from src.network.training_worker import (
    TRAINING_OPERATION,
    TrainingWorker,
    validate_gradient_artifact,
)
from src.network.work_protocol import WorkCoordinator
from src.swarm.node_manager import HardwareProfile


HARDWARE = HardwareProfile(
    cpu_cores=4, ram_gb=8, gpu_vram_gb=0, disk_gb=20, has_gpu=False
)


class TinyTrainingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 8)
        self.output = torch.nn.Linear(8, 32)
        self.config = SimpleNamespace(
            agent_configs=[SimpleNamespace(vocab_size=32)]
        )

    def forward(self, idx, targets=None, **_kwargs):
        logits = self.output(self.embedding(idx))
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
        )
        return logits, loss, torch.zeros((idx.shape[0], 4))


def _governor(path, mode):
    governor = ResourceGovernor(
        ContributionPolicy.for_mode(mode), hardware=HARDWARE, storage_path=path
    )
    governor._system_busy = lambda: False
    return governor


def _node(tmp_path, name, governor, work):
    return LSPNodeV3(
        LSPIdentity.generate(), host="127.0.0.1", port=0,
        address_book_path=tmp_path / f"{name}.json", target_outbound=0,
        allow_private=True,
        resource_profile=governor.advertised_profile(available_work=work),
    )


def test_training_worker_fetches_dataset_and_returns_verified_gradient(tmp_path):
    requester_governor = _governor(tmp_path / "requester-state", "relay")
    worker_governor = _governor(tmp_path / "worker-state", "balanced")
    requester = _node(tmp_path, "requester", requester_governor, {"artifact"})
    worker = _node(tmp_path, "worker", worker_governor, {"artifact", "training"})
    requester.start()
    worker.start()
    requester_work = WorkCoordinator(requester, requester_governor)
    worker_work = WorkCoordinator(worker, worker_governor)
    requester_store = ArtifactStore(tmp_path / "requester-artifacts")
    worker_store = ArtifactStore(tmp_path / "worker-artifacts")
    requester_artifacts = ArtifactService(requester_work, requester_store)
    worker_artifacts = ArtifactService(worker_work, worker_store)
    model_id = "ab" * 32
    model = TinyTrainingModel()
    TrainingWorker(
        worker_work,
        worker_artifacts,
        model,
        model_artifact_id=model_id,
        device="cpu",
    )
    dataset_path = tmp_path / "tokens.npy"
    np.save(dataset_path, np.arange(24, dtype=np.int32) % 32, allow_pickle=False)
    dataset = requester_store.import_file(
        dataset_path, kind="dataset", media_type="application/x-npy"
    )
    try:
        assert requester.connect_peer("127.0.0.1", worker.port)
        result = requester_work.submit(
            TRAINING_OPERATION,
            {
                "model_artifact_id": model_id,
                "dataset_artifact_id": dataset.artifact_id,
                "start_token": 0,
                "token_count": 16,
            },
            ResourceRequirements(kind="training", ram_gb=0.1, disk_gb=0.1),
            peer_id=worker.identity.node_id_hex,
            timeout_s=10,
        )
        assert result.status == "ok", result.error
        gradient_id = result.output["gradient"]["artifact_id"]
        gradient_path = requester_artifacts.fetch(
            gradient_id, peer_id=worker.identity.node_id_hex, timeout_s=5
        )
        with np.load(gradient_path, allow_pickle=False) as gradient:
            assert "embedding.weight" in gradient.files
            assert "output.weight" in gradient.files
            assert "__lixy_metadata__" in gradient.files
        validation = validate_gradient_artifact(
            gradient_path,
            model,
            expected_model_id=model_id,
            expected_dataset_id=dataset.artifact_id,
            expected_start_token=0,
            expected_token_count=16,
        )
        assert validation["token_count"] == 16
        assert worker_store.has(dataset.artifact_id)
        assert result.output["loss"] > 0
    finally:
        requester_work.close()
        worker_work.close()
        requester.stop()
        worker.stop()
