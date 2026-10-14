"""Productive distinct shards, quality gates and durable promotion recovery."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from src.network.artifact_store import ArtifactStore
from src.network.distributed_learning import DistributedLearningCycle, ShardContribution
from src.network.lsp import LSPIdentity
from src.network.training_worker import TrainingWorkError
from src.network.work_protocol import ResultReceipt, WorkResult
from src.release import ReleaseManifest, ReleaseRegistry, TrustPolicy


def setup(tmp_path):
    identity = LSPIdentity.generate()
    workers = [LSPIdentity.generate(), LSPIdentity.generate()]
    store = ArtifactStore(tmp_path / "artifacts")
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    initial = tmp_path / "base.pt"
    torch.save({"format": 1, "config": {"family": "test-linear"}, "model": model.state_dict()}, initial)
    artifact = store.import_file(initial, kind="model")
    release = ReleaseManifest.create(model_artifact_id=artifact.artifact_id,
                                      model_format="pytorch-weights-only-v1", sequence=0).sign(identity)
    policy = TrustPolicy(trusted_signers=(identity.node_id_hex,), pinned_genesis_release_id=release.release_id)
    registry = ReleaseRegistry(tmp_path / "releases")
    registry.accept(release, policy, store); registry.activate(release.release_id, policy, store)
    args = dict(store=store, registry=registry, policy=policy, identity=identity,
                trusted_workers=[w.node_id_hex for w in workers], minimum_workers=2,
                model_factory=lambda _: torch.nn.Linear(1, 1, bias=False),
                optimizer_factory=lambda m: torch.optim.SGD(m.parameters(), lr=0.1),
                evaluator=lambda m: (m(torch.ones(1, 1)) - 1).square().mean().item(),
                evaluation_id="ef" * 32, state_path=tmp_path / "learning-state.json", max_gradient_norm=10)
    return args, release, workers


def shard(tmp_path, args, release, worker, *, start, count, gradient, index):
    dataset_id = "dc" * 32
    metadata = {"format": 1, "model_artifact_id": release.model_artifact_id,
                "dataset_artifact_id": dataset_id, "start_token": start, "token_count": count}
    path = tmp_path / f"gradient-{index}.npz"
    np.savez_compressed(path, weight=np.array([[gradient]], dtype=np.float32),
                        __lixy_metadata__=np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8))
    artifact = args["store"].import_file(path, kind="gradient", media_type="application/x-npz")
    result = WorkResult(job_id=f"{index + 1:02x}" * 32, status="ok", output={
        "gradient": artifact.to_dict(), "loss": 1.0, "gradient_norm": abs(gradient),
        "model_artifact_id": release.model_artifact_id, "dataset_artifact_id": dataset_id, "token_count": count})
    result = replace(result, receipt=ResultReceipt.create(result, worker, args["identity"].node_id_hex).to_dict())
    return ShardContribution(worker.node_id_hex, start, count, result, path)


def contributions(tmp_path, args, release, workers, sign=-1):
    return [shard(tmp_path, args, release, workers[0], start=0, count=5, gradient=sign, index=0),
            shard(tmp_path, args, release, workers[1], start=5, count=9, gradient=sign*3, index=1)]


def test_distinct_shards_update_weights_and_publish_a_better_version(tmp_path):
    args, base, workers = setup(tmp_path)
    outcome = DistributedLearningCycle(**args).run(base_release=base, dataset_id="dc"*32,
                                                  contributions=contributions(tmp_path, args, base, workers))
    assert outcome.accepted and outcome.after_loss < outcome.before_loss
    assert outcome.predicted_tokens == 12
    assert outcome.model.weight.item() == pytest.approx(0.1 * (4 + 24) / 12)
    assert args["registry"].active().release_id == outcome.release.release_id
    assert outcome.release.previous_release_id == base.release_id
    assert outcome.release.valid_signers() == {args["identity"].node_id_hex}
    # The base and its parameters are retained unchanged.
    checkpoint = torch.load(args["store"]._object_path(base.model_artifact_id), weights_only=True)
    assert checkpoint["model"]["weight"].item() == 0


def test_quality_regression_preserves_the_accepted_model(tmp_path):
    args, base, workers = setup(tmp_path)
    outcome = DistributedLearningCycle(**args).run(base_release=base, dataset_id="dc"*32,
                                                  contributions=contributions(tmp_path, args, base, workers, sign=1))
    assert not outcome.accepted and outcome.release is None
    assert outcome.after_loss > outcome.before_loss
    assert args["registry"].active().release_id == base.release_id
    assert json.loads(args["state_path"].read_text())["status"] == "rejected"


def test_repeated_verification_range_is_not_productive_training(tmp_path):
    args, base, workers = setup(tmp_path)
    items = [shard(tmp_path, args, base, worker, start=0, count=5, gradient=-1, index=i)
             for i, worker in enumerate(workers)]
    with pytest.raises(TrainingWorkError, match="overlap"):
        DistributedLearningCycle(**args).run(base_release=base, dataset_id="dc"*32, contributions=items)
    assert args["registry"].active().release_id == base.release_id


def test_result_receipt_binds_inputs_gradient_and_worker(tmp_path):
    args, base, workers = setup(tmp_path)
    items = contributions(tmp_path, args, base, workers)
    items[0].result.output["token_count"] = 1000
    with pytest.raises(TrainingWorkError, match="receipt"):
        DistributedLearningCycle(**args).run(base_release=base, dataset_id="dc"*32, contributions=items)


def test_pending_promotion_recovers_after_crash_without_reapplying_gradients(tmp_path, monkeypatch):
    args, base, workers = setup(tmp_path)
    cycle = DistributedLearningCycle(**args)
    def crash(*_):
        raise RuntimeError("simulated interruption before activation")
    monkeypatch.setattr(args["registry"], "activate", crash)
    with pytest.raises(RuntimeError, match="interruption"):
        cycle.run(base_release=base, dataset_id="dc"*32, contributions=contributions(tmp_path, args, base, workers))
    pending = json.loads(args["state_path"].read_text())
    assert pending["status"] == "promotion_pending"
    args["registry"] = ReleaseRegistry(tmp_path / "releases")
    recovered = DistributedLearningCycle(**args)
    assert args["registry"].active().release_id == pending["release"]["release_id"]
    assert json.loads(args["state_path"].read_text())["status"] == "accepted"
    with pytest.raises(TrainingWorkError, match="active model version"):
        recovered.run(base_release=base, dataset_id="dc"*32, contributions=contributions(tmp_path, args, base, workers))
