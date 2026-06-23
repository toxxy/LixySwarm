import json

import numpy as np
import pytest
import torch

from src.network.artifact_store import ArtifactStore, digest_file
from src.network.gradient_aggregation import GradientAggregator, GradientCandidate
from src.network.training_worker import TrainingWorkError


def _candidate(tmp_path, index, value, model_id, dataset_id):
    path = tmp_path / f"candidate-{index}.npz"
    metadata = json.dumps({
        "format": 1,
        "model_artifact_id": model_id,
        "dataset_artifact_id": dataset_id,
        "start_token": 0,
        "token_count": 8,
    }, sort_keys=True, separators=(",", ":")).encode()
    np.savez_compressed(
        path,
        weight=np.full((2, 3), value, dtype=np.float32),
        __lixy_metadata__=np.frombuffer(metadata, dtype=np.uint8),
    )
    artifact_id, _ = digest_file(path)
    return GradientCandidate(
        artifact_id=artifact_id,
        peer_id=f"{index + 1:02x}" * 32,
        path=path,
    )


def test_coordinate_median_aggregation_resists_one_outlier(tmp_path):
    model = torch.nn.Linear(3, 2, bias=False)
    model_id = "ab" * 32
    dataset_id = "cd" * 32
    candidates = [
        _candidate(tmp_path, 0, 1.0, model_id, dataset_id),
        _candidate(tmp_path, 1, 2.0, model_id, dataset_id),
        _candidate(tmp_path, 2, 1000.0, model_id, dataset_id),
    ]
    store = ArtifactStore(tmp_path / "store")
    manifest, metadata = GradientAggregator(
        store, chunk_values=2
    ).aggregate(
        candidates,
        model,
        model_artifact_id=model_id,
        dataset_artifact_id=dataset_id,
        start_token=0,
        token_count=8,
    )
    assert manifest.kind == "gradient"
    assert metadata["quorum"] == 3
    assert metadata["applied"] is False
    with np.load(store._object_path(manifest.artifact_id), allow_pickle=False) as value:
        assert np.all(value["weight"] == 2.0)
        aggregate_metadata = json.loads(
            value["__lixy_metadata__"].tobytes().decode()
        )
    assert aggregate_metadata["aggregation"] == "coordinate_median"


def test_gradient_quorum_requires_distinct_peers(tmp_path):
    model = torch.nn.Linear(3, 2, bias=False)
    model_id = "ab" * 32
    dataset_id = "cd" * 32
    candidates = [
        _candidate(tmp_path, index, float(index), model_id, dataset_id)
        for index in range(3)
    ]
    candidates[2] = GradientCandidate(
        candidates[2].artifact_id,
        candidates[1].peer_id,
        candidates[2].path,
    )
    with pytest.raises(TrainingWorkError, match="distinct"):
        GradientAggregator(ArtifactStore(tmp_path / "store")).aggregate(
            candidates,
            model,
            model_artifact_id=model_id,
            dataset_artifact_id=dataset_id,
            start_token=0,
            token_count=8,
        )
