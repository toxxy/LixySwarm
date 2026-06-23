"""Verified-input gradient work for an already loaded local model.

The worker never loads peer-provided executable checkpoints. It accepts only a
content-addressed NumPy token dataset, computes gradients against the exact
locally configured model version, and returns a content-addressed NPZ artifact.
Gradients are candidates: this module never applies or aggregates them.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from .artifact_store import ArtifactService, ArtifactStore


TRAINING_OPERATION = "training.gradient.v1"
MAX_BATCH_TOKENS = 4096


class TrainingWorkError(ValueError):
    pass


def validate_gradient_artifact(
    path: str | Path,
    model,
    *,
    expected_model_id: str,
    expected_dataset_id: str,
    expected_start_token: int,
    expected_token_count: int,
) -> dict:
    """Validate an untrusted NPZ candidate without applying it."""
    path = Path(path)
    parameters = dict(model.named_parameters())
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if not entries or len(entries) > len(parameters) + 1:
                raise TrainingWorkError("gradient archive entry count is invalid")
            for entry in entries:
                if entry.is_dir() or "/" in entry.filename or not entry.filename.endswith(".npy"):
                    raise TrainingWorkError("gradient archive contains an invalid entry")
                name = entry.filename[:-4]
                if name == "__lixy_metadata__":
                    maximum = 8192
                elif name in parameters:
                    maximum = parameters[name].numel() * 4 + 4096
                else:
                    raise TrainingWorkError("gradient archive contains an unknown parameter")
                if entry.file_size > maximum:
                    raise TrainingWorkError("gradient archive entry exceeds its expected size")
    except (OSError, zipfile.BadZipFile) as exc:
        raise TrainingWorkError("gradient artifact is not a valid NPZ archive") from exc

    try:
        with np.load(path, allow_pickle=False) as values:
            if "__lixy_metadata__" not in values.files:
                raise TrainingWorkError("gradient metadata is missing")
            raw_metadata = values["__lixy_metadata__"]
            if raw_metadata.dtype != np.uint8 or raw_metadata.ndim != 1:
                raise TrainingWorkError("gradient metadata encoding is invalid")
            metadata = json.loads(raw_metadata.tobytes().decode("utf-8"))
            if not isinstance(metadata, dict) or set(metadata) != {
                "dataset_artifact_id", "format", "model_artifact_id",
                "start_token", "token_count",
            }:
                raise TrainingWorkError("gradient metadata schema is invalid")
            if metadata != {
                "dataset_artifact_id": expected_dataset_id,
                "format": 1,
                "model_artifact_id": expected_model_id,
                "start_token": int(expected_start_token),
                "token_count": int(expected_token_count),
            }:
                raise TrainingWorkError("gradient metadata does not match the request")
            if not isinstance(metadata["start_token"], int) or metadata["start_token"] < 0:
                raise TrainingWorkError("gradient metadata token range is invalid")
            if not isinstance(metadata["token_count"], int) or metadata["token_count"] < 2:
                raise TrainingWorkError("gradient metadata token range is invalid")
            gradient_names = [
                name for name in values.files if name != "__lixy_metadata__"
            ]
            if not gradient_names:
                raise TrainingWorkError("gradient artifact contains no gradients")
            for name in gradient_names:
                array = values[name]
                parameter = parameters.get(name)
                if (
                    parameter is None
                    or array.dtype != np.float32
                    or tuple(array.shape) != tuple(parameter.shape)
                    or not np.isfinite(array).all()
                ):
                    raise TrainingWorkError("gradient tensor does not match the local model")
    except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        if isinstance(exc, TrainingWorkError):
            raise
        raise TrainingWorkError("gradient artifact payload is invalid") from exc
    return {
        "model_artifact_id": expected_model_id,
        "dataset_artifact_id": expected_dataset_id,
        "gradient_count": len(gradient_names),
        "start_token": int(expected_start_token),
        "token_count": int(expected_token_count),
    }


class TrainingWorker:
    """Compute bounded gradient artifacts without mutating model weights."""

    def __init__(
        self,
        coordinator,
        artifact_service: ArtifactService,
        model,
        *,
        model_artifact_id: str,
        device: str,
        execution_lock: threading.RLock | threading.Lock | None = None,
        max_batch_tokens: int = MAX_BATCH_TOKENS,
    ):
        if len(model_artifact_id) != 64:
            raise ValueError("model_artifact_id must be a SHA-256 identifier")
        self.coordinator = coordinator
        self.artifact_service = artifact_service
        self.store: ArtifactStore = artifact_service.store
        self.model = model
        self.model_artifact_id = model_artifact_id
        self.device = device
        self.execution_lock = execution_lock or threading.RLock()
        self.max_batch_tokens = max(2, min(int(max_batch_tokens), MAX_BATCH_TOKENS))
        coordinator.register_handler(TRAINING_OPERATION, "training", self._handle)

    def _handle(self, payload: dict, work) -> dict:
        request = self._validate(payload)
        if request["model_artifact_id"] != self.model_artifact_id:
            raise TrainingWorkError("worker model version does not match")
        estimated_bytes = sum(
            parameter.numel() * 4
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        estimated_gb = estimated_bytes / (1024 ** 3)
        requirements = work.resource_requirements()
        if requirements.ram_gb + 1e-9 < estimated_gb:
            raise TrainingWorkError("declared RAM is below the gradient estimate")
        if requirements.disk_gb + 1e-9 < estimated_gb:
            raise TrainingWorkError("declared disk is below the gradient estimate")
        if self.store.total_bytes() + estimated_bytes > self.store.max_total_bytes:
            raise TrainingWorkError("artifact quota cannot hold the gradient")
        dataset_id = request["dataset_artifact_id"]
        if not self.store.has(dataset_id):
            self.artifact_service.fetch(
                dataset_id,
                peer_id=work.origin_node_id,
                timeout_s=max(1.0, min(300.0, work.deadline - time.time())),
            )
        dataset_manifest = self.store.manifest(dataset_id)
        if (
            dataset_manifest.kind != "dataset"
            or dataset_manifest.media_type != "application/x-npy"
        ):
            raise TrainingWorkError("training dataset must be an application/x-npy artifact")
        dataset_path = self.store._object_path(dataset_id)
        try:
            tokens = np.load(dataset_path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise TrainingWorkError("dataset is not a safe NumPy array") from exc
        if tokens.ndim != 1 or tokens.dtype.kind not in {"i", "u"}:
            raise TrainingWorkError("dataset must be a one-dimensional integer token array")
        start = request["start_token"]
        count = request["token_count"]
        if start + count > int(tokens.shape[0]):
            raise TrainingWorkError("requested token range exceeds the dataset")
        batch = np.asarray(tokens[start:start + count], dtype=np.int64)
        vocab_size = int(self.model.config.agent_configs[0].vocab_size)
        if batch.size < 2 or batch.min() < 0 or batch.max() >= vocab_size:
            raise TrainingWorkError("dataset contains tokens outside the model vocabulary")

        x = torch.from_numpy(batch[:-1].copy()).long().unsqueeze(0).to(self.device)
        y = torch.from_numpy(batch[1:].copy()).long().unsqueeze(0).to(self.device)
        with self.execution_lock:
            result = self._compute_gradient(x, y, dataset_id, start, count)
        return result

    def _compute_gradient(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        dataset_id: str,
        start: int,
        count: int,
    ) -> dict:
        was_training = self.model.training
        temporary = self.store.incoming / (
            f"gradient.{os.getpid()}.{os.urandom(8).hex()}.npz.tmp"
        )
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        try:
            _, loss, _ = self.model(
                x,
                targets=y,
                store_memory=False,
                update_runtime_state=False,
                update_memory_importance=False,
                use_memory=False,
            )
            if not isinstance(loss, torch.Tensor) or not torch.isfinite(loss):
                raise TrainingWorkError("model returned a non-finite loss")
            loss.backward()
            gradients = {}
            squared_norm = 0.0
            for name, parameter in self.model.named_parameters():
                if parameter.grad is None:
                    continue
                gradient_tensor = parameter.grad.detach().float()
                squared_norm += float(
                    gradient_tensor.square().sum().cpu().item()
                )
                gradient = gradient_tensor.cpu().numpy()
                if not np.isfinite(gradient).all():
                    raise TrainingWorkError("model produced non-finite gradients")
                gradients[name] = gradient
            if not gradients:
                raise TrainingWorkError("model produced no gradients")
            metadata = json.dumps({
                "format": 1,
                "model_artifact_id": self.model_artifact_id,
                "dataset_artifact_id": dataset_id,
                "start_token": start,
                "token_count": count,
            }, sort_keys=True, separators=(",", ":")).encode("utf-8")
            gradients["__lixy_metadata__"] = np.frombuffer(metadata, dtype=np.uint8)
            with temporary.open("xb") as destination:
                np.savez_compressed(destination, **gradients)
                destination.flush()
                os.fsync(destination.fileno())
            manifest = self.store.commit_generated(
                temporary,
                kind="gradient",
                media_type="application/x-npz",
            )
            return {
                "gradient": manifest.to_dict(),
                "loss": float(loss.detach().cpu().item()),
                "gradient_norm": math.sqrt(squared_norm),
                "model_artifact_id": self.model_artifact_id,
                "dataset_artifact_id": dataset_id,
                "token_count": count,
            }
        finally:
            temporary.unlink(missing_ok=True)
            self.model.zero_grad(set_to_none=True)
            self.model.train(was_training)

    def _validate(self, payload: dict) -> dict:
        if not isinstance(payload, dict) or set(payload) != {
            "model_artifact_id", "dataset_artifact_id", "start_token", "token_count"
        }:
            raise TrainingWorkError("training request fields do not match the schema")
        model_id = str(payload["model_artifact_id"])
        dataset_id = str(payload["dataset_artifact_id"])
        if any(
            len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
            for value in (model_id, dataset_id)
        ):
            raise TrainingWorkError("training artifact identifier is invalid")
        try:
            start = int(payload["start_token"])
            count = int(payload["token_count"])
        except (TypeError, ValueError) as exc:
            raise TrainingWorkError("training token range is invalid") from exc
        if start < 0 or not 2 <= count <= self.max_batch_tokens:
            raise TrainingWorkError("training token range is out of bounds")
        return {
            "model_artifact_id": model_id,
            "dataset_artifact_id": dataset_id,
            "start_token": start,
            "token_count": count,
        }
