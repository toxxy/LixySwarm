"""Apply distinct, authenticated shards to an isolated evaluated candidate.

This productive training path is separate from repeated-shard verification.
The caller explicitly pins pilot workers and supplies model/evaluation code;
peers cannot provide executable model implementations.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.release import ReleaseManifest
from .artifact_store import ArtifactManifest, digest_file
from .telemetry import atomic_json
from .training_worker import TrainingWorkError, validate_gradient_artifact
from .work_protocol import ResultReceipt, WorkResult


@dataclass(frozen=True)
class ShardContribution:
    peer_id: str
    start_token: int
    token_count: int
    result: WorkResult
    gradient_path: Path


@dataclass
class LearningOutcome:
    accepted: bool
    before_loss: float
    after_loss: float
    predicted_tokens: int
    release: ReleaseManifest | None
    model: object
    report: dict


class DistributedLearningCycle:
    def __init__(self, *, store, registry, policy, identity, trusted_workers,
                 model_factory, optimizer_factory, evaluator, evaluation_id,
                 state_path: Path, minimum_workers=3, minimum_improvement=0.001,
                 max_gradient_norm=1.0, requester_node_id=None, promote=True):
        self.store, self.registry, self.policy = store, registry, policy
        self.identity = identity
        self.requester_node_id = requester_node_id or identity.node_id_hex
        self.promote = bool(promote)
        self.trusted_workers = set(trusted_workers)
        self.model_factory, self.optimizer_factory = model_factory, optimizer_factory
        self.evaluator, self.evaluation_id = evaluator, evaluation_id
        self.state_path = Path(state_path)
        if not 2 <= minimum_workers <= 31 or len(self.trusted_workers) < minimum_workers:
            raise ValueError("Distributed learning needs enough explicitly trusted workers")
        if not math.isfinite(minimum_improvement) or minimum_improvement <= 0:
            raise ValueError("Minimum evaluation improvement must be positive")
        if not math.isfinite(max_gradient_norm) or max_gradient_norm <= 0:
            raise ValueError("Gradient norm limit must be positive")
        self.minimum_workers = minimum_workers
        self.minimum_improvement = minimum_improvement
        self.max_gradient_norm = max_gradient_norm
        self.recover()

    def recover(self):
        if not self.promote:
            return
        if not self.state_path.exists():
            return
        state = json.loads(self.state_path.read_text())
        if state.get("status") != "promotion_pending":
            return
        release = ReleaseManifest.from_dict(state["release"])
        self.policy.authorize(release)
        self.registry.accept(release, self.policy, self.store)
        active = self.registry.active()
        if active is None or active.release_id != release.release_id:
            self.registry.activate(release.release_id, self.policy, self.store)
        atomic_json(self.state_path, {**state, "status": "accepted"})

    def run(self, *, base_release: ReleaseManifest, dataset_id: str,
            contributions: list[ShardContribution]) -> LearningOutcome:
        self.recover()
        active = self.registry.verify_active(self.policy, self.store)
        if active.release_id != base_release.release_id:
            raise TrainingWorkError("Contributions do not extend the active model version")
        if not self.minimum_workers <= len(contributions) <= 31:
            raise TrainingWorkError("Not enough distinct productive workers")
        if len({item.peer_id for item in contributions}) != len(contributions):
            raise TrainingWorkError("Productive contributions require distinct worker identities")
        if len({item.result.job_id for item in contributions}) != len(contributions):
            raise TrainingWorkError("A work result cannot be counted twice")
        ranges = sorted((item.start_token, item.start_token + item.token_count) for item in contributions)
        if any(first[1] > second[0] for first, second in zip(ranges, ranges[1:])):
            raise TrainingWorkError("Productive training shards must not overlap")

        model_id = base_release.model_artifact_id
        checkpoint = torch.load(self.store._object_path(model_id), map_location="cpu", weights_only=True)
        model = self.model_factory(checkpoint["config"])
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer = self.optimizer_factory(model)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

        gradient_names = None
        records = []
        for contribution in contributions:
            if contribution.peer_id not in self.trusted_workers:
                raise TrainingWorkError("Worker is not authorized by the local pilot policy")
            if type(contribution.start_token) is not int or type(contribution.token_count) is not int or contribution.start_token < 0 or contribution.token_count < 2:
                raise TrainingWorkError("Invalid productive token range")
            result = contribution.result
            result.validate()
            if result.status != "ok" or not result.receipt or not ResultReceipt.from_dict(result.receipt).verify(
                    result, expected_worker_id=contribution.peer_id,
                    expected_requester_id=self.requester_node_id):
                raise TrainingWorkError("Productive work receipt is invalid")
            manifest = ArtifactManifest.from_dict(result.output.get("gradient"))
            actual_id, actual_size = digest_file(contribution.gradient_path)
            if actual_id != manifest.artifact_id or actual_size != manifest.size or manifest.kind != "gradient":
                raise TrainingWorkError("Productive gradient does not match its signed receipt")
            if (result.output.get("model_artifact_id") != model_id
                    or result.output.get("dataset_artifact_id") != dataset_id
                    or result.output.get("token_count") != contribution.token_count):
                raise TrainingWorkError("Productive result does not match its assigned inputs")
            validate_gradient_artifact(
                contribution.gradient_path, model, expected_model_id=model_id,
                expected_dataset_id=dataset_id, expected_start_token=contribution.start_token,
                expected_token_count=contribution.token_count,
            )
            with np.load(contribution.gradient_path, allow_pickle=False) as arrays:
                names = set(arrays.files) - {"__lixy_metadata__"}
            if gradient_names is None:
                gradient_names = names
            elif gradient_names != names:
                raise TrainingWorkError("Productive gradients contain different parameter sets")
            records.append({
                "peer_id": contribution.peer_id, "job_id": result.job_id,
                "gradient_id": manifest.artifact_id, "start_token": contribution.start_token,
                "token_count": contribution.token_count, "receipt": result.receipt,
            })

        predicted_tokens = sum(item.token_count - 1 for item in contributions)
        model.eval()
        before_loss = float(self.evaluator(model))
        if not math.isfinite(before_loss):
            raise TrainingWorkError("Base evaluation is non-finite")
        parameters = dict(model.named_parameters())
        optimizer.zero_grad(set_to_none=True)
        # Accumulate one shard at a time to bound temporary memory.
        for contribution in contributions:
            weight = (contribution.token_count - 1) / predicted_tokens
            with np.load(contribution.gradient_path, allow_pickle=False) as arrays:
                for name in sorted(gradient_names):
                    parameter = parameters[name]
                    gradient = torch.from_numpy(arrays[name].copy()).to(parameter.device, dtype=parameter.dtype)
                    if parameter.grad is None:
                        parameter.grad = gradient.mul(weight)
                    else:
                        parameter.grad.add_(gradient, alpha=weight)
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_gradient_norm, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        after_loss = float(self.evaluator(model))
        accepted = math.isfinite(after_loss) and before_loss - after_loss >= self.minimum_improvement
        report = {
            "aggregation": "prediction-token-weighted-mean", "evaluation_id": self.evaluation_id,
            "before_loss": before_loss, "after_loss": after_loss if math.isfinite(after_loss) else None,
            "minimum_improvement": self.minimum_improvement, "predicted_tokens": predicted_tokens,
            "base_release_id": base_release.release_id, "dataset_id": dataset_id,
            "contributions": records, "accepted": accepted,
        }
        release = None
        if accepted:
            temporary = self.store.incoming / f"candidate.{os.urandom(8).hex()}.pt"
            try:
                torch.save({"format": 1, "config": checkpoint["config"],
                            "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                            "optimizer": optimizer.state_dict(), "learning_report": report}, temporary)
                artifact = self.store.commit_generated(temporary, kind="model", media_type="application/x-pytorch-state-dict")
            finally:
                temporary.unlink(missing_ok=True)
            report_path = self.store.incoming / f"evaluation.{os.urandom(8).hex()}.json"
            try:
                atomic_json(report_path, {"model_config": checkpoint["config"], "learning": report})
                config_artifact = self.store.commit_generated(report_path, kind="evaluation", media_type="application/json")
            finally:
                report_path.unlink(missing_ok=True)
            release = ReleaseManifest.create(
                model_artifact_id=artifact.artifact_id, model_format="pytorch-weights-only-v1",
                sequence=base_release.sequence + 1, previous_release_id=base_release.release_id,
                config_artifact_id=config_artifact.artifact_id,
                tokenizer_artifact_id=base_release.tokenizer_artifact_id,
            )
            if self.promote:
                release=release.sign(self.identity)
                self.policy.authorize(release)
                pending = {"status": "promotion_pending", "release": release.to_dict(), "report": report}
                atomic_json(self.state_path, pending)
                self.registry.accept(release, self.policy, self.store)
                self.registry.activate(release.release_id, self.policy, self.store)
                atomic_json(self.state_path, {**pending, "status": "accepted"})
        else:
            atomic_json(self.state_path, {"status": "rejected", "report": report})
        return LearningOutcome(accepted, before_loss, after_loss, predicted_tokens, release, model, report)
