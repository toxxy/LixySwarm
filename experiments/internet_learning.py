"""Real learning pilot: bounded public byte swarm, two known physical PCs.

All model/data/gradient transfer uses LSP. State must be outside the checkout.
The VPS serves only as discovery/relay. This does not enable unknown workers.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from client_daemon import hardware_resources
from src.contribution import ContributionPolicy, ResourceGovernor, ResourceRequirements
from src.network.artifact_store import ArtifactService, ArtifactStore
from src.network.bootstrap import get_seed_endpoints
from src.network.distributed_learning import DistributedLearningCycle, ShardContribution
from src.network.lsp import LSPIdentity
from src.network.lsp_v3 import LSPNodeV3
from src.network.reference_model import (
    REFERENCE_CONFIG, evaluate_reference, load_reference_checkpoint,
    make_reference_model, reference_corpus, reference_logits,
)
from src.network.telemetry import atomic_json, build_status, sign_status
from src.network.training_worker import TRAINING_OPERATION, TrainingWorker
from src.network.work_protocol import ResultReceipt, WorkCoordinator
from src.release import ReleaseManifest, ReleaseRegistry, TrustPolicy
from src.swarm.node_manager import HardwareProfile


def report(event, **values):
    print(json.dumps({"event": event, **values}, allow_nan=False), flush=True)


def identity_at(state):
    path = state / "identity.key"
    identity = LSPIdentity.load(str(path))
    if identity is None:
        identity = LSPIdentity.generate(); identity.save(str(path))
    return identity


def wait_for(predicate, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.2)
    raise RuntimeError("Pilot peer did not become ready before its deadline")


class Pilot:
    def __init__(self, args):
        self.args = args
        self.state = args.state_dir.resolve()
        repo = Path(__file__).resolve().parent.parent
        if self.state == repo or repo in self.state.parents:
            raise ValueError("Pilot state must be outside the code/Dropbox checkout")
        self.state.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.identity = identity_at(self.state)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.resources = hardware_resources()
        self.policy = ContributionPolicy.for_mode("balanced")
        self.policy.save(self.state / "pilot-policy.json")
        # The user activated the Internet-learning plan. This scoped pilot
        # authorizes these known machines without changing app consent files.
        torch.set_num_threads(max(1, min(4, int(self.resources["cpu_cores"] * 0.25))))
        if self.device == "cuda":
            torch.cuda.set_per_process_memory_fraction(0.25)
        hardware = HardwareProfile(cpu_cores=self.resources["cpu_cores"], ram_gb=self.resources["ram_gb"],
            gpu_vram_gb=self.resources["gpu_vram_gb"], has_gpu=self.resources["has_gpu"], disk_gb=10)
        self.governor = ResourceGovernor(self.policy, hardware=hardware, storage_path=self.state)
        self.node = LSPNodeV3(self.identity, host="0.0.0.0", port=args.port,
            seeds=get_seed_endpoints(), address_book_path=self.state / "peers.json", target_outbound=6,
            capabilities={"peer_exchange": True, "pheromone": False, "global_memory": False,
                          "learning_pilot": True},
            resource_profile=self.governor.advertised_profile(available_work={"artifact", "training", "inference"}))
        self.work = WorkCoordinator(self.node, self.governor, max_workers=3,
                                    max_offers_per_peer=3, max_offers_per_minute=240)
        self.store = ArtifactStore(self.state / "artifacts", max_artifact_bytes=32*1024**2,
                                   max_total_bytes=512*1024**2)
        self.artifacts = ArtifactService(self.work, self.store)
        self.registry = ReleaseRegistry(self.state / "releases")
        self.lock = threading.RLock()
        self.model = None
        self.trainer = None
        self.training_jobs = 0
        self.stop = threading.Event()
        self.started = time.monotonic()
        status_path = self.state / "pilot-status.json"
        self.status_sequence = json.loads(status_path.read_text()).get("status_sequence", 0) if status_path.exists() else 0

    def status(self):
        active = self.governor.status()["active_cpu_jobs"]
        self.status_sequence += 1
        value = build_status(self.node, sequence=self.status_sequence, role="participant",
            state="working" if active else "available", uptime_s=time.monotonic()-self.started,
            resources=self.resources)
        value["jobs"] = {"active": active, "completed": self.training_jobs}
        envelope = sign_status(self.identity, value)
        for peer in self.node.peers():
            if peer.get("capabilities", {}).get("status_collector_v1"):
                self.node.send_node_status(peer["node_id"], envelope)
        atomic_json(self.state / "pilot-status.json", {
            "identity": self.identity.node_id_hex, "role": self.args.role, "device": self.device,
            "model_id": self.trainer.model_artifact_id if self.trainer else None,
            "training_jobs": self.training_jobs, "peers": self.node.peers(),
            "status_sequence": self.status_sequence,
        })

    def prepare(self, payload, execution):
        if execution.origin_node_id != self.args.coordinator_id or set(payload) != {"release"}:
            raise ValueError("Only the pinned pilot coordinator may prepare models")
        release = ReleaseManifest.from_dict(payload["release"])
        policy = TrustPolicy(trusted_signers=(self.args.coordinator_id,))
        policy.authorize(release)
        with self.lock:
            for identifier in (release.model_artifact_id, release.config_artifact_id, release.tokenizer_artifact_id):
                if identifier:
                    self.artifacts.fetch(identifier, peer_id=execution.origin_node_id, timeout_s=120,
                                         cancel_event=execution.cancel_event)
            model = load_reference_checkpoint(self.store._object_path(release.model_artifact_id), self.device)
            active = self.registry.active()
            self.registry.accept(release, policy, self.store)
            if active is None or active.release_id != release.release_id:
                self.registry.activate(release.release_id, policy, self.store)
            self.model = model
            self.trainer = TrainingWorker(self.work, self.artifacts, model,
                model_artifact_id=release.model_artifact_id, device=self.device, execution_lock=self.lock,
                max_batch_tokens=65)
            self.work.register_handler(TRAINING_OPERATION, "training", self.gradient)
        return {"model_id": release.model_artifact_id, "release_id": release.release_id, "device": self.device}

    def gradient(self, payload, execution):
        if execution.origin_node_id != self.args.coordinator_id or self.trainer is None:
            raise ValueError("Training is restricted to the known pilot coordinator")
        with self.lock:
            result = self.trainer._handle(payload, execution)
            self.training_jobs += 1
            return result

    def inference(self, payload, execution):
        if execution.origin_node_id != self.args.coordinator_id or set(payload) != {"prompt"}:
            raise ValueError("Inference is restricted to the pilot")
        text = payload["prompt"]
        if not isinstance(text, str) or not 1 <= len(text.encode()) <= 256 or self.model is None:
            raise ValueError("Invalid reference inference input")
        with self.lock:
            return {"model_id": self.trainer.model_artifact_id,
                    "next_byte_logits": reference_logits(self.model, text.encode("utf-8"))}

    def checked(self, operation, payload, peer, kind, timeout=120):
        result = self.work.submit(operation, payload,
            ResourceRequirements(kind=kind, gpu_required=self.device=="cuda" and kind in {"training", "memory"},
                                 ram_gb=0.1, disk_gb=0.1), peer_id=peer, timeout_s=timeout)
        if result.status != "ok":
            raise RuntimeError(f"{operation}: {result.error}")
        if not ResultReceipt.from_dict(result.receipt).verify(result, expected_worker_id=peer,
                                                             expected_requester_id=self.identity.node_id_hex):
            raise RuntimeError("Pilot result receipt is invalid")
        return result

    def coordinator(self):
        workers = self.args.worker_ids.split(",")
        if len(workers) != 2 or len(set(workers)) != 2:
            raise ValueError("This first pilot requires exactly two pinned physical workers")
        wait_for(lambda: set(workers) <= self.node.connected_node_ids)
        training, evaluation = reference_corpus()
        train_path = self.state / "public-tokens.npy"; np.save(train_path, training, allow_pickle=False)
        dataset = self.store.import_file(train_path, kind="dataset", media_type="application/x-npy")
        evaluation_id = hashlib.sha256(evaluation.tobytes()).hexdigest()
        active = self.registry.active()
        policy = TrustPolicy(trusted_signers=(self.identity.node_id_hex,))
        if active is None:
            torch.manual_seed(20261014)
            model = make_reference_model(REFERENCE_CONFIG, self.device)
            checkpoint_path = self.state / "reference-base.pt"
            torch.save({"format": 1, "config": REFERENCE_CONFIG,
                        "model": {k:v.detach().cpu() for k,v in model.state_dict().items()}}, checkpoint_path)
            initial = self.store.import_file(checkpoint_path, kind="model", media_type="application/x-pytorch-state-dict")
            active = ReleaseManifest.create(model_artifact_id=initial.artifact_id,
                model_format="pytorch-weights-only-v1", sequence=0).sign(self.identity)
            self.registry.accept(active, policy, self.store); self.registry.activate(active.release_id, policy, self.store)
        cycle = DistributedLearningCycle(store=self.store, registry=self.registry, policy=policy,
            identity=self.identity, trusted_workers=workers, minimum_workers=2,
            model_factory=lambda cfg: make_reference_model(cfg, self.device),
            optimizer_factory=lambda model: torch.optim.AdamW(model.parameters(), lr=0.001),
            evaluator=lambda model: evaluate_reference(model, evaluation), evaluation_id=evaluation_id,
            state_path=self.state / "learning-state.json", minimum_improvement=0.001)
        rounds = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            for index in range(self.args.cycles):
                active = self.registry.active()
                def contribution(pair):
                    position, peer = pair
                    self.checked("learning.prepare.v1", {"release": active.to_dict()}, peer, "memory")
                    start = position * (len(training)//2) + (index*65) % (len(training)//2-65)
                    result = self.checked(TRAINING_OPERATION, {
                        "model_artifact_id": active.model_artifact_id, "dataset_artifact_id": dataset.artifact_id,
                        "start_token": start, "token_count": 65}, peer, "training")
                    path = self.artifacts.fetch(result.output["gradient"]["artifact_id"], peer_id=peer, timeout_s=120)
                    return ShardContribution(peer, start, 65, result, path)
                contributions = list(pool.map(contribution, enumerate(workers)))
                outcome = cycle.run(base_release=active, dataset_id=dataset.artifact_id, contributions=contributions)
                rounds.append(outcome.report)
                report("learning", round=index, accepted=outcome.accepted, before=outcome.before_loss,
                       after=outcome.after_loss, predicted_tokens=outcome.predicted_tokens)
                if outcome.accepted:
                    self.node.announce_release(outcome.release.to_dict())
                # Keep the reference bursts separate under the existing LSP
                # byte window. General bandwidth enforcement is a later gate.
                time.sleep(15)
        final = self.registry.active()
        for peer in workers:
            self.checked("learning.prepare.v1", {"release": final.to_dict()}, peer, "memory")
        left = self.checked("reference.inference.v1", {"prompt":"La memoria personal "}, workers[0], "inference")
        right = self.checked("reference.inference.v1", {"prompt":"La memoria personal "}, workers[1], "inference")
        max_difference = max(abs(a-b) for a,b in zip(left.output["next_byte_logits"], right.output["next_byte_logits"]))
        success = final.sequence > 0 and max_difference < 0.001
        result = {"success":success, "rounds":rounds, "accepted_sequence":final.sequence,
                  "final_release":final.to_dict(), "inference_max_difference":max_difference,
                  "replicas":workers, "device":self.device,
                  "routes":[{"node_id":p["node_id"],"transport":p["transport"],"host":p["host"],
                             "relay_host":p.get("relay_host")} for p in self.node.peers()]}
        atomic_json(self.state / "report.json", result)
        report("summary", success=success, accepted_sequence=final.sequence, inference_max_difference=max_difference)
        if not success:
            raise RuntimeError("Internet learning gate did not pass")

    def run(self):
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, lambda *_: self.stop.set())
        if self.args.role == "worker":
            self.work.register_handler("learning.prepare.v1", "memory", self.prepare)
            self.work.register_handler("reference.inference.v1", "inference", self.inference)
        try:
            self.node.start()
            report("started", role=self.args.role, identity=self.identity.node_id_hex, device=self.device)
            if self.args.role == "coordinator":
                self.coordinator()
            else:
                while not self.stop.is_set():
                    self.status(); self.stop.wait(15)
        finally:
            self.work.close(); self.node.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--role", choices=("worker", "coordinator"))
    parser.add_argument("--print-id", action="store_true")
    parser.add_argument("--coordinator-id", default="")
    parser.add_argument("--worker-ids", default="")
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    if args.print_id:
        print(identity_at(args.state_dir).node_id_hex); return
    if not args.role or not 1 <= args.cycles <= 12:
        parser.error("role and a bounded cycle count are required")
    Pilot(args).run()


if __name__ == "__main__":
    main()
