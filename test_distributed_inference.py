import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from lixy_orchestrator import (
    LixyOrchestrator,
    _inference_majority,
    _validate_remote_inference_payload,
)


def test_remote_inference_schema_is_bounded():
    value = _validate_remote_inference_payload({
        "prompt": "hello",
        "max_tokens": 32,
        "temperature": 0.5,
        "top_k": 20,
    })
    assert value["max_tokens"] == 32
    assert value["deterministic"] is False

    with pytest.raises(ValueError, match="unsupported"):
        _validate_remote_inference_payload({"prompt": "x", "private_path": "/tmp"})
    with pytest.raises(ValueError, match="16 KiB"):
        _validate_remote_inference_payload({"prompt": "x" * (16 * 1024 + 1)})
    with pytest.raises(ValueError, match="max_tokens"):
        _validate_remote_inference_payload({"prompt": "x", "max_tokens": 513})


def test_remote_handler_disables_all_personal_state():
    orchestrator = object.__new__(LixyOrchestrator)
    orchestrator.cfg = SimpleNamespace(device="cpu")
    orchestrator.swarm = object()
    orchestrator.enc = object()
    orchestrator.model_artifact_id = "ab" * 32
    orchestrator._inference_lock = threading.RLock()
    observed = {}

    class FakeSession:
        def __init__(self, *_args, **kwargs):
            observed["session_file"] = kwargs["session_file"]

        def turn(self, prompt, **kwargs):
            observed["prompt"] = prompt
            observed.update(kwargs)
            return "isolated"

    with patch("lixy_orchestrator.RuntimeSession", FakeSession):
        result = orchestrator._handle_remote_inference(
            {"prompt": "network request", "max_tokens": 8}, None
        )

    assert result["text"] == "isolated"
    assert result["deterministic"] is False
    assert observed["session_file"] is None
    assert observed["store_memory"] is False
    assert observed["record_history"] is False
    assert observed["update_runtime_state"] is False
    assert observed["use_memory"] is False
    assert observed["greedy"] is False


def test_distributed_generation_never_silently_falls_back():
    orchestrator = object.__new__(LixyOrchestrator)
    orchestrator.cfg = SimpleNamespace(max_tokens=20, temperature=0.7, top_k=50)
    orchestrator.net = SimpleNamespace(work_coordinator=None)
    with pytest.raises(RuntimeError, match="not available"):
        orchestrator.generate_distributed("hello")


def test_inference_quorum_requires_exact_majority_and_distinct_peers():
    result = _inference_majority([
        ("01" * 32, "same", {"a": 1}),
        ("02" * 32, "other", {"b": 2}),
        ("03" * 32, "same", {"c": 3}),
    ])
    assert result["text"] == "same"
    assert result["votes"] == 2
    assert len(result["supporters"]) == 2
    with pytest.raises(RuntimeError, match="majority"):
        _inference_majority([
            ("01" * 32, "a", {}),
            ("02" * 32, "b", {}),
            ("03" * 32, "c", {}),
        ])
    with pytest.raises(RuntimeError, match="distinct"):
        _inference_majority([
            ("01" * 32, "same", {}),
            ("01" * 32, "same", {}),
            ("03" * 32, "same", {}),
        ])


def test_verified_inference_replaces_failed_peer_without_weakening_quorum():
    peers = [f"{index:02x}" * 32 for index in range(1, 5)]
    model_id = "ab" * 32

    class Coordinator:
        def __init__(self):
            self.selections = 0

        def select_peers(self, _requirements, **kwargs):
            self.selections += 1
            if self.selections == 1:
                return peers[:3]
            assert kwargs["excluded_peer_ids"] == set(peers[:3])
            return [peers[3]]

    class Net:
        work_coordinator = Coordinator()

        def submit_work(self, _operation, _payload, _requirements, **kwargs):
            peer_id = kwargs["peer_id"]
            if peer_id == peers[0]:
                raise ConnectionError("peer disconnected")
            return SimpleNamespace(
                status="ok",
                error="",
                output={
                    "text": "stable",
                    "model_artifact_id": model_id,
                    "deterministic": True,
                },
                receipt={"worker_node_id": peer_id},
            )

    orchestrator = object.__new__(LixyOrchestrator)
    orchestrator.net = Net()
    orchestrator.model_artifact_id = model_id
    result = orchestrator.generate_distributed_verified(
        "hello",
        replicas=3,
        max_replacements=1,
        max_tokens=8,
        timeout_s=2,
    )
    assert result["text"] == "stable"
    assert result["votes"] == 3
    assert result["replicas"] == 3
    assert result["worker_attempts"] == 4
    assert result["replacements_used"] == 1
