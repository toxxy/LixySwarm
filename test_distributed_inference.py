import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from lixy_orchestrator import (
    LixyOrchestrator,
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

    assert result == {"text": "isolated"}
    assert observed["session_file"] is None
    assert observed["store_memory"] is False
    assert observed["record_history"] is False
    assert observed["update_runtime_state"] is False
    assert observed["use_memory"] is False


def test_distributed_generation_never_silently_falls_back():
    orchestrator = object.__new__(LixyOrchestrator)
    orchestrator.cfg = SimpleNamespace(max_tokens=20, temperature=0.7, top_k=50)
    orchestrator.net = SimpleNamespace(work_coordinator=None)
    with pytest.raises(RuntimeError, match="not available"):
        orchestrator.generate_distributed("hello")
