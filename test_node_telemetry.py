"""Per-node status authenticity, persistence, privacy and transport tests."""
import asyncio
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from src.network.lsp import LSPIdentity
from src.network.lsp_v3 import LSPNodeV3, PacketType
from src.network.telemetry import (
    NodeStatusStore, build_status, canonical, sign_status, verify_status,
)


def node(tmp_path, name):
    return LSPNodeV3(LSPIdentity.generate(), host="127.0.0.1", port=0,
                     address_book_path=tmp_path / f"{name}.json", target_outbound=0, allow_private=True)


def envelope(peer, *, sequence=1):
    status = build_status(peer, sequence=sequence, role="participant", state="connected",
                          uptime_s=5, resources={"cpu_cores": 4, "ram_gb": 16.0,
                          "gpu_vram_gb": 8.0, "has_gpu": True})
    return sign_status(peer.identity, status)


def wait_for(predicate, timeout=5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_transport_import_does_not_load_calculation_engines():
    result = subprocess.run([sys.executable, "-c", """
import sys
import client_daemon
from src.network import LSPNodeV3, LSPIdentity
assert 'torch' not in sys.modules
assert 'numpy' not in sys.modules
assert 'src.network.training_worker' not in sys.modules
"""], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_client_restart_keeps_identity_sequence_and_clean_shutdown(tmp_path):
    state = tmp_path / "machine-state"
    env = {**os.environ, "LIXYSWARM_BOOTSTRAP_SEEDS": ""}
    identities, sequences = [], []
    for _ in range(2):
        process = subprocess.Popen(
            [sys.executable, "client_daemon.py", "--state-dir", str(state), "--host", "127.0.0.1", "--port", "0"],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            previous = sequences[-1] if sequences else 0
            def started():
                try:
                    status = json.loads((state / "status.json").read_text())["status"]
                    return status["sequence"] > previous and status["state"] == "connecting"
                except (OSError, ValueError):
                    return False
            assert wait_for(started, timeout=10)
        finally:
            process.terminate()
            _, error = process.communicate(timeout=15)
        assert process.returncode == 0, error
        status = verify_status(json.loads((state / "status.json").read_text()))
        assert status["state"] == "disconnected"
        identities.append(status["node_id"])
        sequences.append(status["sequence"])
    assert identities[0] == identities[1]
    assert sequences[1] > sequences[0]
    assert (state / "identity.key").stat().st_mode & 0o777 == 0o600


def test_multiple_nodes_and_sequence_survive_collector_restart(tmp_path):
    a, b = node(tmp_path, "a"), node(tmp_path, "b")
    store = NodeStatusStore(tmp_path / "telemetry")
    for peer in (a, b):
        store.save(envelope(peer), peer.identity.node_id_hex)
    assert len(store.nodes()) == 2
    assert all(n["active"] and n["signature_verified"] for n in store.nodes())
    assert not any(n["capacity_verified"] or n["work_counters_verified"] for n in store.nodes())
    restarted = NodeStatusStore(store.directory)
    restarted.reset_connections()
    assert not any(n["active"] for n in restarted.nodes())
    with pytest.raises(ValueError, match="Replayed"):
        restarted.save(envelope(a), a.identity.node_id_hex)
    restarted.save(envelope(a, sequence=2), a.identity.node_id_hex)
    assert sum(n["active"] for n in restarted.nodes()) == 1


def test_offline_and_expired_nodes_are_not_counted_active(tmp_path):
    a = node(tmp_path, "a")
    store = NodeStatusStore(tmp_path / "telemetry")
    value = envelope(a)
    received = value["status"]["timestamp_ms"] / 1000
    store.save(value, a.identity.node_id_hex, now=received)
    assert store.nodes(now=received + 59)[0]["active"]
    assert not store.nodes(now=received + 61)[0]["active"]
    store.disconnected(a.identity.node_id_hex)
    assert store.nodes(now=received)[0]["state"] == "disconnected"


def test_signature_identity_timestamp_and_private_fields_are_checked(tmp_path):
    a, b = node(tmp_path, "a"), node(tmp_path, "b")
    value = envelope(a)
    tampered = copy.deepcopy(value)
    tampered["status"]["resources"]["gpu_vram_gb"] = 999
    with pytest.raises(ValueError, match="signature"):
        verify_status(tampered)
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_status(value, sender_id=b.identity.node_id_hex)
    with pytest.raises(ValueError, match="Expired"):
        verify_status(value, now_ms=value["status"]["timestamp_ms"] + 61000)
    for name in ("host", "hostname", "personal_memory", "conversation", "dataset"):
        private = copy.deepcopy(value["status"])
        private[name] = "private-canary"
        with pytest.raises(ValueError, match="schema"):
            sign_status(a.identity, private)
    assert b"private-canary" not in canonical(value)


def test_capacity_and_disk_tampering_do_not_invent_nodes(tmp_path):
    a, b = node(tmp_path, "a"), node(tmp_path, "b")
    store = NodeStatusStore(tmp_path / "telemetry", max_nodes=1)
    store.save(envelope(a), a.identity.node_id_hex)
    with pytest.raises(ValueError, match="capacity"):
        store.save(envelope(b), b.identity.node_id_hex)
    path = store.directory / f"{a.identity.node_id_hex}.json"
    record = json.loads(path.read_text())
    record["envelope"]["status"]["jobs"]["completed"] = 100
    path.write_text(json.dumps(record))
    assert store.nodes() == []


def test_repeated_publication_is_bounded_and_a_remote_seed_is_not_self(tmp_path):
    a = node(tmp_path, "a")
    store = NodeStatusStore(tmp_path / "telemetry")
    value = envelope(a)
    value["status"]["role"] = "seed"
    value = sign_status(a.identity, value["status"])
    store.save(value, a.identity.node_id_hex)
    assert not store.nodes()[0]["self"]
    with pytest.raises(ValueError, match="frequent"):
        store.save(envelope(a, sequence=2), a.identity.node_id_hex)


def test_api_combines_nodes_without_overwriting_and_separates_training_freshness(tmp_path, monkeypatch):
    from api import swarm_state
    a, b = node(tmp_path, "a"), node(tmp_path, "b")
    store = NodeStatusStore(tmp_path / "telemetry")
    for peer in (a, b):
        store.save(envelope(peer), peer.identity.node_id_hex)
    monkeypatch.setenv("LIXYSWARM_STATUS_DIRECTORY", str(store.directory))
    monkeypatch.setattr(swarm_state, "STATUS_FILE", tmp_path / "absent.json")
    status = swarm_state.get_full_swarm_status()
    assert status["network"]["connected_count"] == 2
    assert status["network"]["participant_count"] == 2
    assert status["network_data_fresh"] and status["data_fresh"]
    assert not status["training_data_fresh"]
    store.disconnected(a.identity.node_id_hex)
    assert swarm_state.get_network_state()["connected_count"] == 1


def test_expired_training_is_not_reported_as_current_activity(tmp_path, monkeypatch):
    from api import swarm_state
    published = tmp_path / "old-training.json"
    published.write_text(json.dumps({
        "agents": [{"id": "old-agent"}], "training_active": True,
        "swarm_diversity": 0.75, "metrics": {"step": 123, "val_loss": 1.5},
        "matriarca": {"memory_count": 90}, "dolphin": {"active_pings": 5},
    }))
    os.utime(published, (time.time() - 120, time.time() - 120))
    monkeypatch.setattr(swarm_state, "STATUS_FILE", published)
    monkeypatch.setenv("LIXYSWARM_STATUS_DIRECTORY", str(tmp_path / "telemetry"))
    status = swarm_state.get_full_swarm_status()
    assert status["agents"] == []
    assert status["swarm_diversity"] is None
    assert status["metrics"]["step"] is None
    assert status["metrics"]["val_loss"] is None
    assert not status["metrics"]["training_active"]
    assert status["matriarca"]["memory_count"] is None
    assert status["dolphin"]["active_pings"] is None


def test_lsp_delivers_signed_status_and_rejects_another_identity(tmp_path):
    collector, a, b = [node(tmp_path, name) for name in ("collector", "a", "b")]
    store = NodeStatusStore(tmp_path / "telemetry")
    collector.on_node_status_received(lambda value, sender: store.save(value, sender))
    try:
        for peer in (collector, a, b):
            peer.start()
        for peer in (a, b):
            assert peer.connect_peer("127.0.0.1", collector.port)
            assert peer.send_node_status(collector.identity.node_id_hex, envelope(peer))
        assert wait_for(lambda: len(store.nodes()) == 2)
        assert all(p["encrypted"] for p in collector.peers())
        # B has A's signed status but may not publish as A on B's session.
        future = asyncio.run_coroutine_threadsafe(b._send_to_peer(
            PacketType.NODE_STATUS, collector.identity.node_id_hex, canonical(envelope(a, sequence=2))
        ), b._loop)
        assert future.result(timeout=5)
        assert wait_for(lambda: b.identity.node_id_hex not in collector.connected_node_ids)
        assert store.sequence(a.identity.node_id_hex) == 1
    finally:
        for peer in (b, a, collector):
            peer.stop()
