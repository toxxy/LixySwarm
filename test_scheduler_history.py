import json
import stat

from src.network.scheduler_history import SchedulerHistory


def test_scheduler_history_persists_identity_age_without_addresses(tmp_path):
    path = tmp_path / "scheduler.json"
    node_id = "ab" * 32
    history = SchedulerHistory(
        path, exploration_interval=5, minimum_age_s=60
    )
    history.observe([node_id], now=1_000)
    assert not history.is_aged(node_id, now=1_059)
    assert history.is_aged(node_id, now=1_060)
    history.record_dispatch([node_id], selected_at=1_061)
    history.close()

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    raw = path.read_text()
    assert node_id in raw
    assert "host" not in raw and "address" not in raw and "127.0.0.1" not in raw
    stored = json.loads(raw)
    assert stored["dispatch_count"] == 1

    loaded = SchedulerHistory(
        path, exploration_interval=5, minimum_age_s=60
    )
    assert loaded.snapshot() == {
        "dispatch_count": 1,
        "known_identities": 1,
    }
    assert loaded.is_aged(node_id, now=1_060)


def test_scheduler_history_ignores_malformed_entries(tmp_path):
    path = tmp_path / "scheduler.json"
    path.write_text(json.dumps({
        "version": 1,
        "dispatch_count": 3,
        "peers": {
            "not-a-node": {
                "first_seen_at": 1,
                "last_selected_at": 0,
                "selections": 1,
            },
        },
    }))
    history = SchedulerHistory(path)
    assert history.snapshot() == {
        "dispatch_count": 3,
        "known_identities": 0,
    }
