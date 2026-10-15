"""Local control integrity and encrypted personal persistence."""
import json
import os
from pathlib import Path

import pytest

from src.contribution.policy import ContributionPolicy
from src.network.client_state import ClientSettings, NodeStateLock
from desktop.personal_store import PersonalStore, PersonalStorageUnavailable


class KeyStore:
    def __init__(self):
        self.values = {}

    def get_password(self, service, account):
        return self.values.get((service, account))

    def set_password(self, service, account, value):
        self.values[service, account] = value


def test_learning_button_uses_the_configured_language_progress(tmp_path,monkeypatch):
    from desktop.backend import NodeController
    from src.network.language_learning import LanguageLearningSettings
    from src.release.language_versions import WORKERS
    controls=LanguageLearningSettings(workers=sorted(WORKERS),signing_key_file=str(tmp_path/'publisher.key'))
    controls.save(tmp_path)
    controller=NodeController(tmp_path)
    monkeypatch.setattr(controller,'snapshot',lambda:{'diagnostics':{'learning':{'round':9},'language_learning':{'round':2}}})
    controller.start_learning()
    selected=LanguageLearningSettings.load(tmp_path)
    assert controller.learning_channel()=='language_learning' and selected.enabled
    assert selected.cycles==3 and selected.generation==1
    controller.stop_learning()
    assert not LanguageLearningSettings.load(tmp_path).enabled


def test_personal_content_is_encrypted_and_survives_a_new_store_instance(tmp_path):
    backend = KeyStore()
    store = PersonalStore(tmp_path, backend=backend)
    value = {"memories": [{"text": "private-personal-canary-72", "importance": 0.5}]}
    store.save(value)
    data = (tmp_path / "personal.enc").read_text()
    assert "private-personal-canary-72" not in data
    assert "importance" not in data
    assert PersonalStore(tmp_path, backend=backend).load() == value
    assert len(backend.values) == 1


def test_missing_key_cannot_replace_existing_encrypted_memory(tmp_path):
    store = PersonalStore(tmp_path, backend=KeyStore())
    store.save({"memory": "keep-this"})
    before = (tmp_path / "personal.enc").read_bytes()
    with pytest.raises(PersonalStorageUnavailable, match="Falta"):
        PersonalStore(tmp_path, backend=KeyStore())
    assert (tmp_path / "personal.enc").read_bytes() == before


def test_unavailable_keystore_has_no_plaintext_key_file_fallback(tmp_path):
    class Unavailable(KeyStore):
        def get_password(self, *_):
            raise RuntimeError("locked")
    with pytest.raises(PersonalStorageUnavailable):
        PersonalStore(tmp_path, backend=Unavailable())
    assert list(tmp_path.iterdir()) == []


def test_single_node_lock_is_released_after_context_exit(tmp_path):
    with NodeStateLock(tmp_path):
        with pytest.raises(RuntimeError, match="already"):
            with NodeStateLock(tmp_path):
                pass
    with NodeStateLock(tmp_path):
        pass


def test_transient_file_lock_preserves_old_record_until_atomic_replacement(tmp_path, monkeypatch):
    from src.network import telemetry
    destination = tmp_path / "status.json"
    destination.write_text('{"state":"old"}')
    original = telemetry.os.replace
    attempts = []
    def locked_once(source, target):
        attempts.append(source)
        if len(attempts) <= 2:
            assert json.loads(destination.read_text()) == {"state": "old"}
            raise PermissionError("temporary reader lock")
        original(source, target)
    monkeypatch.setattr(telemetry.os, "replace", locked_once)
    telemetry.atomic_json(destination, {"state": "new"})
    assert json.loads(destination.read_text()) == {"state": "new"}
    assert list(tmp_path.iterdir()) == [destination]


def test_control_settings_preserve_consent_and_reject_invalid_budgets(tmp_path):
    settings = ClientSettings(paused=True, policy=ContributionPolicy.for_mode("balanced"))
    settings.policy.cpu_percent = 17
    settings.save(tmp_path)
    loaded = ClientSettings.load(tmp_path)
    assert loaded == settings
    assert loaded.policy.consented_at > 0
    for value in (float("nan"), float("inf"), -1):
        loaded.policy.max_ram_gb = value
        with pytest.raises(ValueError):
            loaded.save(tmp_path)
    assert ClientSettings.load(tmp_path) == settings


def test_controls_cannot_enable_compute_without_consent_or_change_schema(tmp_path):
    settings = ClientSettings()
    assert settings.policy.gpu_percent == 0
    assert settings.policy.max_cpu_jobs == 0
    assert settings.policy.max_gpu_jobs == 0
    settings.policy.mode = "balanced"
    with pytest.raises(ValueError, match="consent"):
        settings.save(tmp_path)
    (tmp_path / "controls.json").write_text(json.dumps({"version": 1, "connected": True, "paused": False,
                                                      "policy": {}, "unexpected": "private"}))
    with pytest.raises(ValueError, match="schema"):
        ClientSettings.load(tmp_path)


def test_user_installation_keeps_state_separate_and_creates_launcher(tmp_path):
    from desktop.install import install_user_app
    import sys
    if sys.platform != "linux":
        pytest.skip("Linux reference launcher")
    bundle = tmp_path / "downloaded app"
    bundle.mkdir()
    executable = bundle / "LixySwarm"
    executable.write_bytes(b"bundled-executable")
    executable.chmod(0o755)
    (bundle / "_internal").mkdir()
    (bundle / "_internal" / "lib-example.so").write_bytes(b"library")
    installed, launcher = install_user_app(bundle, data_root=tmp_path / "apps", launcher_root=tmp_path / "menu")
    assert installed.read_bytes() == b"bundled-executable"
    assert (installed.parent / "_internal" / "lib-example.so").exists()
    assert "Terminal=false" in launcher.read_text()
    assert str(installed) in launcher.read_text()
    assert not (installed.parent / "identity.key").exists()
    repeated, _ = install_user_app(bundle, data_root=tmp_path / "apps", launcher_root=tmp_path / "menu")
    assert repeated == installed
    (bundle / "_internal" / "lib-example.so").write_bytes(b"updated-library")
    updated, _ = install_user_app(bundle, data_root=tmp_path / "apps", launcher_root=tmp_path / "menu")
    assert updated != installed
    assert installed.exists()
