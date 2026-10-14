import hashlib
import json
import platform
import sys

import pytest

from src.network.engine_package import install_package, verify_package
from src.network.reference_compute import EngineSettings


def package(tmp_path):
    root = tmp_path / "package"
    root.mkdir()
    entry = "LixySwarmEngineCPU.exe" if sys.platform == "win32" else "LixySwarmEngineCPU"
    (root / entry).write_bytes(b"trusted-test-package")
    value = {"version": 1, "name": "LixySwarmEngineCPU", "platform": sys.platform,
             "machine": platform.machine(), "torch": "2.12.0+cpu",
             "operations": ["reference.inference.v1", "training.gradient.v1", "reference.candidate.v1"],
             "entrypoint": entry, "files": [{"path": entry, "bytes": 20,
             "sha256": hashlib.sha256(b"trusted-test-package").hexdigest()}]}
    path = root / "engine-package.json"
    path.write_text(json.dumps(value))
    return root, value, hashlib.sha256(path.read_bytes()).hexdigest()


def test_install_verifies_package_and_preserves_training_authority(tmp_path):
    root, value, digest = package(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    EngineSettings(training_requesters=["a" * 64]).save(state)
    entry = install_package(root, digest, state)
    assert EngineSettings.load(state).command == [entry]
    assert EngineSettings.load(state).training_requesters == ["a" * 64]
    assert EngineSettings.load(state).devices == ["cpu"]
    assert install_package(root, digest, state) == entry
    (root / value["entrypoint"]).write_bytes(b"modified")
    with pytest.raises(ValueError, match="verification"):
        install_package(root, digest, state)


def test_digest_cannot_be_supplied_by_package_or_unknown_files(tmp_path):
    root, _, digest = package(tmp_path)
    with pytest.raises(ValueError, match="not trusted"):
        verify_package(root, "0" * 64)
    (root / "extra-code.py").write_text("pass")
    with pytest.raises(ValueError, match="undeclared"):
        verify_package(root, digest)


@pytest.mark.parametrize("name", ["../outside", "/outside", "C:/outside", "a\\outside", "./outside"])
def test_even_a_trusted_manifest_cannot_escape_install_root(tmp_path, name):
    root, value, _ = package(tmp_path)
    value["files"][0]["path"] = name
    path = root / "engine-package.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="path"):
        verify_package(root, hashlib.sha256(path.read_bytes()).hexdigest())


def test_symlink_outside_package_is_rejected(tmp_path):
    root, value, digest = package(tmp_path)
    source = root / value["entrypoint"]
    outside = tmp_path / "outside"
    source.rename(outside)
    try:
        source.symlink_to(outside)
    except OSError:
        pytest.skip("Symlink creation unavailable on this platform")
    with pytest.raises(ValueError, match="verification"):
        verify_package(root, digest)
