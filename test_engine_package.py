import hashlib
import json
import platform
from pathlib import Path
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


def test_own_cpu_profile_requires_its_explicit_trust_without_enabling_reference_or_own_work(tmp_path):
    root,value,_=package(tmp_path)
    value['operations']+=['swarm.gradient.v1','swarm.candidate.v1']
    path=root/'engine-package.json';path.write_text(json.dumps(value))
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='operations'):verify_package(root,digest)
    assert verify_package(root,digest,profile='own-cpu')['operations']==value['operations']
    state=tmp_path/'state'
    installed=install_package(root,digest,state,profile='own-cpu')
    assert Path(installed).is_file() and not EngineSettings.load(state).enabled
    assert (state/'own-cpu-prepared.json').is_file() and not (state/'own-swarm.json').exists()


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


def test_only_owned_downloaded_staging_can_move_without_a_second_package_copy(tmp_path,monkeypatch):
    root,value,digest=package(tmp_path);state=tmp_path/'state'
    with pytest.raises(ValueError,match='owned downloaded'):
        install_package(root,digest,state,consume_download=True)
    assert root.is_dir()
    download=state/'engine-downloads'/('a'*64+'.unpacked');download.parent.mkdir(parents=True)
    root.rename(download)
    def no_copy(*args,**kwargs):pytest.fail('Downloaded engine must install without another full copy')
    monkeypatch.setattr('src.network.engine_package.shutil.copytree',no_copy)
    command=install_package(download,digest,state,consume_download=True)
    assert not download.exists() and Path(command).read_bytes()==b'trusted-test-package'


def test_owned_download_is_reverified_after_moving_before_activation(tmp_path,monkeypatch):
    import src.network.engine_package as module
    root,value,digest=package(tmp_path);state=tmp_path/'state'
    download=state/'engine-downloads'/('b'*64+'.unpacked');download.parent.mkdir(parents=True);root.rename(download)
    original=module.verify_package;checks=[]
    def verify(path,*args,**kwargs):
        result=original(path,*args,**kwargs);checks.append(str(path))
        if len(checks)==1:(Path(path)/value['entrypoint']).write_bytes(b'changed during move')
        return result
    monkeypatch.setattr(module,'verify_package',verify)
    with pytest.raises(ValueError,match='verification'):install_package(download,digest,state,consume_download=True)
    assert not EngineSettings.load(state).enabled and not (state/'engines'/digest).exists()
