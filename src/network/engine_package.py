"""Verify and install a locally selected package against a trusted digest.

The trusted digest is supplied by local application configuration, never by an
LSP peer or by the downloaded package itself. Verification does not execute
package code. An installed engine remains subject to contribution permissions.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath

from .reference_compute import EngineSettings
from .engine_profiles import engine_profile

HEX = re.compile(r"^[0-9a-f]{64}$")
MAX_MANIFEST_BYTES = 2 * 1024**2
MAX_PACKAGE_BYTES = 2 * 1024**3
MAX_PACKAGE_FILES = 10000


def digest_file(path):
    return _checked_digest(path,lambda:None)


def _checked_digest(path,check):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        before=os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(1024**2), b""):
            check()
            digest.update(block)
        after=os.fstat(stream.fileno())
        current=Path(path).stat()
        signature=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        if signature(before)!=signature(after) or signature(after)!=signature(current):
            raise ValueError('Engine file changed while its hash was verified')
    return digest.hexdigest()


def safe_relative(value):
    if not isinstance(value, str) or not value or len(value) > 1024 or "\\" in value or ":" in value:
        raise ValueError("Invalid engine package path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(part in {".", ".."} for part in path.parts):
        raise ValueError("Engine package path escaped its root")
    return path


def verify_package(directory, trusted_manifest_sha256, *, target_platform=None, target_machine=None,profile='reference-cpu',check=None):
    selected=engine_profile(profile);maximum=selected['max_package_bytes']
    root = Path(directory).resolve()
    checked_digest=digest_file if check is None else lambda p:_checked_digest(p,check)
    if check is not None:check()
    if not isinstance(trusted_manifest_sha256, str) or not HEX.fullmatch(trusted_manifest_sha256):
        raise ValueError("Engine installation requires a locally trusted manifest digest")
    manifest_path = root / "engine-package.json"
    if manifest_path.is_symlink() or manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("Engine manifest exceeds its bound or is a symlink")
    if checked_digest(manifest_path) != trusted_manifest_sha256:
        raise ValueError("Engine package manifest is not trusted")
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or set(manifest) != {"version", "name", "platform", "machine", "torch", "operations", "entrypoint", "files"}:
        raise ValueError("Invalid engine package schema")
    if manifest["version"] != 1 or manifest["name"] != selected['name'] or manifest["platform"] != (target_platform or sys.platform) or manifest["machine"] != (target_machine or platform.machine()):
        raise ValueError("Engine package is incompatible with this machine")
    if manifest["operations"] != selected['operations']:
        raise ValueError("Unsupported engine operations")
    expected_entry = selected['name']+('.exe' if (target_platform or sys.platform)=='win32' else '')
    if manifest["entrypoint"] != expected_entry:
        raise ValueError("Invalid engine entry point")
    files = manifest["files"]
    if not isinstance(files, list) or not 1 <= len(files) <= MAX_PACKAGE_FILES:
        raise ValueError("Engine package file count exceeds its bound")
    declared = set()
    size = 0
    for item in files:
        if check is not None:check()
        if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
            raise ValueError("Invalid engine file schema")
        relative = safe_relative(item["path"])
        if item["path"] in declared or item["path"] == "engine-package.json":
            raise ValueError("Duplicate engine package path")
        declared.add(item["path"])
        if type(item["bytes"]) is not int or not 0 <= item["bytes"] <= maximum or not isinstance(item["sha256"], str) or not HEX.fullmatch(item["sha256"]):
            raise ValueError("Invalid engine file bounds")
        size += item["bytes"]
        if size > maximum:
            raise ValueError("Engine package exceeds its disk bound")
        path = root.joinpath(*relative.parts)
        if not path.resolve().is_relative_to(root) or not path.is_file() or path.stat().st_size != item["bytes"] or checked_digest(path) != item["sha256"]:
            raise ValueError("Engine package file failed verification")
    actual = set()
    for path in root.rglob("*"):
        if check is not None:check()
        if path.is_symlink() and not path.resolve().is_relative_to(root):
            raise ValueError("Engine package symlink escaped its root")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
        elif not path.is_dir():
            raise ValueError("Engine package contains a non-regular object")
    if actual != declared | {"engine-package.json"} or expected_entry not in declared:
        raise ValueError("Engine package contains undeclared or missing files")
    return manifest


def install_package(directory, trusted_manifest_sha256, state_directory,*,profile='reference-cpu',consume_download=False):
    source = Path(directory)
    if type(consume_download) is not bool:raise ValueError('Invalid downloaded-package ownership')
    if consume_download:
        # Only the downloader's disposable, pinned staging directory may move.
        if (source.is_symlink() or source.parent.resolve() != (Path(state_directory)/'engine-downloads').resolve()
                or not re.fullmatch('[0-9a-f]{64}\\.unpacked',source.name)):
            raise ValueError('Only owned downloaded staging may be consumed')
    verify_package(source, trusted_manifest_sha256,profile=profile)
    engines = Path(state_directory) / "engines"
    engines.mkdir(parents=True, exist_ok=True)
    destination = engines / trusted_manifest_sha256
    if not destination.exists():
        with tempfile.TemporaryDirectory(prefix=".install-", dir=engines) as temporary:
            staged = Path(temporary) / "package"
            if consume_download:
                # Both locations belong to the same state tree. Fail on a
                # cross-filesystem move rather than silently duplicate gigabytes.
                source.rename(staged)
            else:
                shutil.copytree(source, staged, symlinks=True)
            # Verify the copy before atomically exposing it. A changed source
            # during copying must not become an executable selected by the node.
            verify_package(staged, trusted_manifest_sha256,profile=profile)
            try:
                staged.rename(destination)
            except FileExistsError:
                verify_package(destination, trusted_manifest_sha256,profile=profile)
    manifest = verify_package(destination, trusted_manifest_sha256,profile=profile)
    if sys.platform != "win32":
        # Archive extraction can lose executable bits. Apply this only after
        # the locally trusted digest and every installed file were verified.
        (destination / manifest["entrypoint"]).chmod(0o755)
    command = [str((destination / manifest["entrypoint"]).resolve())]
    if profile!='reference-cpu':
        from .telemetry import atomic_json
        atomic_json(Path(state_directory)/f'{profile}-prepared.json',{'version':1,'profile':profile,
            'manifest_sha256':trusted_manifest_sha256,'command':command})
        return command[0]
    settings = EngineSettings.load(state_directory)
    if settings.enabled and "cuda" in settings.devices and settings.command and all(Path(arg).is_file() for arg in settings.command):
        settings.cpu_command = command
        if "cpu" not in settings.devices: settings.devices.append("cpu")
    else:
        settings.command = command
        settings.cpu_command = []
        settings.devices = ["cpu"]
    settings.enabled = True
    settings.save(state_directory)
    return command[0]
