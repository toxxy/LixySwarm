"""Machine-local control settings and portable single-instance lock."""
from __future__ import annotations
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.contribution.policy import ContributionPolicy
from .telemetry import atomic_json


def state_directory() -> Path:
    override = os.environ.get("LIXYSWARM_STATE_DIR")
    if override:
        return Path(override)
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))) / "LixySwarm" / "State"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "LixySwarm"
    return Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local" / "state"))) / "lixyswarm"


class NodeStateLock:
    def __init__(self, directory: Path):
        self.directory = directory
        self.stream = None

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.stream = (self.directory / "node.lock").open("a+b")
        try:
            if sys.platform == "win32":
                import msvcrt
                self.stream.seek(0)
                if not self.stream.read(1):
                    self.stream.write(b"0"); self.stream.flush()
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close()
            raise RuntimeError("A node already uses this state directory") from exc
        return self

    def __exit__(self, *_):
        if sys.platform == "win32":
            import msvcrt
            self.stream.seek(0)
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        self.stream.close()


@dataclass
class ClientSettings:
    version: int = 1
    connected: bool = True
    paused: bool = False
    policy: ContributionPolicy = field(default_factory=ContributionPolicy)

    def validate(self):
        if type(self.version) is not int or self.version != 1:
            raise ValueError("Unsupported client settings version")
        if type(self.connected) is not bool or type(self.paused) is not bool:
            raise ValueError("Client controls must be boolean")
        self.policy.validate()

    def save(self, directory: Path):
        self.validate()
        atomic_json(directory / "controls.json", asdict(self))

    @classmethod
    def load(cls, directory: Path):
        path = directory / "controls.json"
        if not path.exists():
            return cls()
        if path.stat().st_size > 8192:
            raise ValueError("Client settings are too large")
        value = json.loads(path.read_text())
        if not isinstance(value, dict) or set(value) != {"version", "connected", "paused", "policy"}:
            raise ValueError("Invalid client settings schema")
        settings = cls(**{**value, "policy": ContributionPolicy(**value["policy"])})
        settings.validate()
        return settings
