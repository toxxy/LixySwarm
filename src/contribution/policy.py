"""Lightweight local consent policy; independent of model engines."""
from __future__ import annotations
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

POLICY_VERSION = 1
VALID_MODES = {"disabled", "relay", "balanced", "maximum"}
COMPUTE_KINDS = {"inference", "training"}
VALID_KINDS = COMPUTE_KINDS | {"memory", "connectivity", "artifact"}


@dataclass
class ContributionPolicy:
    version: int = POLICY_VERSION
    mode: str = "relay"
    consented_at: float = 0.0
    cpu_percent: int = 25
    max_cpu_jobs: int = 1
    gpu_percent: int = 25
    max_gpu_jobs: int = 1
    max_ram_gb: float = 2.0
    max_disk_gb: float = 10.0
    max_bandwidth_mbps: float = 10.0
    pause_on_battery: bool = True
    pause_above_cpu_percent: int = 80
    pause_above_gpu_temp_c: int = 82
    only_when_idle: bool = False

    @classmethod
    def for_mode(cls, mode: str) -> "ContributionPolicy":
        mode = str(mode).lower()
        if mode not in VALID_MODES:
            raise ValueError(f"Unknown contribution mode: {mode}")
        now = time.time()
        if mode == "disabled":
            return cls(mode=mode, consented_at=now, cpu_percent=0, gpu_percent=0,
                       max_cpu_jobs=0, max_gpu_jobs=0, max_ram_gb=0, max_disk_gb=0,
                       max_bandwidth_mbps=0)
        if mode == "relay":
            return cls(mode=mode, consented_at=now, cpu_percent=5, gpu_percent=0,
                       max_cpu_jobs=0, max_gpu_jobs=0, max_ram_gb=0.5,
                       max_disk_gb=2, max_bandwidth_mbps=5)
        if mode == "maximum":
            return cls(mode=mode, consented_at=now, cpu_percent=90, gpu_percent=90,
                       max_cpu_jobs=max(1, (os.cpu_count() or 2) - 1), max_gpu_jobs=2,
                       max_ram_gb=32, max_disk_gb=100, max_bandwidth_mbps=100,
                       pause_above_cpu_percent=98)
        return cls(mode="balanced", consented_at=now)

    def validate(self):
        if type(self.version) is not int or self.version != POLICY_VERSION:
            raise ValueError("Unsupported contribution policy version")
        if self.mode not in VALID_MODES:
            raise ValueError("Invalid contribution mode")
        for name in ("cpu_percent", "gpu_percent", "pause_above_cpu_percent"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= 100:
                raise ValueError(f"{name} must be in [0, 100]")
        for name in ("max_cpu_jobs", "max_gpu_jobs"):
            if type(getattr(self, name)) is not int or not 0 <= getattr(self, name) <= 1024:
                raise ValueError(f"{name} is out of range")
        for name in ("max_ram_gb", "max_disk_gb", "max_bandwidth_mbps"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1_000_000:
                raise ValueError(f"{name} is out of range")
        if type(self.consented_at) not in (int, float) or not math.isfinite(self.consented_at) or self.consented_at < 0:
            raise ValueError("Invalid contribution consent timestamp")
        for name in ("pause_on_battery", "only_when_idle"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if type(self.pause_above_gpu_temp_c) is not int or not 30 <= self.pause_above_gpu_temp_c <= 120:
            raise ValueError("Invalid GPU temperature limit")
        if self.mode in {"balanced", "maximum"} and self.consented_at <= 0:
            raise ValueError("Compute contribution requires explicit persisted consent")

    def save(self, path: str | Path):
        self.validate()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))
        os.chmod(temporary, 0o600)
        temporary.replace(destination)
        os.chmod(destination, 0o600)

    @classmethod
    def load(cls, path: str | Path) -> "ContributionPolicy":
        source = Path(path)
        if not source.exists():
            # Safe first start: connectivity only until onboarding records a
            # balanced/maximum choice.
            return cls.for_mode("relay")
        payload = json.loads(source.read_text())
        policy = cls(**{
            key: value for key, value in payload.items()
            if key in cls.__dataclass_fields__
        })
        policy.validate()
        return policy
