"""Consent-based resource limits for distributed LixySwarm work.

Network peers may advertise work, but only a ResourceGovernor can authorize
local execution. No peer-provided code is executed by this module.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from src.swarm.node_manager import HardwareProfile


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
        if self.version != POLICY_VERSION:
            raise ValueError("Unsupported contribution policy version")
        if self.mode not in VALID_MODES:
            raise ValueError("Invalid contribution mode")
        for name in ("cpu_percent", "gpu_percent", "pause_above_cpu_percent"):
            value = int(getattr(self, name))
            if not 0 <= value <= 100:
                raise ValueError(f"{name} must be in [0, 100]")
        for name in ("max_cpu_jobs", "max_gpu_jobs"):
            if not 0 <= int(getattr(self, name)) <= 1024:
                raise ValueError(f"{name} is out of range")
        for name in ("max_ram_gb", "max_disk_gb", "max_bandwidth_mbps"):
            if not 0 <= float(getattr(self, name)) <= 1_000_000:
                raise ValueError(f"{name} is out of range")
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


@dataclass(frozen=True)
class ResourceRequirements:
    kind: str
    cpu_slots: int = 1
    gpu_required: bool = False
    ram_gb: float = 0.0
    disk_gb: float = 0.0

    def validate(self):
        if self.kind not in VALID_KINDS:
            raise ValueError("Unknown work kind")
        if not 0 <= self.cpu_slots <= 1024:
            raise ValueError("cpu_slots is out of range")
        if not 0 <= self.ram_gb <= 1_000_000 or not 0 <= self.disk_gb <= 1_000_000:
            raise ValueError("Resource requirement is out of range")


class ResourceLease(AbstractContextManager):
    def __init__(self, governor: "ResourceGovernor", requirements: ResourceRequirements):
        self.governor = governor
        self.requirements = requirements
        self._released = False

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.release()
        return False

    def release(self):
        if not self._released:
            self._released = True
            self.governor._release(self.requirements)


class ResourceGovernor:
    """Authorize bounded, allowlisted work against a local consent policy."""

    def __init__(
        self,
        policy: ContributionPolicy,
        *,
        hardware: Optional[HardwareProfile] = None,
        storage_path: str | Path = ".",
    ):
        policy.validate()
        self.policy = policy
        self.hardware = hardware or HardwareProfile.from_local()
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._active_cpu_jobs = 0
        self._active_gpu_jobs = 0
        self._leased_ram_gb = 0.0
        self._leased_disk_gb = 0.0

    def advertised_profile(self, available_work: Optional[set[str]] = None) -> dict:
        """Return scheduler metadata without usernames, paths, or addresses."""
        mode = self.policy.mode
        allowed = {
            "connectivity": mode != "disabled",
            "memory": mode != "disabled",
            "artifact": mode != "disabled",
            "inference": mode in {"balanced", "maximum"},
            "training": mode in {"balanced", "maximum"},
        }
        if available_work is not None:
            available = set(available_work) | {"connectivity"}
            allowed = {
                kind: consented and kind in available
                for kind, consented in allowed.items()
            }
        return {
            "mode": "moderate" if mode == "balanced" else "maximum" if mode == "maximum" else "relay",
            "consent_version": self.policy.version,
            "cpu_cores": max(1, min(int(self.hardware.cpu_cores), 1024)),
            "cpu_percent": int(self.policy.cpu_percent),
            "ram_gb": round(min(float(self.hardware.ram_gb), self.policy.max_ram_gb), 2),
            "gpu_vram_gb": round(float(self.hardware.gpu_vram_gb), 2),
            "gpu_percent": int(self.policy.gpu_percent),
            "has_gpu": bool(self.hardware.has_gpu and self.policy.gpu_percent > 0),
            "disk_gb": round(min(float(self.hardware.disk_gb), self.policy.max_disk_gb), 2),
            "bandwidth_mbps": round(float(self.policy.max_bandwidth_mbps), 2),
            "work": allowed,
        }

    def can_accept(self, requirements: ResourceRequirements) -> tuple[bool, str]:
        requirements.validate()
        with self._lock:
            if self.policy.mode == "disabled":
                return False, "contribution_disabled"
            if requirements.kind in COMPUTE_KINDS and self.policy.mode == "relay":
                return False, "compute_not_consented"
            if requirements.gpu_required:
                if not self.hardware.has_gpu or self.policy.gpu_percent <= 0:
                    return False, "gpu_unavailable"
                if self._active_gpu_jobs >= self.policy.max_gpu_jobs:
                    return False, "gpu_job_limit"
            if requirements.kind in COMPUTE_KINDS:
                if self._active_cpu_jobs >= self.policy.max_cpu_jobs:
                    return False, "cpu_job_limit"
                if self._system_busy():
                    return False, "user_activity_or_system_load"
            if self._leased_ram_gb + requirements.ram_gb > self.policy.max_ram_gb:
                return False, "ram_limit"
            if self._leased_disk_gb + requirements.disk_gb > self.policy.max_disk_gb:
                return False, "disk_policy_limit"
            try:
                free_disk_gb = shutil.disk_usage(self.storage_path).free / (1024 ** 3)
                if requirements.disk_gb > free_disk_gb:
                    return False, "disk_unavailable"
            except OSError:
                return False, "storage_unavailable"
            return True, "ok"

    def acquire(self, requirements: ResourceRequirements) -> tuple[Optional[ResourceLease], str]:
        with self._lock:
            accepted, reason = self.can_accept(requirements)
            if not accepted:
                return None, reason
            if requirements.kind in COMPUTE_KINDS:
                self._active_cpu_jobs += 1
            if requirements.gpu_required:
                self._active_gpu_jobs += 1
            self._leased_ram_gb += requirements.ram_gb
            self._leased_disk_gb += requirements.disk_gb
            return ResourceLease(self, requirements), "ok"

    def _release(self, requirements: ResourceRequirements):
        with self._lock:
            if requirements.kind in COMPUTE_KINDS:
                self._active_cpu_jobs = max(0, self._active_cpu_jobs - 1)
            if requirements.gpu_required:
                self._active_gpu_jobs = max(0, self._active_gpu_jobs - 1)
            self._leased_ram_gb = max(0.0, self._leased_ram_gb - requirements.ram_gb)
            self._leased_disk_gb = max(0.0, self._leased_disk_gb - requirements.disk_gb)

    def _system_busy(self) -> bool:
        if self.policy.pause_on_battery:
            try:
                import psutil
                battery = psutil.sensors_battery()
                if battery is not None and not battery.power_plugged:
                    return True
            except (ImportError, AttributeError):
                pass
        try:
            import psutil
            if psutil.cpu_percent(interval=None) >= self.policy.pause_above_cpu_percent:
                return True
            available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
            if available_ram_gb < 0.5:
                return True
        except ImportError:
            pass
        return False

    def status(self) -> dict:
        with self._lock:
            return {
                "mode": self.policy.mode,
                "active_cpu_jobs": self._active_cpu_jobs,
                "active_gpu_jobs": self._active_gpu_jobs,
                "leased_ram_gb": round(self._leased_ram_gb, 3),
                "leased_disk_gb": round(self._leased_disk_gb, 3),
            }
