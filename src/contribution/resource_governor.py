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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from typing import TYPE_CHECKING

from .policy import ContributionPolicy, COMPUTE_KINDS, VALID_KINDS, POLICY_VERSION, VALID_MODES

if TYPE_CHECKING:
    from src.swarm.node_manager import HardwareProfile


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
        if hardware is None:
            from src.swarm.node_manager import HardwareProfile
            hardware = HardwareProfile.from_local()
        self.hardware = hardware
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
            "max_gpu_jobs": int(self.policy.max_gpu_jobs),
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
            if self.policy.only_when_idle and psutil.cpu_percent(interval=0.1) >= 35:
                return True
            available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
            if available_ram_gb < 0.5:
                return True
        except ImportError:
            if self.policy.only_when_idle:
                return True
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
