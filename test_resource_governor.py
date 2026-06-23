from pathlib import Path

import pytest

from src.contribution.resource_governor import (
    ContributionPolicy,
    ResourceGovernor,
    ResourceRequirements,
)
from src.swarm.node_manager import HardwareProfile


HARDWARE = HardwareProfile(
    cpu_cores=8,
    ram_gb=16,
    gpu_vram_gb=12,
    disk_gb=100,
    has_gpu=True,
)


def test_policy_roundtrip_is_private(tmp_path):
    path = tmp_path / "contribution.json"
    policy = ContributionPolicy.for_mode("balanced")
    policy.save(path)
    loaded = ContributionPolicy.load(path)
    assert loaded == policy
    assert path.stat().st_mode & 0o777 == 0o600
    assert str(tmp_path) not in path.read_text()


def test_missing_policy_defaults_to_relay(tmp_path):
    policy = ContributionPolicy.load(tmp_path / "missing.json")
    assert policy.mode == "relay"
    governor = ResourceGovernor(policy, hardware=HARDWARE, storage_path=tmp_path)
    accepted, reason = governor.can_accept(ResourceRequirements(kind="training"))
    assert not accepted
    assert reason == "compute_not_consented"


def test_balanced_policy_advertises_bounded_compute(tmp_path):
    governor = ResourceGovernor(
        ContributionPolicy.for_mode("balanced"),
        hardware=HARDWARE,
        storage_path=tmp_path,
    )
    profile = governor.advertised_profile()
    assert profile["mode"] == "moderate"
    assert profile["work"]["inference"]
    assert profile["work"]["training"]
    assert profile["ram_gb"] <= governor.policy.max_ram_gb
    assert "path" not in profile


def test_lease_enforces_concurrency_and_releases(tmp_path, monkeypatch):
    policy = ContributionPolicy.for_mode("balanced")
    policy.max_cpu_jobs = 1
    governor = ResourceGovernor(policy, hardware=HARDWARE, storage_path=tmp_path)
    monkeypatch.setattr(governor, "_system_busy", lambda: False)
    requirements = ResourceRequirements(kind="inference", ram_gb=0.5)

    lease, reason = governor.acquire(requirements)
    assert reason == "ok" and lease is not None
    second, reason = governor.acquire(requirements)
    assert second is None and reason == "cpu_job_limit"
    lease.release()
    third, reason = governor.acquire(requirements)
    assert reason == "ok" and third is not None
    third.release()
    assert governor.status()["active_cpu_jobs"] == 0


def test_gpu_job_requires_gpu_and_gpu_consent(tmp_path, monkeypatch):
    policy = ContributionPolicy.for_mode("balanced")
    policy.gpu_percent = 0
    governor = ResourceGovernor(policy, hardware=HARDWARE, storage_path=tmp_path)
    monkeypatch.setattr(governor, "_system_busy", lambda: False)
    accepted, reason = governor.can_accept(ResourceRequirements(
        kind="training", gpu_required=True
    ))
    assert not accepted and reason == "gpu_unavailable"


def test_ram_and_disk_policy_limits(tmp_path, monkeypatch):
    policy = ContributionPolicy.for_mode("balanced")
    policy.max_ram_gb = 1
    policy.max_disk_gb = 1
    governor = ResourceGovernor(policy, hardware=HARDWARE, storage_path=tmp_path)
    monkeypatch.setattr(governor, "_system_busy", lambda: False)
    assert governor.can_accept(ResourceRequirements(
        kind="inference", ram_gb=2
    )) == (False, "ram_limit")
    assert governor.can_accept(ResourceRequirements(
        kind="artifact", disk_gb=2
    )) == (False, "disk_policy_limit")


def test_invalid_policy_is_rejected():
    policy = ContributionPolicy.for_mode("balanced")
    policy.cpu_percent = 101
    with pytest.raises(ValueError):
        policy.validate()


def test_resource_requirements_reject_arbitrary_work_kind(tmp_path):
    governor = ResourceGovernor(
        ContributionPolicy.for_mode("maximum"),
        hardware=HARDWARE,
        storage_path=tmp_path,
    )
    with pytest.raises(ValueError):
        governor.can_accept(ResourceRequirements(kind="execute_python"))
