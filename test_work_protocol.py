import pytest
from dataclasses import replace

from src.contribution import ContributionPolicy, ResourceGovernor, ResourceRequirements
from src.network.lsp import LSPIdentity
from src.network.lsp_v3 import LSPNodeV3
from src.network.work_protocol import (
    ResultReceipt,
    WorkCoordinator,
    WorkProtocolError,
    WorkResult,
    WorkUnit,
)
from src.swarm.node_manager import HardwareProfile


HARDWARE = HardwareProfile(
    cpu_cores=8, ram_gb=16, gpu_vram_gb=12, disk_gb=100, has_gpu=True
)


def _governor(tmp_path, mode):
    governor = ResourceGovernor(
        ContributionPolicy.for_mode(mode), hardware=HARDWARE, storage_path=tmp_path
    )
    governor._system_busy = lambda: False
    return governor


def _node(tmp_path, name, governor):
    return LSPNodeV3(
        LSPIdentity.generate(),
        host="127.0.0.1",
        port=0,
        address_book_path=tmp_path / f"{name}.json",
        target_outbound=0,
        allow_private=True,
        resource_profile=governor.advertised_profile(),
    )


def test_work_unit_is_content_addressed_and_rejects_code():
    requirements = ResourceRequirements(kind="inference", ram_gb=0.5)
    work = WorkUnit.create(
        origin_node_id="ab" * 32,
        operation="inference.echo.v1",
        kind="inference",
        requirements=requirements,
        payload={"prompt": "hello"},
    )
    assert WorkUnit.from_dict(work.to_dict()) == work
    tampered = work.to_dict()
    tampered["payload"] = {"prompt": "changed"}
    with pytest.raises(WorkProtocolError, match="job ID"):
        WorkUnit.from_dict(tampered)
    with pytest.raises(WorkProtocolError, match="Executable"):
        WorkUnit.create(
            origin_node_id="ab" * 32,
            operation="inference.echo.v1",
            kind="inference",
            requirements=requirements,
            payload={"command": "rm -rf /"},
        )
    with pytest.raises(WorkProtocolError, match="Executable"):
        WorkUnit.create(
            origin_node_id="ab" * 32,
            operation="inference.echo.v1",
            kind="inference",
            requirements=requirements,
            payload={"nested": {"script": "ignored but forbidden"}},
        )


def test_result_receipt_binds_worker_requester_and_output():
    worker = LSPIdentity.generate()
    requester_id = "cd" * 32
    result = WorkResult(
        job_id="ab" * 32,
        status="ok",
        output={"text": "verified"},
    )
    receipt = ResultReceipt.create(result, worker, requester_id)
    signed = replace(result, receipt=receipt.to_dict())
    assert ResultReceipt.from_dict(signed.receipt).verify(
        signed,
        expected_worker_id=worker.node_id_hex,
        expected_requester_id=requester_id,
    )
    assert not receipt.verify(
        replace(result, output={"text": "tampered"}),
        expected_worker_id=worker.node_id_hex,
        expected_requester_id=requester_id,
    )
    assert not receipt.verify(
        result,
        expected_worker_id=worker.node_id_hex,
        expected_requester_id="ef" * 32,
    )
def test_distributed_inference_executes_allowlisted_handler(tmp_path):
    requester_governor = _governor(tmp_path / "requester", "relay")
    worker_governor = _governor(tmp_path / "worker", "balanced")
    requester = _node(tmp_path, "requester", requester_governor)
    worker = _node(tmp_path, "worker", worker_governor)
    requester.start()
    worker.start()
    requester_coordinator = WorkCoordinator(requester, requester_governor)
    worker_coordinator = WorkCoordinator(worker, worker_governor)
    worker_coordinator.register_handler(
        "inference.echo.v1",
        "inference",
        lambda payload, _work: {"text": payload["prompt"].upper()},
    )
    try:
        assert requester.connect_peer("127.0.0.1", worker.port)
        result = requester_coordinator.submit(
            "inference.echo.v1",
            {"prompt": "distributed"},
            ResourceRequirements(kind="inference", ram_gb=0.25),
            timeout_s=5,
        )
        assert result.status == "ok"
        assert result.output == {"text": "DISTRIBUTED"}
        assert result.receipt
        assert worker_governor.status()["active_cpu_jobs"] == 0
    finally:
        requester_coordinator.close()
        worker_coordinator.close()
        requester.stop()
        worker.stop()


def test_compute_offer_is_rejected_without_consent(tmp_path):
    requester_governor = _governor(tmp_path / "requester", "relay")
    worker_governor = _governor(tmp_path / "worker", "relay")
    requester = _node(tmp_path, "requester", requester_governor)
    worker = _node(tmp_path, "worker", worker_governor)
    requester.start()
    worker.start()
    requester_coordinator = WorkCoordinator(requester, requester_governor)
    worker_coordinator = WorkCoordinator(worker, worker_governor)
    worker_coordinator.register_handler(
        "training.batch.v1", "training", lambda _payload, _work: {"loss": 1.0}
    )
    try:
        assert requester.connect_peer("127.0.0.1", worker.port)
        result = requester_coordinator.submit(
            "training.batch.v1",
            {"shard": "00" * 32},
            ResourceRequirements(kind="training", ram_gb=0.1),
            peer_id=worker.identity.node_id_hex,
            timeout_s=5,
        )
        assert result.status == "rejected"
        assert result.error == "compute_not_consented"
    finally:
        requester_coordinator.close()
        worker_coordinator.close()
        requester.stop()
        worker.stop()


def test_unregistered_operation_is_never_executed(tmp_path):
    requester_governor = _governor(tmp_path / "requester", "relay")
    worker_governor = _governor(tmp_path / "worker", "maximum")
    requester = _node(tmp_path, "requester", requester_governor)
    worker = _node(tmp_path, "worker", worker_governor)
    requester.start()
    worker.start()
    requester_coordinator = WorkCoordinator(requester, requester_governor)
    worker_coordinator = WorkCoordinator(worker, worker_governor)
    try:
        assert requester.connect_peer("127.0.0.1", worker.port)
        result = requester_coordinator.submit(
            "inference.unknown.v1",
            {"prompt": "hello"},
            ResourceRequirements(kind="inference"),
            peer_id=worker.identity.node_id_hex,
            timeout_s=5,
        )
        assert result.status == "rejected"
        assert result.error == "operation_not_allowed"
    finally:
        requester_coordinator.close()
        worker_coordinator.close()
        requester.stop()
        worker.stop()


def test_scheduler_filters_peers_that_cannot_meet_requirements(tmp_path):
    requester_governor = _governor(tmp_path / "requester", "relay")
    worker_governor = _governor(tmp_path / "worker", "balanced")
    requester = _node(tmp_path, "requester", requester_governor)
    worker = _node(tmp_path, "worker", worker_governor)
    requester.start()
    worker.start()
    coordinator = WorkCoordinator(requester, requester_governor)
    try:
        assert requester.connect_peer("127.0.0.1", worker.port)
        with pytest.raises(RuntimeError, match="No peer"):
            coordinator.submit(
                "inference.echo.v1",
                {"prompt": "too large"},
                ResourceRequirements(kind="inference", ram_gb=3.0),
                timeout_s=5,
            )
    finally:
        coordinator.close()
        requester.stop()
        worker.stop()
