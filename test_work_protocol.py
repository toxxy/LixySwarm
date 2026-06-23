import json
import threading
import time

import pytest
from dataclasses import replace

from src.contribution import ContributionPolicy, ResourceGovernor, ResourceRequirements
from src.network.lsp import LSPIdentity
from src.network.lsp_v3 import LSPNodeV3
from src.network.identity_work import load_or_mine_identity_work
from src.network.work_protocol import (
    ResultReceipt,
    WorkCancelledError,
    WorkCoordinator,
    WorkExecution,
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


def _wait_for(predicate, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


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


def test_compute_scheduler_requires_configured_identity_work(tmp_path):
    valid_identity = LSPIdentity.generate()
    invalid_identity = LSPIdentity.generate()
    proof = load_or_mine_identity_work(
        tmp_path / "valid-proof.json", valid_identity.node_id_hex, bits=8
    )

    class FakeNode:
        identity = LSPIdentity.generate()

        def on_work_offer_received(self, _callback):
            return None

        def on_work_result_received(self, _callback):
            return None

        def peers(self):
            base = {
                "work": {"inference": True}, "cpu_cores": 4,
                "ram_gb": 4, "disk_gb": 4, "gpu_vram_gb": 0,
                "has_gpu": False,
            }
            return [
                {"node_id": invalid_identity.node_id_hex, "host": "8.8.8.8",
                 "resources": {**base, "identity_work": {}}},
                {"node_id": valid_identity.node_id_hex, "host": "1.1.1.1",
                 "resources": {**base, "identity_work": proof}},
            ]

    coordinator = WorkCoordinator(
        FakeNode(), _governor(tmp_path / "governor", "relay"),
        minimum_identity_work_bits=8,
    )
    try:
        assert coordinator.select_peer(
            ResourceRequirements(kind="inference")
        ) == valid_identity.node_id_hex
    finally:
        coordinator.close()


def test_scheduler_prioritizes_verified_useful_work(tmp_path):
    experienced = LSPIdentity.generate()
    unproven = LSPIdentity.generate()

    class FakeNode:
        identity = LSPIdentity.generate()

        def on_work_offer_received(self, _callback):
            return None

        def on_work_result_received(self, _callback):
            return None

        def peers(self):
            base = {
                "work": {"training": True}, "cpu_cores": 4,
                "ram_gb": 8, "disk_gb": 8, "has_gpu": True,
            }
            return [
                {
                    "node_id": unproven.node_id_hex,
                    "host": "8.8.8.8",
                    "resources": {**base, "gpu_vram_gb": 24},
                },
                {
                    "node_id": experienced.node_id_hex,
                    "host": "1.1.1.1",
                    "resources": {
                        **base,
                        "gpu_vram_gb": 8,
                        "useful_work": {
                            "verified": True,
                            "firsthand_credits": 1,
                            "firsthand_tokens": 512,
                            "distinct_issuers": 1,
                            "presented_credits": 1,
                            "validated_tokens": 512,
                        },
                    },
                },
            ]

    coordinator = WorkCoordinator(
        FakeNode(), _governor(tmp_path / "useful-governor", "relay")
    )
    try:
        assert coordinator.select_peer(
            ResourceRequirements(kind="training")
        ) == experienced.node_id_hex
    finally:
        coordinator.close()


def test_scheduler_gives_aged_new_identity_bounded_exploration(tmp_path):
    experienced = LSPIdentity.generate()
    newcomer = LSPIdentity.generate()

    class FakeNode:
        identity = LSPIdentity.generate()

        def on_work_offer_received(self, _callback):
            return None

        def on_work_result_received(self, _callback):
            return None

        def peers(self):
            base = {
                "work": {"training": True}, "cpu_cores": 4,
                "ram_gb": 8, "disk_gb": 8, "has_gpu": True,
                "gpu_vram_gb": 8,
            }
            return [
                {
                    "node_id": experienced.node_id_hex,
                    "host": "8.8.8.8",
                    "resources": {
                        **base,
                        "useful_work": {
                            "verified": True,
                            "firsthand_credits": 2,
                            "firsthand_tokens": 1024,
                        },
                    },
                },
                {
                    "node_id": newcomer.node_id_hex,
                    "host": "1.1.1.1",
                    "resources": base,
                },
            ]

    state_path = tmp_path / "scheduler-state.json"
    coordinator = WorkCoordinator(
        FakeNode(),
        _governor(tmp_path / "exploration-governor", "relay"),
        scheduler_state_path=state_path,
        exploration_interval=5,
        exploration_minimum_age_s=0,
    )
    try:
        selected = [
            coordinator.select_peer(ResourceRequirements(kind="training"))
            for _ in range(6)
        ]
        assert selected[:4] == [experienced.node_id_hex] * 4
        assert selected[4] == newcomer.node_id_hex
        assert selected[5] == experienced.node_id_hex
    finally:
        coordinator.close()

    persisted = json.loads(state_path.read_text())
    assert persisted["dispatch_count"] == 6
    assert persisted["peers"][newcomer.node_id_hex]["selections"] == 1
    assert "host" not in state_path.read_text()


def test_quorum_exploration_does_not_reduce_network_group_diversity(tmp_path):
    identities = [LSPIdentity.generate() for _ in range(4)]

    class FakeNode:
        identity = LSPIdentity.generate()

        def on_work_offer_received(self, _callback):
            return None

        def on_work_result_received(self, _callback):
            return None

        def peers(self):
            hosts = ["8.8.1.1", "1.1.1.1", "9.9.9.9", "8.8.2.2"]
            credits = [4, 3, 2, 0]
            return [{
                "node_id": identity.node_id_hex,
                "host": host,
                "resources": {
                    "work": {"training": True},
                    "cpu_cores": 4,
                    "ram_gb": 8,
                    "disk_gb": 8,
                    "has_gpu": True,
                    "gpu_vram_gb": 8,
                    "useful_work": {
                        "verified": True,
                        "firsthand_credits": credit_count,
                    },
                },
            } for identity, host, credit_count in zip(identities, hosts, credits)]

    coordinator = WorkCoordinator(
        FakeNode(),
        _governor(tmp_path / "diversity-governor", "relay"),
        exploration_interval=1,
        exploration_minimum_age_s=0,
    )
    try:
        selected = coordinator.select_peers(
            ResourceRequirements(kind="training"), limit=3
        )
        assert selected == [identity.node_id_hex for identity in identities[:3]]
    finally:
        coordinator.close()


def test_inbound_work_queue_is_bounded_per_identity(tmp_path):
    worker = LSPIdentity.generate()
    requester = LSPIdentity.generate()

    class FakeNode:
        identity = worker

        def __init__(self):
            self.results = []

        def on_work_offer_received(self, callback):
            self.offer_callback = callback

        def on_work_result_received(self, callback):
            self.result_callback = callback

        def send_work_result(self, peer_id, value):
            self.results.append((peer_id, value))
            return True

    node = FakeNode()
    started = threading.Event()
    release = threading.Event()
    coordinator = WorkCoordinator(
        node,
        _governor(tmp_path / "queue-governor", "balanced"),
        max_workers=1,
        max_queued_offers=1,
        max_offers_per_peer=2,
        max_offers_per_minute=10,
    )
    coordinator.register_handler(
        "training.block.v1",
        "training",
        lambda _payload, _work: (
            started.set(), release.wait(3), {"accepted": True}
        )[-1],
    )

    def make_offer(index):
        return WorkUnit.create(
            origin_node_id=requester.node_id_hex,
            operation="training.block.v1",
            kind="training",
            requirements=ResourceRequirements(kind="training"),
            payload={"batch": index},
        ).to_dict()

    offers = [make_offer(index) for index in range(3)]
    try:
        node.offer_callback(offers[0], requester.node_id_hex)
        assert started.wait(1.0)
        node.offer_callback(offers[1], requester.node_id_hex)
        node.offer_callback(offers[2], requester.node_id_hex)
        assert _wait_for(lambda: any(
            value["job_id"] == offers[2]["job_id"]
            for _, value in node.results
        ))
        rejected = next(
            WorkResult.from_dict(value) for _, value in node.results
            if value["job_id"] == offers[2]["job_id"]
        )
        assert rejected.status == "rejected"
        assert rejected.error == "peer_queue_limit"
        receipt = ResultReceipt.from_dict(rejected.receipt)
        assert receipt.verify(
            rejected,
            expected_worker_id=worker.node_id_hex,
            expected_requester_id=requester.node_id_hex,
        )
        assert coordinator.queue_status()["active_or_queued"] == 2
    finally:
        release.set()
        coordinator.close()
    assert coordinator.queue_status()["active_or_queued"] == 0


def test_global_work_queue_and_offer_rate_are_bounded(tmp_path):
    worker = LSPIdentity.generate()
    first_requester = LSPIdentity.generate()
    second_requester = LSPIdentity.generate()

    class FakeNode:
        identity = worker

        def __init__(self):
            self.results = []

        def on_work_offer_received(self, callback):
            self.offer_callback = callback

        def on_work_result_received(self, callback):
            self.result_callback = callback

        def send_work_result(self, peer_id, value):
            self.results.append((peer_id, value))
            return True

    node = FakeNode()
    started = threading.Event()
    release = threading.Event()
    coordinator = WorkCoordinator(
        node,
        _governor(tmp_path / "global-queue-governor", "balanced"),
        max_workers=1,
        max_queued_offers=0,
        max_offers_per_peer=2,
        max_offers_per_minute=1,
    )
    coordinator.register_handler(
        "training.block.v1",
        "training",
        lambda _payload, _work: (
            started.set(), release.wait(3), {"accepted": True}
        )[-1],
    )

    def make_offer(identity, index):
        return WorkUnit.create(
            origin_node_id=identity.node_id_hex,
            operation="training.block.v1",
            kind="training",
            requirements=ResourceRequirements(kind="training"),
            payload={"batch": index},
        ).to_dict()

    first = make_offer(first_requester, 1)
    rate_limited = make_offer(first_requester, 2)
    queue_limited = make_offer(second_requester, 3)
    try:
        node.offer_callback(first, first_requester.node_id_hex)
        assert started.wait(1.0)
        node.offer_callback(rate_limited, first_requester.node_id_hex)
        node.offer_callback(queue_limited, second_requester.node_id_hex)
        assert _wait_for(lambda: len(node.results) >= 2)
        errors = {
            value["job_id"]: value["error"] for _, value in node.results
        }
        assert errors[rate_limited["job_id"]] == "peer_rate_limit"
        assert errors[queue_limited["job_id"]] == "work_queue_full"
        assert coordinator.queue_status()["capacity"] == 1
    finally:
        release.set()
        coordinator.close()


def test_requester_cancellation_stops_cooperative_remote_work(tmp_path):
    class LoopNode:
        def __init__(self):
            self.identity = LSPIdentity.generate()
            self.peer = None

        def on_work_offer_received(self, callback):
            self.offer_callback = callback

        def on_work_result_received(self, callback):
            self.result_callback = callback

        def on_work_cancel_received(self, callback):
            self.cancel_callback = callback

        def send_work_offer(self, peer_id, value):
            assert peer_id == self.peer.identity.node_id_hex
            self.peer.offer_callback(value, self.identity.node_id_hex)
            return True

        def send_work_result(self, peer_id, value):
            assert peer_id == self.peer.identity.node_id_hex
            self.peer.result_callback(value, self.identity.node_id_hex)
            return True

        def send_work_cancel(self, peer_id, value):
            assert peer_id == self.peer.identity.node_id_hex
            self.peer.cancel_callback(value, self.identity.node_id_hex)
            return True

        def peers(self):
            return []

    requester_node = LoopNode()
    worker_node = LoopNode()
    requester_node.peer = worker_node
    worker_node.peer = requester_node
    requester = WorkCoordinator(
        requester_node, _governor(tmp_path / "cancel-requester", "relay")
    )
    worker = WorkCoordinator(
        worker_node, _governor(tmp_path / "cancel-worker", "balanced")
    )
    started = threading.Event()
    stopped = threading.Event()

    def cancellable_handler(_payload, work):
        started.set()
        try:
            while True:
                work.raise_if_cancelled()
                time.sleep(0.01)
        except WorkCancelledError:
            stopped.set()
            raise

    worker.register_handler("training.cancel.v1", "training", cancellable_handler)
    caller_cancel = threading.Event()
    returned = []

    def submit():
        returned.append(requester.submit(
            "training.cancel.v1",
            {"batch": 1},
            ResourceRequirements(kind="training"),
            peer_id=worker_node.identity.node_id_hex,
            timeout_s=10,
            cancel_event=caller_cancel,
        ))

    thread = threading.Thread(target=submit)
    thread.start()
    try:
        assert started.wait(1.0)
        with worker._lock:
            active_job_id = next(iter(worker._job_cancel_events))[1]
        worker._on_cancel({
            "version": 1,
            "job_id": active_job_id,
            "requested_at": time.time(),
            "reason": "requester_cancelled",
        }, LSPIdentity.generate().node_id_hex)
        time.sleep(0.05)
        assert not stopped.is_set()
        caller_cancel.set()
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert returned[0].error == "work_cancelled"
        assert stopped.wait(1.0)
        assert _wait_for(lambda: worker.queue_status()["active_or_queued"] == 0)
        with worker._lock:
            completed = next(iter(worker._completed.values()))
        assert completed.status == "error"
        assert completed.error == "work_cancelled"
        receipt = ResultReceipt.from_dict(completed.receipt)
        assert receipt.verify(
            completed,
            expected_worker_id=worker_node.identity.node_id_hex,
            expected_requester_id=requester_node.identity.node_id_hex,
        )
    finally:
        caller_cancel.set()
        thread.join(timeout=2.0)
        requester.close()
        worker.close()


def test_work_execution_enforces_deadline():
    work = WorkUnit.create(
        origin_node_id="ab" * 32,
        operation="training.deadline.v1",
        kind="training",
        requirements=ResourceRequirements(kind="training"),
        payload={},
        timeout_s=1,
    )
    execution = WorkExecution(
        replace(work, deadline=time.time() - 1), threading.Event()
    )
    assert execution.cancelled()
    assert execution.cancellation_reason() == "work_deadline_exceeded"
    with pytest.raises(WorkCancelledError, match="work_deadline_exceeded"):
        execution.raise_if_cancelled()
