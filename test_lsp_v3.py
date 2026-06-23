"""Security and seed-independence tests for LSP v3."""

import json
import os
import threading
import time

import pytest
import torch

from src.network.lsp import LSPIdentity
from src.network.lsp_v3 import (
    ENCRYPTED_FLAG,
    MAX_PAYLOAD_SIZE,
    LSPNodeV3,
    PacketType,
    ProtocolError,
    ReplayGuard,
    V3Packet,
)
from src.network.peer_manager import (
    AddressBook,
    PeerReputation,
    network_group,
    validate_peer_address,
)
from src.network.swarm_network import SwarmNetwork
from src.network.useful_work import UsefulWorkCredit, UsefulWorkLedger
from src.network.work_protocol import ResultReceipt, WorkResult
from src.network.node import NodeIdentity
from src.swarm.node_manager import NodeManager


def _packet(identity, *, sequence=1, session_id=b"s" * 16, payload=b"hello"):
    packet = V3Packet.create(
        PacketType.PING,
        payload,
        sequence=sequence,
        session_id=session_id,
    )
    return packet.pack(identity)


def _wait_for(predicate, timeout=8.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def _node(tmp_path, name, *, seeds=None, target_outbound=2):
    return LSPNodeV3(
        LSPIdentity.generate(),
        host="127.0.0.1",
        port=0,
        advertised_host="127.0.0.1",
        seeds=seeds,
        address_book_path=tmp_path / f"{name}.json",
        target_outbound=target_outbound,
        allow_private=True,
        maintenance_interval=0.1,
        resource_profile={"mode": "moderate", "cpu_cores": 4},
    )


def test_v3_signed_packet_roundtrip():
    identity = LSPIdentity.generate()
    raw = _packet(identity, payload=b'{"value":1}')
    packet = V3Packet.unpack(raw)
    assert packet.sender_id == identity.node_id
    assert packet.sequence == 1
    assert packet.payload == b'{"value":1}'


def test_v3_rejects_unsigned_and_tampered_packets():
    identity = LSPIdentity.generate()
    raw = bytearray(_packet(identity))

    unsigned = bytearray(raw)
    unsigned[6] = 0  # flags byte
    with pytest.raises(ProtocolError, match="Unsigned"):
        V3Packet.unpack(bytes(unsigned))

    tampered = bytearray(raw)
    tampered[-1] ^= 0x01
    with pytest.raises(ProtocolError, match="signature"):
        V3Packet.unpack(bytes(tampered))


def test_v3_replay_guard_rejects_duplicates_and_old_sequences():
    identity = LSPIdentity.generate()
    guard = ReplayGuard()
    first = V3Packet.unpack(_packet(identity, sequence=2))
    assert guard.accept(first)
    assert not guard.accept(first)

    older = V3Packet.unpack(_packet(identity, sequence=1))
    assert not guard.accept(older)


def test_v3_payload_bound_is_enforced():
    identity = LSPIdentity.generate()
    with pytest.raises(ProtocolError, match="maximum"):
        V3Packet.create(
            PacketType.GLOBAL_DELTA,
            b"x" * (MAX_PAYLOAD_SIZE + 1),
            sequence=1,
            session_id=os.urandom(16),
        ).pack(identity)


def test_v3_session_payload_is_confidential_and_authenticated(tmp_path):
    left = _node(tmp_path, "crypto-left", target_outbound=0)
    right = _node(tmp_path, "crypto-right", target_outbound=0)
    left.port = 17338
    right.port = 27338
    left_key = left._derive_session_key(
        left._parse_hello(right._hello_payload()),
        right.identity.node_id,
        right.session_id,
    )
    right_key = right._derive_session_key(
        right._parse_hello(left._hello_payload()),
        left.identity.node_id,
        left.session_id,
    )
    assert left_key == right_key
    plaintext = b'{"private_prompt":"not visible on the wire"}'
    raw = left._new_packet(
        PacketType.WORK_OFFER,
        plaintext,
        ttl=1,
        encryption_key=left_key,
    )
    assert plaintext not in raw
    packet = V3Packet.unpack(raw)
    assert packet.flags & ENCRYPTED_FLAG
    assert left._decrypt_packet(packet, right_key).payload == plaintext

    packet = V3Packet.unpack(raw)
    with pytest.raises(ProtocolError, match="encrypted payload"):
        left._decrypt_packet(packet, os.urandom(32))


def test_v3_peer_address_validation(tmp_path):
    assert validate_peer_address("seed.example.org", 7338, allow_private=False)
    assert validate_peer_address("127.0.0.1", 7338, allow_private=True)
    assert not validate_peer_address("127.0.0.1", 7338, allow_private=False)
    assert not validate_peer_address("0.0.0.0", 7338, allow_private=True)
    assert not validate_peer_address("bad host", 7338, allow_private=True)
    assert not validate_peer_address("example.org", 70000, allow_private=False)

    public_book = AddressBook(tmp_path / "public-peers.json", allow_private=False)
    assert public_book.add_many([
        {"host": "attacker-controlled.example", "port": 7338}
    ]) == 0


def test_v3_address_book_persists_without_private_data(tmp_path):
    path = tmp_path / "peers.json"
    book = AddressBook(path, allow_private=True)
    assert book.add("127.0.0.1", 17338, node_id="ab" * 32, source="peer")
    loaded = AddressBook(path, allow_private=True)
    assert loaded.count == 1
    exported = loaded.export()
    assert exported[0]["node_id"] == "ab" * 32
    assert "resources" not in json.loads(path.read_text())["peers"][0]


def test_peer_selection_prefers_network_group_diversity(tmp_path):
    book = AddressBook(tmp_path / "diverse.json", allow_private=False)
    for host in ("8.8.1.1", "8.8.2.2", "1.1.1.1", "9.9.9.9"):
        assert book.add(host, 7338)
    candidates = book.candidates(
        excluded_node_ids=set(),
        excluded_network_groups={network_group("8.8.9.9")},
        limit=4,
    )
    groups = [network_group(candidate.host) for candidate in candidates]
    assert network_group("8.8.9.9") not in groups[:2]
    assert len(set(groups[:2])) == 2


def test_peer_reputation_persists_hashed_local_bans(tmp_path):
    path = tmp_path / "reputation.json"
    reputation = PeerReputation(path)
    peer_ip = "203.0.113.77"
    for _ in range(4):
        reputation.report("ip", peer_ip, points=30)
    assert reputation.is_banned("ip", peer_ip)
    assert peer_ip not in path.read_text()
    assert PeerReputation(path).is_banned("ip", peer_ip)


def test_v3_persistent_session_carries_pheromone(tmp_path):
    left = _node(tmp_path, "left", target_outbound=0)
    right = _node(tmp_path, "right", target_outbound=0)
    received = []
    event = threading.Event()

    @right.on_feromon_received
    def capture(pheromone, node_id):
        received.append((torch.as_tensor(pheromone), node_id))
        event.set()

    left.start()
    right.start()
    try:
        assert left.connect_peer("127.0.0.1", right.port)
        assert _wait_for(lambda: left.peer_count == 1 and right.peer_count == 1)
        value = torch.randn(256)
        assert left.broadcast_feromon(value, fitness=0.8, step=42) == 1
        assert event.wait(3.0)
        similarity = torch.nn.functional.cosine_similarity(
            value.unsqueeze(0), received[0][0].unsqueeze(0)
        ).item()
        assert similarity >= 0.999
        assert received[0][1] == left.identity.node_id_hex
    finally:
        left.stop()
        right.stop()


def test_v3_delivers_useful_work_credit_to_exact_peer(tmp_path):
    left = _node(tmp_path, "credit-left", target_outbound=0)
    right = _node(tmp_path, "credit-right", target_outbound=0)
    received = []
    event = threading.Event()

    @right.on_useful_work_credit
    def capture(credit, node_id):
        received.append((credit, node_id))
        event.set()

    left.start()
    right.start()
    try:
        assert left.connect_peer("127.0.0.1", right.port)
        assert _wait_for(lambda: left.peer_count == 1 and right.peer_count == 1)
        assert left.send_useful_work_credit(
            right.identity.node_id_hex, {"credit_id": "01" * 32}
        )
        assert event.wait(3.0)
        assert received == [(
            {"credit_id": "01" * 32}, left.identity.node_id_hex
        )]
    finally:
        left.stop()
        right.stop()


def test_v3_can_send_immediate_result_from_transport_callback(tmp_path):
    requester = _node(tmp_path, "callback-requester", target_outbound=0)
    worker = _node(tmp_path, "callback-worker", target_outbound=0)
    received = []
    event = threading.Event()

    @worker.on_work_offer_received
    def reject_immediately(offer, node_id):
        assert worker.send_work_result(node_id, {
            "job_id": offer["job_id"],
            "status": "rejected",
            "error": "work_queue_full",
        })

    @requester.on_work_result_received
    def capture(result, node_id):
        received.append((result, node_id))
        event.set()

    requester.start()
    worker.start()
    try:
        assert requester.connect_peer("127.0.0.1", worker.port)
        assert requester.send_work_offer(worker.identity.node_id_hex, {
            "job_id": "16" * 32,
        })
        assert event.wait(3.0)
        assert received[0][0]["error"] == "work_queue_full"
        assert received[0][1] == worker.identity.node_id_hex
    finally:
        requester.stop()
        worker.stop()


def test_v3_exchanges_useful_work_proofs_after_encrypted_handshake(tmp_path):
    left = _node(tmp_path, "proof-left", target_outbound=0)
    right = _node(tmp_path, "proof-right", target_outbound=0)
    bundle = {
        "version": 1,
        "worker_node_id": left.identity.node_id_hex,
        "credits": [],
    }
    received = []
    event = threading.Event()
    left.set_useful_work_proof_provider(lambda: bundle)

    left.start()
    right.start()
    try:
        assert left.connect_peer("127.0.0.1", right.port)
        assert _wait_for(
            lambda: left.identity.node_id_hex
            in right._latest_useful_work_proofs
        )

        @right.on_useful_work_proofs
        def capture(proofs, node_id):
            received.append((proofs, node_id))
            event.set()

        assert event.wait(3.0)
        assert received == [(bundle, left.identity.node_id_hex)]
    finally:
        left.stop()
        right.stop()


def test_v3_hello_cannot_self_declare_useful_work(tmp_path):
    node = _node(tmp_path, "untrusted-reputation", target_outbound=0)
    node.port = 17338
    node.resource_profile["useful_work"] = {
        "verified": True, "firsthand_credits": 16
    }
    parsed = node._parse_hello(node._hello_payload())
    assert "useful_work" not in parsed["resources"]


def test_v3_network_continues_after_seed_shutdown(tmp_path):
    seed = _node(tmp_path, "seed", target_outbound=0)
    seed.start()
    left = _node(
        tmp_path,
        "left",
        seeds=[("127.0.0.1", seed.port)],
        target_outbound=2,
    )
    right = _node(
        tmp_path,
        "right",
        seeds=[("127.0.0.1", seed.port)],
        target_outbound=2,
    )
    received = []
    event = threading.Event()

    @right.on_gossip_delta_received
    def capture(delta, node_id):
        received.append((delta, node_id))
        event.set()

    left.start()
    right.start()
    try:
        assert _wait_for(
            lambda: any(
                peer["node_id"] == right.identity.node_id_hex
                for peer in left.peers()
            )
            and any(
                peer["node_id"] == left.identity.node_id_hex
                for peer in right.peers()
            )
        ), "nodes did not discover a direct route through peer exchange"

        seed.stop()
        assert _wait_for(lambda: left.peer_count >= 1 and right.peer_count >= 1)

        delta = {"kind": "matriarca_global_delta", "version": 1, "count": 0}
        assert left.broadcast_global_delta(delta) >= 1
        assert event.wait(3.0)
        assert received[0][0] == delta
        assert received[0][1] == left.identity.node_id_hex
    finally:
        try:
            seed.stop()
        except Exception:
            pass
        left.stop()
        right.stop()


def test_swarm_network_uses_v3_by_default(tmp_path):
    left = SwarmNetwork.create(
        mode="lan",
        gossip_port=0,
        checkpoint_dir=str(tmp_path / "left-network"),
        target_outbound=0,
    )
    right = SwarmNetwork.create(
        mode="lan",
        gossip_port=0,
        checkpoint_dir=str(tmp_path / "right-network"),
        target_outbound=0,
    )
    left.start()
    right.start()
    try:
        assert left.protocol == "v3"
        assert left._lsp_v3_node is not None
        assert right._lsp_v3_node is not None
        assert left.connect_peer("127.0.0.1", right._lsp_v3_node.port)
        assert _wait_for(lambda: left.peer_count == 1 and right.peer_count == 1)

        value = torch.randn(64)
        assert left.broadcast_feromon(value) == 1
        assert _wait_for(lambda: len(right.collect_feromons()) == 1)
        received = right.collect_feromons()[0]
        similarity = torch.nn.functional.cosine_similarity(
            value.unsqueeze(0), received.unsqueeze(0)
        ).item()
        assert similarity >= 0.999
    finally:
        left.stop()
        right.stop()


def test_swarm_network_verifies_firsthand_training_evidence(tmp_path):
    worker_network = SwarmNetwork.create(
        mode="lan", gossip_port=0,
        checkpoint_dir=str(tmp_path / "evidence-worker"), target_outbound=0,
    )
    requester_network = SwarmNetwork.create(
        mode="lan", gossip_port=0,
        checkpoint_dir=str(tmp_path / "evidence-requester"), target_outbound=0,
    )
    worker_network.start()
    requester_network.start()
    try:
        worker = worker_network._lsp_v3_node.identity
        requester = requester_network._lsp_v3_node.identity
        result = WorkResult(
            job_id="11" * 32,
            status="ok",
            output={"gradient_artifact_id": "12" * 32},
        )
        receipt = ResultReceipt.create(
            result, worker, requester.node_id_hex
        ).to_dict()
        credit = UsefulWorkCredit.issue(
            requester,
            worker_node_id=worker.node_id_hex,
            receipt=receipt,
            gradient_artifact_id="12" * 32,
            aggregate_artifact_id="13" * 32,
            model_artifact_id="14" * 32,
            dataset_artifact_id="15" * 32,
            token_count=256,
        )
        worker_ledger = UsefulWorkLedger(
            tmp_path / "worker-credits.json", worker.node_id_hex
        )
        assert worker_ledger.add(credit.to_dict())
        worker_network.attach_useful_work_ledger(worker_ledger)
        requester_network.attach_useful_work_ledger(UsefulWorkLedger(
            tmp_path / "requester-credits.json", requester.node_id_hex
        ))

        assert requester_network.connect_peer(
            "127.0.0.1", worker_network._lsp_v3_node.port
        )
        assert _wait_for(lambda: bool(
            requester_network._lsp_v3_node.peers()
            and requester_network._lsp_v3_node.peers()[0]
            .get("resources", {}).get("useful_work", {})
            .get("firsthand_credits") == 1
        ))
        evidence = requester_network._lsp_v3_node.peers()[0]["resources"][
            "useful_work"
        ]
        assert evidence["verified"]
        assert evidence["firsthand_tokens"] == 256
    finally:
        worker_network.stop()
        requester_network.stop()


def test_v3_peer_resources_join_runtime_node_manager(tmp_path):
    class RuntimeOnlySwarm:
        def __init__(self):
            self.node_manager = NodeManager()

    runtime_swarm = RuntimeOnlySwarm()
    left = SwarmNetwork(
        identity=NodeIdentity.generate_anonymous(gossip_port=0),
        mode="lan",
        checkpoint_dir=str(tmp_path / "runtime-left"),
        swarm=runtime_swarm,
        target_outbound=0,
    )
    right = SwarmNetwork.create(
        mode="lan",
        gossip_port=0,
        checkpoint_dir=str(tmp_path / "runtime-right"),
        target_outbound=0,
    )
    left.start()
    right.start()
    try:
        assert left.connect_peer("127.0.0.1", right._lsp_v3_node.port)
        remote_id = right._lsp_v3_node.identity.node_id_hex
        assert _wait_for(lambda: runtime_swarm.node_manager.get_node(remote_id) is not None)
        record = runtime_swarm.node_manager.get_node(remote_id)
        assert record.hardware.cpu_cores >= 1
        assert record.contribution_mode.value in {"maximum", "moderate", "relay"}
    finally:
        right.stop()
        assert _wait_for(lambda: runtime_swarm.node_manager.n_nodes() == 0)
        left.stop()
