import asyncio
import time

import pytest

from src.network.lsp import LSPIdentity
from src.network.lsp_v3 import ENCRYPTED_FLAG, LSPNodeV3, PacketType, ProtocolError, V3Packet
from src.network.relay import DATA, HEADER
from src.contribution import ContributionPolicy, ResourceGovernor, ResourceRequirements
from src.network.work_protocol import ResultReceipt, WorkCoordinator
from src.swarm.node_manager import HardwareProfile


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def node(tmp_path, name, *, relay=False, per_peer=4):
    return LSPNodeV3(
        LSPIdentity.generate(), host="127.0.0.1", port=0,
        address_book_path=tmp_path / f"{name}.json", target_outbound=0,
        allow_private=True, relay_enabled=relay,
        max_relay_streams_per_peer=per_peer, maintenance_interval=0.05,
    )


def on_loop(node, coroutine):
    return asyncio.run_coroutine_threadsafe(coroutine, node._loop).result(timeout=8)


def connect_relay(source, target):
    return on_loop(source, source._connect_via_relay(
        target.identity.node_id_hex, "127.0.0.1", target.port
    ))


def test_nat_fallback_keeps_end_to_end_identity_and_ciphertext(tmp_path):
    relay = node(tmp_path, "relay", relay=True)
    a, b = node(tmp_path, "a"), node(tmp_path, "b")
    observed = []
    forwarded = []
    original_send = relay.relay.send

    async def capture(peer, operation, stream, remote, data=b""):
        if operation == DATA:
            forwarded.append(data)
        return await original_send(peer, operation, stream, remote, data)

    relay.relay.send = capture
    a.on_gossip_delta_received(lambda value, sender: observed.append((value, sender)))
    try:
        for peer in (relay, a, b):
            peer.start()
        for peer in (a, b):
            assert peer.connect_peer("127.0.0.1", relay.port)

            original_connect = peer._connect_address

            async def nat_connect(host, port, original=original_connect):
                if port == relay.port:
                    return await original(host, port)
                return False

            peer._connect_address = nat_connect
        # The normal peer manager sees the advertised worker, fails direct TCP,
        # and must recover through a relay; the test does not call the fallback.
        b.peer_manager.target_outbound = 2
        b.peer_manager.add_discovered([{
            "host": "127.0.0.1", "port": a.port, "node_id": a.identity.node_id_hex,
        }])
        assert wait_for(lambda: a.identity.node_id_hex in b.connected_node_ids)
        snapshot = next(p for p in b.peers() if p["node_id"] == a.identity.node_id_hex)
        assert snapshot["transport"] == "relay"
        assert snapshot["relay_node_id"] == relay.identity.node_id_hex
        assert snapshot["encrypted"] is True

        secret = "test-memory-end-to-end-73c8b2"
        b.broadcast_global_delta({"memory": secret})
        assert wait_for(lambda: ({"memory": secret}, b.identity.node_id_hex) in observed)
        assert secret.encode() not in b"".join(forwarded)
        encrypted = []
        for chunk in forwarded:
            try:
                packet = V3Packet.unpack(chunk[4:])
            except ProtocolError:
                continue
            if packet.packet_type == PacketType.GLOBAL_DELTA:
                encrypted.append(packet)
        assert encrypted and all(packet.flags & ENCRYPTED_FLAG for packet in encrypted)
        with pytest.raises(ProtocolError, match="encrypted payload"):
            relay._decrypt_packet(encrypted[0], relay._sessions[b.identity.node_id_hex].encryption_key)

        large = {"blob": "segment-" * 25_000}
        b.broadcast_global_delta(large)
        assert wait_for(lambda: (large, b.identity.node_id_hex) in observed)
        assert relay.relay.forwarded_bytes > 200_000
    finally:
        for peer in (b, a, relay):
            peer.stop()


def test_relay_disabled_or_missing_destination_cannot_open_stream(tmp_path):
    relay = node(tmp_path, "relay")
    a, b = node(tmp_path, "a"), node(tmp_path, "b")
    try:
        for peer in (relay, a, b):
            peer.start()
        for peer in (a, b):
            assert peer.connect_peer("127.0.0.1", relay.port)
        assert not connect_relay(a, b)
        assert not relay.relay.routes
        assert not a.relay.streams
    finally:
        for peer in (b, a, relay):
            peer.stop()


def test_unrelated_peer_cannot_inject_into_relay_route(tmp_path):
    relay = node(tmp_path, "relay", relay=True)
    a, b, attacker = [node(tmp_path, name) for name in ("a", "b", "attacker")]
    observed = []
    b.on_gossip_delta_received(lambda value, sender: observed.append((value, sender)))
    try:
        for peer in (relay, a, b, attacker):
            peer.start()
        for peer in (a, b, attacker):
            assert peer.connect_peer("127.0.0.1", relay.port)
        assert connect_relay(a, b)
        stream = next(iter(relay.relay.routes))
        forged = HEADER.pack(DATA, stream, b.identity.node_id) + b"forged"
        assert on_loop(attacker, attacker._send_to_peer(
            PacketType.RELAY, relay.identity.node_id_hex, forged
        ))
        assert wait_for(lambda: attacker.identity.node_id_hex not in relay.connected_node_ids)
        a.broadcast_global_delta({"still": "valid"})
        assert wait_for(lambda: ({"still": "valid"}, a.identity.node_id_hex) in observed)
    finally:
        for peer in (attacker, b, a, relay):
            peer.stop()


def test_relay_capacity_and_disconnect_release_streams(tmp_path):
    relay = node(tmp_path, "relay", relay=True, per_peer=1)
    a, b, c = [node(tmp_path, name) for name in ("a", "b", "c")]
    try:
        for peer in (relay, a, b, c):
            peer.start()
        for peer in (a, b, c):
            assert peer.connect_peer("127.0.0.1", relay.port)
        assert connect_relay(a, b)
        assert not connect_relay(a, c)
        assert len(relay.relay.routes) == 1
        b.stop()
        assert wait_for(lambda: not relay.relay.routes)
        assert wait_for(lambda: b.identity.node_id_hex not in a.connected_node_ids)
        assert connect_relay(a, c)
        relay.stop()
        assert wait_for(lambda: not a.relay.streams and not c.relay.streams)
    finally:
        for peer in (c, b, a, relay):
            peer.stop()


def test_work_over_relay_keeps_worker_and_requester_receipt_identities(tmp_path):
    relay = node(tmp_path, "relay", relay=True)
    a, b = node(tmp_path, "a"), node(tmp_path, "b")
    hardware = HardwareProfile(cpu_cores=4, ram_gb=8, gpu_vram_gb=0, disk_gb=20, has_gpu=False)
    governors = []
    for peer, name in ((a, "a"), (b, "b")):
        governor = ResourceGovernor(
            ContributionPolicy.for_mode("balanced"), hardware=hardware,
            storage_path=tmp_path / name,
        )
        governor._system_busy = lambda: False
        peer.resource_profile = governor.advertised_profile()
        governors.append(governor)
    requester, worker = WorkCoordinator(a, governors[0]), WorkCoordinator(b, governors[1])
    worker.register_handler("inference.echo.v1", "inference", lambda value, _: {"text": value["prompt"].upper()})
    try:
        for peer in (relay, a, b):
            peer.start()
        for peer in (a, b):
            assert peer.connect_peer("127.0.0.1", relay.port)
        assert connect_relay(a, b)
        result = requester.submit(
            "inference.echo.v1", {"prompt": "through relay"},
            ResourceRequirements(kind="inference", ram_gb=0.25),
            peer_id=b.identity.node_id_hex, timeout_s=5,
        )
        assert result.status == "ok"
        assert result.output == {"text": "THROUGH RELAY"}
        assert ResultReceipt.from_dict(result.receipt).verify(
            result, expected_worker_id=b.identity.node_id_hex,
            expected_requester_id=a.identity.node_id_hex,
        )
    finally:
        requester.close()
        worker.close()
        for peer in (b, a, relay):
            peer.stop()


def test_tampering_relay_cannot_forge_endpoint_payload(tmp_path):
    relay = node(tmp_path, "relay", relay=True)
    a, b = node(tmp_path, "a"), node(tmp_path, "b")
    observed = []
    b.on_gossip_delta_received(lambda value, _: observed.append(value))
    try:
        for peer in (relay, a, b):
            peer.start()
        for peer in (a, b):
            assert peer.connect_peer("127.0.0.1", relay.port)
        assert connect_relay(a, b)
        original_send = relay.relay.send

        async def corrupt(peer, operation, stream, remote, data=b""):
            if operation == DATA:
                try:
                    inner = V3Packet.unpack(data[4:])
                except ProtocolError:
                    inner = None
                if inner is not None and inner.packet_type == PacketType.GLOBAL_DELTA:
                    changed = bytearray(data)
                    changed[-1] ^= 1
                    data = bytes(changed)
            return await original_send(peer, operation, stream, remote, data)

        relay.relay.send = corrupt
        a.broadcast_global_delta({"forged": "must not arrive"})
        assert wait_for(lambda: a.identity.node_id_hex not in b.connected_node_ids)
        assert not observed
        assert relay.identity.node_id_hex in b.connected_node_ids
    finally:
        for peer in (b, a, relay):
            peer.stop()


def test_clients_recover_through_another_relay_after_failure(tmp_path):
    r1, r2 = node(tmp_path, "r1", relay=True), node(tmp_path, "r2", relay=True)
    a, b = node(tmp_path, "a"), node(tmp_path, "b")
    observed = []
    a.on_gossip_delta_received(lambda value, sender: observed.append((value, sender)))
    try:
        for peer in (r1, r2, a, b):
            peer.start()
        for peer in (a, b):
            for relay in (r1, r2):
                assert peer.connect_peer("127.0.0.1", relay.port)
            original_connect = peer._connect_address

            async def nat_connect(host, port, original=original_connect):
                if port in {r1.port, r2.port}:
                    return await original(host, port)
                return False

            peer._connect_address = nat_connect
        b.peer_manager.target_outbound = 3
        b.peer_manager.add_discovered([{
            "host": "127.0.0.1", "port": a.port, "node_id": a.identity.node_id_hex,
        }])
        assert wait_for(lambda: a.identity.node_id_hex in b.connected_node_ids)
        r1.stop()
        assert wait_for(lambda: any(
            p["node_id"] == a.identity.node_id_hex and p["relay_node_id"] == r2.identity.node_id_hex
            for p in b.peers()
        ))
        b.broadcast_global_delta({"after": "relay failover"})
        assert wait_for(lambda: ({"after": "relay failover"}, b.identity.node_id_hex) in observed)
    finally:
        for peer in (b, a, r2, r1):
            peer.stop()


def test_direct_connection_replaces_relay_without_losing_new_session(tmp_path):
    relay = node(tmp_path, "relay", relay=True)
    a, b = node(tmp_path, "a"), node(tmp_path, "b")
    observed = []
    b.on_gossip_delta_received(lambda value, sender: observed.append((value, sender)))
    try:
        for peer in (relay, a, b):
            peer.start()
        for peer in (a, b):
            assert peer.connect_peer("127.0.0.1", relay.port)
        assert connect_relay(a, b)
        assert a.connect_peer("127.0.0.1", b.port)
        assert wait_for(lambda: all(
            p["transport"] == "direct" for p in a.peers() + b.peers()
        ))
        assert wait_for(lambda: not relay.relay.routes)
        a.broadcast_global_delta({"direct": "wins"})
        assert wait_for(lambda: ({"direct": "wins"}, a.identity.node_id_hex) in observed)
    finally:
        for peer in (b, a, relay):
            peer.stop()


def test_relay_cannot_substitute_another_signed_hello_identity(tmp_path):
    relay = node(tmp_path, "relay", relay=True)
    a, b, impostor = [node(tmp_path, name) for name in ("a", "b", "impostor")]
    impostor.port = 65535  # Valid HELLO metadata; only the signed identity is wrong.
    original_send = relay.relay.send

    async def substitute(peer, operation, stream, remote, data=b""):
        if operation == DATA:
            try:
                inner = V3Packet.unpack(data[4:])
            except ProtocolError:
                inner = None
            if inner is not None and inner.packet_type == PacketType.HELLO:
                raw = impostor._new_packet(PacketType.HELLO, impostor._hello_payload(), ttl=1)
                data = len(raw).to_bytes(4, "big") + raw
        return await original_send(peer, operation, stream, remote, data)

    relay.relay.send = substitute
    try:
        for peer in (relay, a, b):
            peer.start()
        for peer in (a, b):
            assert peer.connect_peer("127.0.0.1", relay.port)
        assert not connect_relay(a, b)
        assert impostor.identity.node_id_hex not in a.connected_node_ids | b.connected_node_ids
        assert wait_for(lambda: not relay.relay.routes)
    finally:
        for peer in (b, a, relay):
            peer.stop()
