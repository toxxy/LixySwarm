"""
test_lsp.py — Tests unitarios para LixySwarm Protocol (LSP) v1
"""
import sys, os, time, threading
sys.path.insert(0, os.path.dirname(__file__))

import unittest
from src.network.lsp import LSPIdentity, LSPPacket, LSPNode, PacketType, Flags

class TestLSPIdentity(unittest.TestCase):

    def test_generate_sign_verify(self):
        """Test 1: Generar identidad → sign → verify ✅"""
        identity = LSPIdentity.generate()
        self.assertEqual(len(identity.node_id), 32)
        self.assertEqual(len(identity.node_id_hex), 64)

        data = b"LixySwarm test data"
        sig = identity.sign(data)
        self.assertEqual(len(sig), 64)

        # Verify con la misma clave
        ok = identity.verify(data, sig, identity.node_id)
        self.assertTrue(ok, "Signature should verify correctly")

        # Verify con datos alterados → False
        bad = identity.verify(b"wrong data", sig, identity.node_id)
        self.assertFalse(bad, "Tampered data should not verify")
        print("✅ Test 1: generate + sign + verify OK")

    def test_save_load(self):
        """Test guardado/cargado de identidad."""
        import tempfile
        identity = LSPIdentity.generate()
        with tempfile.NamedTemporaryFile(suffix=".key", delete=False) as f:
            path = f.name
        try:
            identity.save(path)
            loaded = LSPIdentity.load(path)
            self.assertIsNotNone(loaded)
            self.assertEqual(identity.node_id_hex, loaded.node_id_hex)
            # Sign with original, verify with loaded
            sig = identity.sign(b"hello")
            ok = loaded.verify(b"hello", sig, loaded.node_id)
            self.assertTrue(ok)
            print("✅ Test save/load identity OK")
        finally:
            os.unlink(path)

    def test_load_missing_returns_none(self):
        """LSPIdentity.load returns None for missing file."""
        result = LSPIdentity.load("/tmp/nonexistent_lsp_key_xyz.key")
        self.assertIsNone(result)


class TestLSPPacket(unittest.TestCase):

    def test_pack_unpack_feromon(self):
        """Test 2: Pack/unpack un paquete LSP (feromona) sin pérdida ✅"""
        import json
        identity = LSPIdentity.generate()
        payload_dict = {
            "feromon": [0.1 * i for i in range(256)],
            "fitness": 0.58,
            "step": 54500,
            "ttl": 3,
            "timestamp": 1748649600.0,
        }
        payload = json.dumps(payload_dict).encode("utf-8")
        pkt = LSPPacket.create(PacketType.FEROMON, payload, compress=True)
        raw = pkt.pack(identity)

        # Unpack
        pkt2 = LSPPacket.unpack(raw)
        self.assertEqual(pkt2.type, PacketType.FEROMON)
        self.assertTrue(pkt2.flags & Flags.SIGNED)
        recovered = pkt2.payload_json()
        self.assertAlmostEqual(recovered["fitness"], 0.58)
        self.assertEqual(recovered["step"], 54500)
        self.assertEqual(len(recovered["feromon"]), 256)
        self.assertAlmostEqual(recovered["feromon"][10], 1.0)
        print(f"✅ Test 2: pack/unpack feromon OK (raw={len(raw)}B)")

    def test_invalid_signature(self):
        """Test 3: Verificar firma inválida da False ✅"""
        import json
        identity = LSPIdentity.generate()
        payload = json.dumps({"test": "data"}).encode("utf-8")
        pkt = LSPPacket.create(PacketType.PING, payload, compress=False)
        raw = pkt.pack(identity)

        # Tamper the signature bytes (offset 12+32 = 44)
        raw_tampered = bytearray(raw)
        raw_tampered[44] ^= 0xFF  # flip a byte in signature
        pkt2 = LSPPacket.unpack(bytes(raw_tampered))
        result = pkt2.verify()
        self.assertFalse(result, "Tampered signature should fail verification")
        print("✅ Test 3: invalid signature → False OK")

    def test_header_size(self):
        """Header debe ser exactamente 108 bytes."""
        from src.network.lsp import HEADER_SIZE
        self.assertEqual(HEADER_SIZE, 108)
        print(f"✅ Header size = {HEADER_SIZE} bytes")


class TestLSPNodeUDP(unittest.TestCase):

    def test_udp_loopback_feromon(self):
        """Test 4: LSPNode local UDP loopback: enviar feromon → recibir callback ✅"""
        identity_a = LSPIdentity.generate()
        identity_b = LSPIdentity.generate()

        node_a = LSPNode(identity_a, feromon_port=17337, gossip_port=17338)
        node_b = LSPNode(identity_b, feromon_port=17339, gossip_port=17340)

        received = []
        event = threading.Event()

        @node_b.on_feromon_received
        def got_feromon(feromon, from_node_id):
            received.append((feromon, from_node_id))
            event.set()

        node_a.start()
        node_b.start()
        time.sleep(0.2)

        # Register node_b as peer of node_a
        node_a._register_peer(identity_b.node_id_hex, "127.0.0.1", 17339, 17340)

        try:
            import torch
            feromon_tensor = torch.randn(256)
            node_a.send_feromon(feromon_tensor, fitness=0.75)
        except ImportError:
            feromon_tensor = [float(i) for i in range(256)]
            node_a.send_feromon(feromon_tensor, fitness=0.75)

        ok = event.wait(timeout=3.0)
        self.assertTrue(ok, "Feromon callback should fire within 3s")
        self.assertEqual(len(received), 1)
        feromon_recv, from_id = received[0]
        self.assertEqual(from_id, identity_a.node_id_hex)
        print(f"✅ Test 4: UDP loopback feromon OK (from={from_id[:16]}...)")

        node_a.stop()
        node_b.stop()

    def test_handshake_two_nodes(self):
        """Test 5: HANDSHAKE entre dos LSPNode en el mismo proceso ✅"""
        identity_a = LSPIdentity.generate()
        identity_b = LSPIdentity.generate()

        node_a = LSPNode(identity_a, feromon_port=17341, gossip_port=17342)
        node_b = LSPNode(identity_b, feromon_port=17343, gossip_port=17344)

        peer_events = []
        event_a = threading.Event()

        @node_a.on_peer_connected
        def peer_connected(node_id, host, port):
            peer_events.append((node_id, host, port))
            event_a.set()

        node_a.start()
        node_b.start()
        time.sleep(0.2)

        # node_b connects to node_a
        node_b.connect_peer("127.0.0.1", 17342)

        ok = event_a.wait(timeout=3.0)
        self.assertTrue(ok, "Peer connected callback should fire on node_a within 3s")
        self.assertTrue(len(peer_events) >= 1)
        node_id_connected, host, _ = peer_events[0]
        self.assertEqual(node_id_connected, identity_b.node_id_hex)
        # node_b should have node_a as peer too
        time.sleep(0.2)
        peers_b = node_b.peers()
        self.assertTrue(any(p["node_id"] == identity_a.node_id_hex for p in peers_b),
                        "node_b should have node_a registered after handshake")
        print(f"✅ Test 5: HANDSHAKE between two LSPNodes OK")

        node_a.stop()
        node_b.stop()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    print("=" * 60)
    print("LSP v1 — Tests Unitarios")
    print("=" * 60)
    unittest.main(verbosity=0)
