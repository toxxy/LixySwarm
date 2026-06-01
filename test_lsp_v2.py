"""
test_lsp_v2.py — 12 tests para LSP v2

Run:
    cd /home/toxxy/Dropbox/Lixy/clawd/workspace/lixy-llm
    python -m pytest test_lsp_v2.py -v
"""

import sys
import os
import time
import struct
import threading
import socket

# Asegura que src/ esté en path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import pytest

from network.lsp_v2 import (
    FeromonV2Payload,
    FeromonMergeBuffer,
    LSPNodeV2,
    PacketType,
    DIM_FLOAT16,
    DIM_FLOAT32,
    _BYTES_PER_DIM,
)
from network.lsp_merge import merge_feromons, decay_feromon
from network.lsp import LSPIdentity, LSPNode


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_payload(dim=256, ttl=3, step=0, fitness=0.5, dim_type=DIM_FLOAT16):
    t = torch.randn(dim)
    return FeromonV2Payload(
        feromon=t,
        ttl=ttl,
        step=step,
        fitness=fitness,
        timestamp_ms=int(time.time() * 1000),
        dim_type=dim_type,
    )


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0)
    ).item()


# ─── Test 1: pack size exacta para dim=256 float16 ────────────────────────────

def test_pack_size_256d_float16():
    """FeromonV2Payload.pack() → exactamente 528 bytes para dim=256, float16"""
    p = make_payload(dim=256, dim_type=DIM_FLOAT16)
    data = p.pack()
    # Header: 1+2+1+4+4+4 = 16 bytes
    # Tensor: 256 × 2 bytes = 512 bytes
    # Total: 528 bytes
    assert len(data) == 528, f"Expected 528 bytes, got {len(data)}"


# ─── Test 2: round-trip pack/unpack ───────────────────────────────────────────

def test_pack_unpack_roundtrip():
    """unpack(pack()) → round-trip exacto (cos_sim ≥ 0.999)"""
    p = make_payload(dim=256)
    data = p.pack()
    p2 = FeromonV2Payload.unpack(data)

    sim = cos_sim(p.feromon, p2.feromon)
    assert sim >= 0.999, f"cos_sim too low: {sim}"
    assert p2.ttl == p.ttl
    assert p2.step == p.step
    assert abs(p2.fitness - p.fitness) < 1e-4
    assert p2.dim_type == p.dim_type


# ─── Test 3: apply_decay ──────────────────────────────────────────────────────

def test_apply_decay():
    """apply_decay(0.95) → vector * 0.95, TTL -= 1"""
    p = make_payload(ttl=3, fitness=0.7)
    p2 = p.apply_decay(0.95)

    assert p2.ttl == 2, f"Expected ttl=2, got {p2.ttl}"
    # Verificar que el vector es ~ p.feromon * 0.95
    expected = p.feromon * 0.95
    assert torch.allclose(p2.feromon, expected, atol=1e-5), "Decay vector mismatch"


# ─── Test 4: merge weighted avg ───────────────────────────────────────────────

def test_merge_weighted_avg():
    """merge(other, alpha=0.5) → weighted avg correcta"""
    dim = 64
    a = torch.ones(dim)
    b = torch.zeros(dim)
    pa = FeromonV2Payload(feromon=a, ttl=3, step=1, fitness=0.8, timestamp_ms=1000)
    pb = FeromonV2Payload(feromon=b, ttl=2, step=2, fitness=0.6, timestamp_ms=2000)

    merged = pa.merge(pb, alpha=0.5)
    expected_vec = a * 0.5 + b * 0.5
    assert torch.allclose(merged.feromon, expected_vec, atol=1e-5)
    assert merged.ttl == 2   # min ttl
    assert merged.step == 2  # max step
    assert abs(merged.fitness - 0.7) < 1e-5  # avg fitness


# ─── Test 5: FeromonMergeBuffer push + flush ──────────────────────────────────

def test_merge_buffer_push_flush():
    """FeromonMergeBuffer.push() + flush() → merge de 3 feromonas del mismo nodo"""
    buf = FeromonMergeBuffer()
    node_id = "aabbccdd"
    dim = 32

    payloads = [make_payload(dim=dim, fitness=0.3 + i*0.2) for i in range(3)]
    for p in payloads:
        buf.push(node_id, p)

    results = buf.flush()
    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    nid, merged = results[0]
    assert nid == node_id
    assert merged.feromon.shape == (dim,)
    # Buffer debe estar vacío después de flush
    results2 = buf.flush()
    assert len(results2) == 0


# ─── Test 6: FeromonMergeBuffer descarta feromonas viejas ─────────────────────

def test_merge_buffer_evicts_old():
    """FeromonMergeBuffer elimina feromonas viejas (> MAX_AGE_MS)"""
    buf = FeromonMergeBuffer()
    buf.MAX_AGE_MS = 50  # 50ms para que expire rápido

    p = make_payload()
    # Timestamp muy viejo (1000ms atrás)
    old_ts = int(time.time() * 1000) - 1000
    p_old = FeromonV2Payload(
        feromon=p.feromon,
        ttl=3,
        step=0,
        fitness=0.5,
        timestamp_ms=old_ts,
    )

    node_id = "oldnode"
    buf.push(node_id, p_old)
    # Esperar que expire
    time.sleep(0.1)
    results = buf.flush()
    # La feromona vieja debería ser descartada
    assert len(results) == 0 or all(
        nid != node_id for nid, _ in results
    ), "Old feromon should have been evicted"


# ─── Test 7: merge_feromons fitness_weighted vs equal_weight ──────────────────

def test_merge_feromons_strategies():
    """merge_feromons() fitness_weighted vs equal_weight"""
    dim = 16
    # Nodo A con alta fitness, vector ones
    # Nodo B con baja fitness, vector zeros
    pa = FeromonV2Payload(feromon=torch.ones(dim), ttl=3, step=1, fitness=0.9, timestamp_ms=1000)
    pb = FeromonV2Payload(feromon=torch.zeros(dim), ttl=3, step=1, fitness=0.1, timestamp_ms=1000)

    # fitness_weighted: pa debe pesar más
    merged_fw = merge_feromons([pa, pb], strategy="fitness_weighted")
    # fitness_weighted: weight_a = 0.9, weight_b = 0.1 → resultado cerca de ones
    assert merged_fw.feromon.mean().item() > 0.7, "fitness_weighted should favor pa"

    # equal_weight: promedio simple → ~0.5
    merged_ew = merge_feromons([pa, pb], strategy="equal_weight")
    assert abs(merged_ew.feromon.mean().item() - 0.5) < 0.01, "equal_weight should be 0.5"


# ─── Test 8: decay_feromon(hops=3) ────────────────────────────────────────────

def test_decay_feromon_hops3():
    """decay_feromon(hops=3) → factor 0.95^3 ≈ 0.857"""
    p = make_payload(dim=64, ttl=5)
    p_decayed = decay_feromon(p, hops=3, decay=0.95)

    factor = 0.95 ** 3
    expected = p.feromon * factor
    assert torch.allclose(p_decayed.feromon, expected, atol=1e-5)
    assert p_decayed.ttl == 5 - 3, f"Expected ttl=2, got {p_decayed.ttl}"


# ─── Test 9: LSPNodeV2 send_feromon_v2 + receive loopback ─────────────────────

def test_lspnodev2_loopback():
    """LSPNodeV2 send_feromon_v2 + receive (loopback UDP)"""
    identity_a = LSPIdentity.generate()
    identity_b = LSPIdentity.generate()

    # Usar puertos aleatorios para no colisionar
    port_a = 17337
    port_b = 17338

    node_a = LSPNodeV2(identity_a, feromon_port=port_a, gossip_port=port_a+100)
    node_b = LSPNodeV2(identity_b, feromon_port=port_b, gossip_port=port_b+100)

    received = []
    event = threading.Event()

    @node_b.on_feromon_received
    def on_recv(feromon, node_id):
        received.append((feromon, node_id))
        event.set()

    node_a.start()
    node_b.start()

    # Registrar peer manualmente
    with node_a._lock:
        node_a._peers[identity_b.node_id_hex] = {
            "node_id": identity_b.node_id_hex,
            "host": "127.0.0.1",
            "feromon_port": port_b,
            "gossip_port": port_b + 100,
            "last_seen": time.time(),
        }

    dim = 64
    test_tensor = torch.randn(dim)
    node_a.send_feromon_v2(test_tensor, fitness=0.8, step=42)

    event.wait(timeout=3.0)

    node_a.stop()
    node_b.stop()

    assert len(received) > 0, "No feromon received"
    feromon_recv, nid = received[0]
    sim = cos_sim(test_tensor, feromon_recv)
    assert sim >= 0.999, f"Loopback cos_sim too low: {sim}"


# ─── Test 10: Backward compat v1 recibe de v2 sin crash ──────────────────────

def test_backward_compat_v1_receives_v2():
    """LSPNode v1 recibe de LSPNodeV2 sin crash (ignora paquetes desconocidos)"""
    identity_a = LSPIdentity.generate()
    identity_b = LSPIdentity.generate()

    port_a = 17339
    port_b = 17340

    node_v2 = LSPNodeV2(identity_a, feromon_port=port_a, gossip_port=port_a+100)
    node_v1 = LSPNode(identity_b, feromon_port=port_b, gossip_port=port_b+100)

    crashed = []
    v1_received = []

    @node_v1.on_feromon_received
    def on_recv_v1(feromon, node_id):
        v1_received.append((feromon, node_id))

    node_v2.start()
    node_v1.start()

    # Registrar nodo v1 como peer de v2
    with node_v2._lock:
        node_v2._peers[identity_b.node_id_hex] = {
            "node_id": identity_b.node_id_hex,
            "host": "127.0.0.1",
            "feromon_port": port_b,
            "gossip_port": port_b + 100,
            "last_seen": time.time(),
        }

    # v2 envía feromon v2 → v1 no debe crashear
    try:
        node_v2.send_feromon_v2(torch.randn(32), fitness=0.5)
        time.sleep(0.5)  # dar tiempo para recv
    except Exception as e:
        crashed.append(e)

    node_v2.stop()
    node_v1.stop()

    assert len(crashed) == 0, f"Crash detected: {crashed}"
    # v1 no entiende 0x10, simplemente lo ignora — eso es OK


# ─── Test 11: packet size < 1KB para 256d float16 ─────────────────────────────

def test_packet_size_under_1kb():
    """packet size < 1KB para 256d float16 (objetivo del protocolo)"""
    p = make_payload(dim=256, dim_type=DIM_FLOAT16)
    data = p.pack()
    assert len(data) < 1024, f"Payload {len(data)} bytes >= 1KB"
    # Wire size con header LSP (108 bytes)
    total_wire = 108 + len(data)
    assert total_wire < 1024, f"Total wire {total_wire} bytes >= 1KB"


# ─── Test 12: TTL=0 → paquete se descarta sin callback ───────────────────────

def test_ttl_zero_discarded():
    """TTL=0 → paquete se descarda sin callback en LSPNodeV2"""
    identity_a = LSPIdentity.generate()
    identity_b = LSPIdentity.generate()

    port_a = 17341
    port_b = 17342

    node_a = LSPNodeV2(identity_a, feromon_port=port_a, gossip_port=port_a+100)
    node_b = LSPNodeV2(identity_b, feromon_port=port_b, gossip_port=port_b+100)

    received = []
    event = threading.Event()

    @node_b.on_feromon_received
    def on_recv(feromon, node_id):
        received.append((feromon, node_id))
        event.set()

    node_a.start()
    node_b.start()

    # Registrar peer
    with node_a._lock:
        node_a._peers[identity_b.node_id_hex] = {
            "node_id": identity_b.node_id_hex,
            "host": "127.0.0.1",
            "feromon_port": port_b,
            "gossip_port": port_b + 100,
            "last_seen": time.time(),
        }

    # Crear paquete con TTL=0 manualmente y enviarlo
    from network.lsp import LSPPacket
    payload_obj = FeromonV2Payload(
        feromon=torch.randn(64),
        ttl=0,  # TTL=0
        step=1,
        fitness=0.5,
        timestamp_ms=int(time.time() * 1000),
    )
    payload_bytes = payload_obj.pack()
    pkt = LSPPacket.create(PacketType.FEROMON_V2, payload_bytes, compress=False)
    raw = pkt.pack(identity_a)

    node_a._udp_sock.sendto(raw, ("127.0.0.1", port_b))

    event.wait(timeout=1.0)

    node_a.stop()
    node_b.stop()

    assert len(received) == 0, f"TTL=0 packet should be discarded, but got {len(received)} callbacks"


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_pack_size_256d_float16,
        test_pack_unpack_roundtrip,
        test_apply_decay,
        test_merge_weighted_avg,
        test_merge_buffer_push_flush,
        test_merge_buffer_evicts_old,
        test_merge_feromons_strategies,
        test_decay_feromon_hops3,
        test_lspnodev2_loopback,
        test_backward_compat_v1_receives_v2,
        test_packet_size_under_1kb,
        test_ttl_zero_discarded,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")

    print(f"\nLSP_V2_DONE")
    print(f"tests_passed={passed}/12")
    print(f"files_created=src/network/lsp_v2.py,src/network/lsp_merge.py,test_lsp_v2.py")
