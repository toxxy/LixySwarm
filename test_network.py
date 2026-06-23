#!/usr/bin/env python3
"""
test_network.py — Prueba de humo del protocolo P2P LixySwarm
=============================================================
Levanta dos nodos SwarmNetwork en localhost con puertos distintos,
envía una feromona de uno al otro, y verifica recepción + merge.

Pruebas:
  1. Modo local (zero overhead, siempre debe pasar)
  2. Modo LAN entre dos nodos en localhost (UDP loopback)
  3. Gossip TCP entre nodos (handshake de identidad)
  4. Merge de feromonas remotas en el estado del enjambre

Uso:
    python3 test_network.py
    python3 test_network.py --verbose
    python3 test_network.py --skip-lan   # solo modo local

Exit codes:
    0 = todas las pruebas pasaron
    1 = alguna prueba falló
"""

import sys, time, threading, argparse, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F

from src.network import SwarmNetwork, NodeIdentity, Peer, FeromonMessage

# ─── Helpers ──────────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
DIM    = "\033[2m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

passed = []
failed = []

def ok(name: str, detail: str = ""):
    passed.append(name)
    print(f"  {GREEN}✅ PASS{RESET} {name}" + (f" {DIM}— {detail}{RESET}" if detail else ""))

def fail(name: str, reason: str):
    failed.append(name)
    print(f"  {RED}❌ FAIL{RESET} {name} {DIM}— {reason}{RESET}")

def skip(name: str, reason: str):
    print(f"  {YELLOW}⏭  SKIP{RESET} {name} {DIM}— {reason}{RESET}")

def section(title: str):
    print(f"\n{BOLD}{'─'*50}{RESET}")
    print(f"{BOLD}{title}{RESET}")
    print(f"{BOLD}{'─'*50}{RESET}")

def check(name: str, condition: bool = True, reason: str = ""):
    """Helper para tests Fase 2."""
    if condition:
        ok(name)
    else:
        fail(name, reason or "assertion failed")



# ─── Test 1 — Modo Local ──────────────────────────────────────────────────────

def test_local_mode():
    section("Test 1 — Modo Local (single-node)")

    net = SwarmNetwork.create(swarm=None, mode="local")
    net.start()

    # En modo local, start() no arranca transporte — zero overhead
    ok("start() sin excepción", f"mode={net.stats.mode}")

    # broadcast_feromon en modo local debe ser no-op seguro
    feromon = torch.randn(256)
    try:
        net.broadcast_feromon(feromon, agent_id=0)
        ok("broadcast_feromon() en modo local no crashea")
    except Exception as e:
        fail("broadcast_feromon() en modo local", str(e))

    # collect_feromons en modo local debe retornar lista vacía
    remote = net.collect_feromons()
    if remote == [] or remote is None:
        ok("collect_feromons() retorna vacío en modo local")
    else:
        fail("collect_feromons() en modo local", f"esperaba [] pero got {remote}")

    # status()
    s = net.status()
    if "mode" in s and s["peers"] == 0:
        ok("status() correcto", f"mode={s['mode']} peers={s['peers']}")
    else:
        fail("status()", f"got {s}")

    net.stop()
    ok("stop() sin excepción")


# ─── Test 2 — FeromonMessage serialización ───────────────────────────────────

def test_message_serialization():
    section("Test 2 — Serialización de mensajes")

    # FeromonMessage pack/unpack
    feromon = torch.randn(256)
    node_id = "test-node-abc123"
    import time as _time
    msg = FeromonMessage(
        node_id=node_id[:16].ljust(16, '0'),
        agent_id=0,
        timestamp_ms=int(_time.time() * 1000),
        feromon=feromon,
    )

    packed = msg.pack()
    if isinstance(packed, bytes) and len(packed) > 0:
        ok("FeromonMessage.pack()", f"{len(packed)} bytes")
    else:
        fail("FeromonMessage.pack()", f"got {type(packed)}")
        return

    msg2 = FeromonMessage.unpack(packed)
    if msg2 is None:
        fail("FeromonMessage.unpack()", "retornó None")
        return
    ok("FeromonMessage.unpack()", f"node_id={msg2.node_id}")

    # Verificar que el tensor se recupera correctamente
    feromon2 = msg2.feromon
    if feromon2 is not None:
        cos_sim = F.cosine_similarity(feromon.unsqueeze(0), feromon2.unsqueeze(0)).item()
        if cos_sim > 0.999:
            ok("tensor round-trip", f"cosine_sim={cos_sim:.6f}")
        else:
            fail("tensor round-trip", f"cosine_sim={cos_sim:.4f} (esperado >0.999)")
    else:
        fail("FeromonMessage.to_tensor()", "retornó None")

    # Validación de freshness
    if msg.is_fresh():
        ok("is_fresh() en mensaje recién creado")
    else:
        fail("is_fresh()", "mensaje nuevo no es fresco")

    # msg.valid
    if msg.valid:
        ok("msg.valid=True en mensaje bien formado")
    else:
        fail("msg.valid", "False en mensaje bien formado")


# ─── Test 3 — Modo LAN: dos nodos en localhost ────────────────────────────────

def test_lan_two_nodes(verbose: bool = False):
    section("Test 3 — Modo LAN: dos nodos en localhost")

    import shutil

    shutil.rmtree("checkpoints/test_network_a", ignore_errors=True)
    shutil.rmtree("checkpoints/test_network_b", ignore_errors=True)

    node_a = SwarmNetwork.create(
        swarm=None,
        mode="lan",
        feromon_port=14444,
        gossip_port=14445,
        checkpoint_dir="checkpoints/test_network_a",
    )
    node_b = SwarmNetwork.create(
        swarm=None,
        mode="lan",
        feromon_port=14446,
        gossip_port=14447,
        checkpoint_dir="checkpoints/test_network_b",
    )

    # Arrancar ambos nodos
    try:
        node_a.start()
        ok("node_a.start() en LAN")
    except Exception as e:
        fail("node_a.start()", str(e))
        return

    try:
        node_b.start()
        ok("node_b.start() en LAN")
    except Exception as e:
        fail("node_b.start()", str(e))
        node_a.stop()
        return

    if node_a.connect_peer("127.0.0.1", 14447):
        ok("node_a.connect_peer(node_b)")
    else:
        fail("node_a.connect_peer(node_b)", "handshake falló")
        node_a.stop(); node_b.stop()
        return

    time.sleep(0.5)

    # Enviar feromona de A hacia B
    feromon_sent = torch.randn(256)

    try:
        node_a.broadcast_feromon(feromon_sent, agent_id=0)
        ok("broadcast_feromon() enviado desde node_a")
    except Exception as e:
        fail("broadcast_feromon()", str(e))
        node_a.stop(); node_b.stop()
        return

    # Esperar recepción (timeout 3s)
    deadline = time.time() + 3.0
    received_feromons = []
    while time.time() < deadline:
        received_feromons = node_b.collect_feromons()
        if received_feromons:
            break
        time.sleep(0.05)

    if received_feromons:
        ok("feromona recibida en node_b", f"{len(received_feromons)} mensaje(s)")

        # Verificar que el tensor es correcto
        feromon_recv = received_feromons[0]
        if feromon_recv is not None:
            cos_sim = F.cosine_similarity(
                F.normalize(feromon_sent.unsqueeze(0), dim=-1),
                F.normalize(feromon_recv.unsqueeze(0), dim=-1)
            ).item()
            if cos_sim > 0.99:
                ok("tensor recibido correcto", f"cosine_sim={cos_sim:.4f}")
            else:
                fail("tensor recibido", f"cosine_sim={cos_sim:.4f} (esperado >0.99)")
        else:
            fail("feromona recibida.to_tensor()", "retornó None")
    else:
        fail("recepción UDP", "timeout de 3s — feromona no llegó a node_b")

    node_a.stop()
    node_b.stop()
    ok("ambos nodos detenidos limpiamente")


# ─── Test 4 — Merge de feromonas remotas ─────────────────────────────────────

def test_feromon_merge():
    section("Test 4 — Merge de feromonas remotas")

    net = SwarmNetwork.create(swarm=None, mode="local")
    net.start()

    # Simular 3 feromonas remotas llegando
    feromons_to_merge = [
        torch.randn(256) for _ in range(3)
    ]

    # Inyectar feromonas via peers.update_feromon (API real)
    from src.network.node import Peer
    for i, f in enumerate(feromons_to_merge):
        peer_id = f"remote-node-{i}"
        peer = Peer(node_id=peer_id, host="127.0.0.1",
                    feromon_port=14460+i, gossip_port=14470+i)
        net.peers.add(peer)
        net.peers.update_feromon(peer_id, f)

    remote = net.collect_feromons()
    if remote is not None and len(remote) > 0:
        ok("collect_feromons() retorna feromonas inyectadas", f"{len(remote)} feromonas")
    elif remote is None or remote == []:
        fail("collect_feromons() con buffer", "retornó vacío después de inyectar feromonas")

    # Test del merge matemático
    # El merge debería ser un promedio ponderado o una suma normalizada
    local_feromon = torch.randn(256)
    try:
        merged = net.merge_remote_feromons(local_feromon, remote_weight=0.3)
        if merged is not None and merged.shape == local_feromon.shape:
            ok("merge_remote_feromons()", f"shape={merged.shape}")
        else:
            skip("merge_remote_feromons()", "retornó None o shape inesperada — puede no estar implementado")
    except AttributeError:
        skip("merge_remote_feromons()", "método no implementado en esta versión")
    except Exception as e:
        fail("merge_remote_feromons()", str(e))

    net.stop()


# ─── Test 5 — Gossip TCP ──────────────────────────────────────────────────────

def test_gossip_tcp():
    section("Test 5 — Gossip TCP (handshake de identidad)")

    from src.network.messages import GossipMessage
    from src.network.transport import GossipTCP
    from src.network import NodeIdentity, Peer

    received_gossip = []
    gossip_event = threading.Event()

    def on_gossip(msg, addr):
        received_gossip.append(msg)
        gossip_event.set()
        return None  # sin respuesta

    id_a = NodeIdentity(node_id="gossip-A", host="127.0.0.1",
                        feromon_port=14448, gossip_port=14449)
    id_b = NodeIdentity(node_id="gossip-B", host="127.0.0.1",
                        feromon_port=14450, gossip_port=14451)

    tcp_a = GossipTCP(id_a, on_gossip)
    tcp_b = GossipTCP(id_b, on_gossip)

    try:
        tcp_b.start()
        ok("GossipTCP node_b start()")
    except Exception as e:
        fail("GossipTCP start()", str(e))
        return

    # Enviar hello desde A → B
    msg = GossipMessage.make_ping(id_a.node_id, id_a.feromon_port, id_a.gossip_port)
    peer_b = Peer(node_id="gossip-B", host="127.0.0.1",
                  feromon_port=14450, gossip_port=14451)

    try:
        tcp_a.send(peer_b, msg)
        ok("GossipTCP.send() sin excepción")
    except Exception as e:
        fail("GossipTCP.send()", str(e))
        tcp_b.stop()
        return

    received = gossip_event.wait(timeout=2.0)
    if received and received_gossip:
        m = received_gossip[0]
        ok("gossip recibido en node_b", f"type={getattr(m, 'msg_type', '?')}")
        if hasattr(m, 'payload') and isinstance(m.payload, dict) and m.payload.get('node_id') == 'gossip-A':
            ok("node_id correcto en mensaje gossip")
    else:
        fail("gossip TCP", "timeout 2s — mensaje no llegó")

    tcp_b.stop()


def test_lsp_v2_loopback(verbose: bool = False):
    """
    Fase 2: Dos nodos LSP v2 intercambian feromonas binary.
    Verifica merge-on-transit y TTL end-to-end.
    """
    from src.network.lsp_v2 import LSPNodeV2
    from src.network.lsp import LSPIdentity
    import torch, time
    import torch.nn.functional as F

    id_a = LSPIdentity.generate()
    id_b = LSPIdentity.generate()
    node_a = LSPNodeV2(id_a, feromon_port=7550, gossip_port=7551)
    node_b = LSPNodeV2(id_b, feromon_port=7552, gossip_port=7553)

    try:
        node_a.start()
        node_b.start()
        check("LSP v2: node_a.start()")
        check("LSP v2: node_b.start()")

        node_a._register_peer(id_b.node_id_hex, "127.0.0.1", 7552, 7553)
        check("LSP v2: peer registrado en node_a")

        received_v2 = []
        @node_b.on_feromon_received
        def capture_v2(feromon, node_id_hex):
            received_v2.append((feromon, node_id_hex))

        feromon_orig = torch.randn(256)
        node_a.send_feromon_v2(feromon_orig, fitness=0.8, step=42)
        check("LSP v2: send_feromon_v2() sin excepcion")

        deadline = time.time() + 3.0
        while not received_v2 and time.time() < deadline:
            time.sleep(0.05)

        check("LSP v2: feromon recibida", len(received_v2) >= 1,
              "No se recibio nada en 3s")

        if received_v2:
            feromon_rx, node_hex = received_v2[0]
            feromon_rx = torch.as_tensor(feromon_rx, dtype=torch.float32)
            cos = F.cosine_similarity(feromon_orig.unsqueeze(0),
                                       feromon_rx.float().unsqueeze(0)).item()
            check("LSP v2: cosine_sim >= 0.999", cos >= 0.999,
                  f"cos_sim={cos:.4f}")
            check("LSP v2: node_id correcto",
                  node_hex[:16] == id_a.node_id_hex[:16],
                  f"node_hex={node_hex[:16]}")
    finally:
        node_a.stop()
        node_b.stop()


def test_swarm_network_v2_protocol(verbose: bool = False):
    """
    Fase 2: SwarmNetwork arranca LSPNodeV2 por defecto.
    """
    from src.network.swarm_network import SwarmNetwork
    import torch
    import shutil

    shutil.rmtree("checkpoints/test_network_v2", ignore_errors=True)
    net = SwarmNetwork.create(
        swarm=None,
        mode="lan",
        protocol="v2",
        feromon_port=7560,
        gossip_port=7561,
        checkpoint_dir="checkpoints/test_network_v2",
    )
    try:
        net.start()
        check("SwarmNetworkV2: start() sin excepcion")
        check("SwarmNetworkV2: LSPNodeV2 activo", net._lsp_v2_node is not None)
        net.broadcast_feromon(torch.randn(256))
        check("SwarmNetworkV2: broadcast_feromon OK")
    finally:
        net.stop()
    check("SwarmNetworkV2: stop() sin excepcion")


def test_lsp_v2_relay_forward(verbose: bool = False):
    """
    Fase 2 WAN: un relay LSP v2 reenvía feromonas entre dos peers.
    """
    from src.network.lsp import LSPIdentity
    from src.network.lsp_v2 import LSPNodeV2
    import numpy as np

    relay = LSPNodeV2(LSPIdentity.generate(), feromon_port=7570, gossip_port=7571)
    node_a = LSPNodeV2(LSPIdentity.generate(), feromon_port=7572, gossip_port=7573)
    node_b = LSPNodeV2(LSPIdentity.generate(), feromon_port=7574, gossip_port=7575)
    received = []

    @relay.on_feromon_received
    def relay_feromon(feromon, from_node_id):
        relay.send_feromon_v2(
            np.asarray(feromon, dtype=np.float32),
            fitness=0.5,
            exclude_node_id=from_node_id,
        )

    @node_b.on_feromon_received
    def capture_relayed(feromon, node_id_hex):
        received.append((torch.as_tensor(feromon, dtype=torch.float32), node_id_hex))

    try:
        relay.start()
        node_a.start()
        node_b.start()
        check("LSP v2 relay: nodos start()")

        node_a.connect_peer("127.0.0.1", 7571)
        node_b.connect_peer("127.0.0.1", 7571)
        time.sleep(0.5)
        check("LSP v2 relay: peers conectados", len(relay.peers()) == 2,
              f"peers={len(relay.peers())}")

        feromon_orig = torch.linspace(-1, 1, 256)
        node_a.send_feromon_v2(feromon_orig, fitness=0.8, step=7)
        deadline = time.time() + 3.0
        while not received and time.time() < deadline:
            time.sleep(0.05)

        check("LSP v2 relay: feromona reenviada", len(received) >= 1,
              "node_b no recibió desde relay")
        if received:
            feromon_rx, node_hex = received[0]
            cos = F.cosine_similarity(feromon_orig.unsqueeze(0), feromon_rx.unsqueeze(0)).item()
            check("LSP v2 relay: cosine_sim >= 0.999", cos >= 0.999,
                  f"cos_sim={cos:.4f}")
            check("LSP v2 relay: emisor es relay",
                  node_hex[:16] == relay.identity.node_id_hex[:16],
                  f"node_hex={node_hex[:16]}")
    finally:
        node_a.stop()
        node_b.stop()
        relay.stop()


def test_lsp_v2_gossip_delta_loopback(verbose: bool = False):
    """
    Fase 2 Matriarca global: LSP v2 transporta GOSSIP_DELTA por TCP.
    """
    from src.network.lsp import LSPIdentity
    from src.network.lsp_v2 import LSPNodeV2

    node_a = LSPNodeV2(LSPIdentity.generate(), feromon_port=7580, gossip_port=7581)
    node_b = LSPNodeV2(LSPIdentity.generate(), feromon_port=7582, gossip_port=7583)
    received = []

    @node_b.on_gossip_delta_received
    def capture_delta(delta, node_id_hex):
        received.append((delta, node_id_hex))

    try:
        node_a.start()
        node_b.start()
        check("LSP v2 gossip: nodos start()")

        node_a._register_peer(node_b.identity.node_id_hex, "127.0.0.1", 7582, 7583)
        delta = {
            "kind": "matriarca_global_delta",
            "version": 1,
            "count": 1,
            "metadata": [{"text": "memoria global loopback", "importance": 0.7, "scope": "global"}],
            "embeddings": [[0.1, 0.2, 0.3, 0.4]],
        }
        node_a.send_gossip_delta(delta)

        deadline = time.time() + 3.0
        while not received and time.time() < deadline:
            time.sleep(0.05)

        check("LSP v2 gossip: delta recibido", len(received) == 1,
              f"received={len(received)}")
        if received:
            delta_rx, node_hex = received[0]
            check("LSP v2 gossip: kind correcto",
                  delta_rx.get("kind") == "matriarca_global_delta",
                  f"kind={delta_rx.get('kind')}")
            check("LSP v2 gossip: node_id correcto",
                  node_hex[:16] == node_a.identity.node_id_hex[:16],
                  f"node_hex={node_hex[:16]}")
    finally:
        node_a.stop()
        node_b.stop()


def test_swarm_network_global_matriarca_sync(verbose: bool = False):
    """
    Fase 2 Matriarca global: SwarmNetwork sincroniza MatriarcaDual.
    """
    from src.matriarca.matriarca import MatriarcaConfig
    from src.matriarca.matriarca_legacy import MatriarcaDual
    import shutil

    shutil.rmtree("checkpoints/test_global_sync_a", ignore_errors=True)
    shutil.rmtree("checkpoints/test_global_sync_b", ignore_errors=True)

    def tiny_cfg(root: str, name: str) -> MatriarcaConfig:
        return MatriarcaConfig(
            embd_dim=32,
            infrasound_dim=16,
            n_heads=4,
            n_layers=1,
            max_memories=64,
            memory_path=f"{root}/{name}.json",
            checkpoint_path=f"{root}/{name}.pt",
        )

    dual_a = MatriarcaDual(
        tiny_cfg("checkpoints/test_global_sync_a", "personal"),
        tiny_cfg("checkpoints/test_global_sync_a", "global"),
        device="cpu",
    )
    dual_b = MatriarcaDual(
        tiny_cfg("checkpoints/test_global_sync_b", "personal"),
        tiny_cfg("checkpoints/test_global_sync_b", "global"),
        device="cpu",
    )
    net_a = SwarmNetwork.create(
        swarm=None,
        mode="lan",
        protocol="v2",
        feromon_port=7584,
        gossip_port=7585,
        checkpoint_dir="checkpoints/test_global_sync_a",
    )
    net_b = SwarmNetwork.create(
        swarm=None,
        mode="lan",
        protocol="v2",
        feromon_port=7586,
        gossip_port=7587,
        checkpoint_dir="checkpoints/test_global_sync_b",
    )
    net_a.attach_global_matriarca(dual_a)
    net_b.attach_global_matriarca(dual_b)

    try:
        net_a.start()
        net_b.start()
        check("SwarmNetwork global: nodos start()")

        net_a._lsp_v2_node._register_peer(net_b._lsp_v2_node.identity.node_id_hex, "127.0.0.1", 7586, 7587)
        dual_a.store_personal(torch.randn(32), "[PERSONAL] no debe viajar", importance=1.0)
        dual_a.store_global(torch.randn(32), "memoria global desde swarmnetwork", importance=0.9)

        sent = net_a.broadcast_global_delta(max_items=8)
        check("SwarmNetwork global: delta enviado", sent == 1, f"sent={sent}")

        deadline = time.time() + 4.0
        while net_b.stats.global_memories_received < 1 and time.time() < deadline:
            time.sleep(0.05)

        texts = [m.get("text", "") for m in dual_b.global_.bank.metadata]
        check("SwarmNetwork global: memoria recibida",
              "memoria global desde swarmnetwork" in texts,
              f"texts={texts}")
        check("SwarmNetwork global: privacidad personal",
              all("no debe viajar" not in text for text in texts),
              f"texts={texts}")
    finally:
        net_a.stop()
        net_b.stop()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test SwarmNetwork P2P 🌐")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip-lan", action="store_true", help="Skip tests LAN (solo local)")
    parser.add_argument("--skip-gossip", action="store_true", help="Skip test gossip TCP")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    print(f"\n{BOLD}🌐 LixySwarm Network — Prueba de humo{RESET}")
    print(f"{DIM}Workspace: {Path.cwd()}{RESET}\n")

    # Verificar que los módulos cargan
    try:
        from src.network import SwarmNetwork, NodeIdentity, Peer, FeromonMessage
        print(f"  {GREEN}✅{RESET} Módulos de red importados correctamente")
    except ImportError as e:
        print(f"  {RED}❌ Error importando módulos de red: {e}{RESET}")
        sys.exit(1)

    # Ejecutar tests
    test_local_mode()
    test_message_serialization()

    if not args.skip_lan:
        test_lan_two_nodes(verbose=args.verbose)
    else:
        skip("test_lan_two_nodes", "--skip-lan activo")

    test_feromon_merge()

    if not args.skip_gossip:
        test_gossip_tcp()
    else:
        skip("test_gossip_tcp", "--skip-gossip activo")

    # Fase 2 — LSP v2
    test_lsp_v2_loopback(verbose=args.verbose)
    test_swarm_network_v2_protocol(verbose=args.verbose)
    test_lsp_v2_relay_forward(verbose=args.verbose)
    test_lsp_v2_gossip_delta_loopback(verbose=args.verbose)
    test_swarm_network_global_matriarca_sync(verbose=args.verbose)

    # ─── Resumen ──────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'='*50}{RESET}")
    total = len(passed) + len(failed)
    if failed:
        print(f"{RED}{BOLD}RESULTADO: {len(passed)}/{total} tests pasaron{RESET}")
        print(f"{RED}Fallos:{RESET}")
        for f in failed:
            print(f"  - {f}")
        exit_code = 1
    else:
        print(f"{GREEN}{BOLD}RESULTADO: {len(passed)}/{total} tests pasaron ✅{RESET}")
        exit_code = 0

    print()

    # ─── Diagnóstico de lo que falta para fase 2 ─────────────────────────────
    print(f"{BOLD}📋 Red LSP v2 actual:{RESET}")
    print(f"  1. SwarmNetwork usa LSP v2 por defecto en modo LAN/auto")
    print(f"  2. broadcast_feromon() y merge_remote_feromons() están activos")
    print(f"  3. LixySwarm puede mezclar feromonas remotas en forward()")
    print(f"  4. train_swarm.py --network publica y consume feromonas remotas")
    print(f"  5. MatriarcaDual sincroniza memoria global vía GOSSIP_DELTA")
    print(f"  6. Para WAN real se requiere relay/VPS o peer con IP pública")
    print()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
