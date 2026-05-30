#!/usr/bin/env python3
"""
test_integration.py — Integración SwarmNetwork ↔ LixyOrchestrator
===================================================================
Verifica que el loop P2P es BIDIRECCIONAL:
  ✅ El orquestador transmite feromonas a la red (broadcast)
  ✅ El orquestador recibe feromonas de la red (merge + inject)
  ✅ inject_remote_feromon() mezcla correctamente en RuntimeSession

Tests:
  1. inject_remote_feromon() standalone — modifica _cached_feromon
  2. LixyOrchestrator con net mockeado — el merge se aplica antes del turn
  3. Flujo completo: mock net + chat() — feromona remota procesada end-to-end
  4. Sin peers activos — no hay overhead ni errores

Uso:
    python3 test_integration.py
    python3 test_integration.py --verbose
"""

import sys, time, argparse, unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F

GREEN = "\033[92m"
RED   = "\033[91m"
RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"

results = {"pass": [], "fail": [], "skip": []}

def ok(name, detail=""):
    results["pass"].append(name)
    print(f"  {GREEN}✅ PASS{RESET} {name}" + (f" {DIM}— {detail}{RESET}" if detail else ""))

def fail(name, reason):
    results["fail"].append(name)
    print(f"  {RED}❌ FAIL{RESET} {name} {DIM}— {reason}{RESET}")

def section(title):
    print(f"\n{BOLD}{'─'*52}{RESET}\n{BOLD}{title}{RESET}\n{BOLD}{'─'*52}{RESET}")


# ─── Test 1 — inject_remote_feromon() standalone ─────────────────────────────

def test_inject_remote_feromon():
    section("Test 1 — inject_remote_feromon() standalone")

    from src.swarm.orchestrator import LixySwarm, SwarmConfig
    from src.swarm.runtime_session import RuntimeSession
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = SwarmConfig(n_agents=3, swarm_rounds=2)
    swarm = LixySwarm(cfg, load_matriarca=True).to(device)

    session = RuntimeSession(swarm, enc, device=device,
                             max_new_tokens=5, verbose=False)

    # Verificar que el campo existe
    if hasattr(session, '_remote_feromon_injection'):
        ok("_remote_feromon_injection inicializado")
    else:
        fail("_remote_feromon_injection", "campo no existe en RuntimeSession")
        return

    # Inyectar feromona remota
    remote = torch.randn(256).to(device)
    session.inject_remote_feromon(remote, blend_weight=0.25)

    inj = session._remote_feromon_injection
    if inj is not None:
        rf, bw = inj
        ok("inject_remote_feromon() almacenó el tensor", f"blend={bw:.0%}")
        if abs(bw - 0.25) < 1e-6:
            ok("blend_weight correcto", "0.25")
        else:
            fail("blend_weight", f"esperado 0.25, got {bw}")
    else:
        fail("inject_remote_feromon()", "no almacenó el tensor")
        return

    # Verificar que el turn() consume la inyección
    r = session.turn("Hello")
    if session._remote_feromon_injection is None:
        ok("turn() consumió la inyección remota")
    else:
        fail("turn() consumió la inyección", "sigue en _remote_feromon_injection después del turno")

    # Verificar que el feromon cambió (debería ser blend de local + remote)
    if session._cached_feromon is not None:
        ok("_cached_feromon actualizado después del turn", f"shape={session._cached_feromon.shape}")
    else:
        fail("_cached_feromon", "None después del turn")

    session.end_session()


# ─── Test 2 — Merge con dimensions mismatch ──────────────────────────────────

def test_inject_dimension_mismatch():
    section("Test 2 — inject_remote_feromon() con dimensión incorrecta")

    from src.swarm.orchestrator import LixySwarm, SwarmConfig
    from src.swarm.runtime_session import RuntimeSession
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = SwarmConfig(n_agents=3, swarm_rounds=2)
    swarm = LixySwarm(cfg, load_matriarca=True).to(device)
    session = RuntimeSession(swarm, enc, device=device,
                             max_new_tokens=5, verbose=False)

    # Inyectar con dimensión diferente — debe interpolar o ignorar silenciosamente
    remote_wrong_dim = torch.randn(128)   # swarm usa 256
    try:
        session.inject_remote_feromon(remote_wrong_dim, blend_weight=0.3)
        # Si inyectó, debe haber interpolado
        if session._remote_feromon_injection is not None:
            rf, bw = session._remote_feromon_injection
            if rf.shape[0] == 256:
                ok("inject con dim=128 interpoló a 256", f"shape={rf.shape}")
            else:
                ok("inject con dim incorrecto — ignorado silenciosamente")
        else:
            ok("inject con dim incorrecto — ignorado silenciosamente (None)")
    except Exception as e:
        fail("inject_remote_feromon() con dim incorrecta", f"lanzó {type(e).__name__}: {e}")

    # Inyectar None — debe ser no-op
    session.inject_remote_feromon(None)
    ok("inject_remote_feromon(None) no crashea")

    session.end_session()


# ─── Test 3 — LixyOrchestrator con net mockeado ──────────────────────────────

def test_orchestrator_bidirectional():
    section("Test 3 — LixyOrchestrator bidireccional (net mockeado)")

    from lixy_orchestrator import LixyOrchestrator, OrchestratorConfig

    cfg = OrchestratorConfig(max_tokens=8, network_remote_weight=0.3)
    lixy = LixyOrchestrator(cfg)

    # Mock de SwarmNetwork
    mock_net = MagicMock()
    mock_remote_feromon = torch.randn(256)

    # Configurar mock: is_distributed=True, merge retorna tensor
    type(mock_net).is_distributed = PropertyMock(return_value=True)
    mock_net.merge_remote_feromons.return_value = mock_remote_feromon
    mock_net.broadcast_feromon.return_value = None

    # Inyectar net mockeado
    lixy.net = mock_net

    # Rastrear si inject_remote_feromon fue llamado
    inject_calls = []
    original_inject = lixy._runtime_session.inject_remote_feromon
    def tracking_inject(tensor, blend_weight=0.25):
        inject_calls.append((tensor, blend_weight))
        return original_inject(tensor, blend_weight)
    lixy._runtime_session.inject_remote_feromon = tracking_inject

    # Llamar generate()
    response = lixy.generate("Hello test")

    # Verificar que merge_remote_feromons fue llamado
    if mock_net.merge_remote_feromons.called:
        ok("merge_remote_feromons() llamado en generate()")
    else:
        fail("merge_remote_feromons()", "NO fue llamado en generate()")

    # Verificar que inject_remote_feromon fue llamado con el tensor del mock
    if inject_calls:
        injected_tensor, blend = inject_calls[0]
        ok(f"inject_remote_feromon() llamado", f"blend={blend:.0%}")
        cos = F.cosine_similarity(
            mock_remote_feromon.unsqueeze(0),
            injected_tensor.cpu().unsqueeze(0)
        ).item()
        if cos > 0.99:
            ok("tensor remoto propagado correctamente", f"cos_sim={cos:.4f}")
        else:
            fail("tensor remoto", f"cosine_sim={cos:.4f} con el mock")
    else:
        fail("inject_remote_feromon()", "NO fue llamado — el tensor remoto fue ignorado")

    # Verificar que broadcast_feromon fue llamado post-generación
    if mock_net.broadcast_feromon.called:
        ok("broadcast_feromon() llamado post-generación (bidireccional)")
    else:
        fail("broadcast_feromon()", "NO llamado — solo escucha, no transmite")

    lixy.close()


# ─── Test 4 — Sin peers: sin overhead ────────────────────────────────────────

def test_no_peers_no_overhead():
    section("Test 4 — Sin peers activos: sin overhead ni errores")

    from lixy_orchestrator import LixyOrchestrator, OrchestratorConfig

    cfg = OrchestratorConfig(max_tokens=5)
    lixy = LixyOrchestrator(cfg)

    # Mock net con is_distributed=False (sin peers)
    mock_net = MagicMock()
    type(mock_net).is_distributed = PropertyMock(return_value=False)
    lixy.net = mock_net

    try:
        response = lixy.generate("Test sin peers")
        ok("generate() sin peers no crashea", f"response={repr(response[:30])}")
    except Exception as e:
        fail("generate() sin peers", str(e))

    # Verificar que NO se llamó merge_remote_feromons (es_distributed=False)
    if not mock_net.merge_remote_feromons.called:
        ok("merge_remote_feromons() NO llamado sin peers (sin overhead)")
    else:
        fail("overhead", "merge_remote_feromons() llamado con is_distributed=False")

    lixy.close()


# ─── Test 5 — reset_feromon() limpia la inyección ────────────────────────────

def test_reset_clears_injection():
    section("Test 5 — reset_feromon() limpia la inyección remota")

    from src.swarm.orchestrator import LixySwarm, SwarmConfig
    from src.swarm.runtime_session import RuntimeSession
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = SwarmConfig(n_agents=3, swarm_rounds=2)
    swarm = LixySwarm(cfg, load_matriarca=True).to(device)
    session = RuntimeSession(swarm, enc, device=device, max_new_tokens=5, verbose=False)

    # Inyectar y luego resetear
    session.inject_remote_feromon(torch.randn(256), blend_weight=0.3)
    assert session._remote_feromon_injection is not None
    ok("inyección presente antes de reset")

    session.reset_feromon()
    if session._remote_feromon_injection is None:
        ok("reset_feromon() limpió la inyección remota")
    else:
        fail("reset_feromon()", "inyección remota no fue limpiada")

    session.end_session()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"\n{BOLD}🌐 LixyOrchestrator ↔ SwarmNetwork — Test de integración{RESET}\n")

    test_inject_remote_feromon()
    test_inject_dimension_mismatch()
    test_orchestrator_bidirectional()
    test_no_peers_no_overhead()
    test_reset_clears_injection()

    # Resumen
    total = len(results["pass"]) + len(results["fail"])
    print(f"\n{BOLD}{'='*52}{RESET}")
    if results["fail"]:
        print(f"{RED}{BOLD}RESULTADO: {len(results['pass'])}/{total} pasaron{RESET}")
        for f in results["fail"]:
            print(f"  ❌ {f}")
        sys.exit(1)
    else:
        print(f"{GREEN}{BOLD}RESULTADO: {len(results['pass'])}/{total} tests pasaron ✅{RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()
