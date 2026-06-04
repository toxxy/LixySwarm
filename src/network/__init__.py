"""
LixySwarm Network — P2P LSP v2, zero-config.
Puertos: 7337 UDP / 7338 TCP. Auto-bootstrap via peers.json + seeds + peer exchange.

SwarmNetwork requiere torch (solo nodos GPU/CPU locales).
LSPIdentity + LSPNodeV2 no requieren torch (VPS relay).
"""
from .bootstrap import PeersDB, bootstrap_network

__all__ = ["PeersDB", "bootstrap_network"]

# SwarmNetwork solo si hay torch (nodos locales con GPU/CPU)
try:
    from .swarm_network import SwarmNetwork
    __all__.append("SwarmNetwork")
except ImportError:
    pass
