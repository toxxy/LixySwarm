"""
LixySwarm Network — P2P LSP v2, zero-config.
Puertos: 7337 UDP / 7338 TCP. Auto-bootstrap via peers.json + seeds + peer exchange.
"""
from .swarm_network import SwarmNetwork
from .bootstrap import PeersDB, bootstrap_network

__all__ = ["SwarmNetwork", "PeersDB", "bootstrap_network"]
