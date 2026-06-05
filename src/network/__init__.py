"""Network public API.

LSP v2 es el camino principal. Los mensajes/transporte v1 quedan disponibles
como compatibilidad legacy y se importan solo si sus dependencias existen.
"""
from .bootstrap import PeersDB, bootstrap_network
from .lsp import LSPIdentity
from .lsp_v2 import FeromonMergeBuffer, FeromonV2Payload, LSPNodeV2, PacketType
from .node import NodeIdentity, Peer, PeerTable

__all__ = [
    "LSPIdentity",
    "LSPNodeV2",
    "PacketType",
    "FeromonV2Payload",
    "FeromonMergeBuffer",
    "PeersDB",
    "bootstrap_network",
    "NodeIdentity",
    "Peer",
    "PeerTable",
]

# Compatibilidad v1: requiere torch por FeromonMessage.
try:
    from .messages import FeromonMessage, GossipMessage
    __all__.extend(["FeromonMessage", "GossipMessage"])
except ImportError:
    pass

# SwarmNetwork solo si hay torch (nodos locales con GPU/CPU).
try:
    from .swarm_network import SwarmNetwork
    __all__.append("SwarmNetwork")
except ImportError:
    pass
