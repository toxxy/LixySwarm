"""
LixySwarm Network — Protocolo P2P para enjambre distribuido.
Fase 1: single-node + descubrimiento LAN via mDNS.
"""
from .node import NodeIdentity, Peer, PeerTable
from .messages import FeromonMessage, GossipMessage
from .transport import FeromonUDP, GossipTCP
from .swarm_network import SwarmNetwork

__all__ = [
    "NodeIdentity", "Peer", "PeerTable",
    "FeromonMessage", "GossipMessage",
    "FeromonUDP", "GossipTCP",
    "SwarmNetwork",
]
