"""Network public API. LSP v3 is primary; v2 remains for migration."""
from .bootstrap import PeersDB, bootstrap_network
from .lsp import LSPIdentity
from .lsp_v2 import FeromonMergeBuffer, FeromonV2Payload, LSPNodeV2, PacketType
from .lsp_v3 import LSPNodeV3, ReplayGuard, V3Packet
from .node import NodeIdentity, Peer, PeerTable
from .peer_manager import AddressBook, PeerManager, PeerReputation, network_group
from .work_protocol import ResultReceipt, WorkCoordinator, WorkResult, WorkUnit
from .artifact_store import (
    ArtifactError,
    ArtifactManifest,
    ArtifactService,
    ArtifactStore,
    digest_file,
)
from .training_worker import (
    TRAINING_OPERATION,
    TrainingWorker,
    TrainingWorkError,
    validate_gradient_artifact,
)
from .gradient_aggregation import GradientAggregator, GradientCandidate
from .identity_work import load_or_mine_identity_work, verify_identity_work
from .useful_work import UsefulWorkCredit, UsefulWorkLedger

__all__ = [
    "LSPIdentity",
    "LSPNodeV2",
    "LSPNodeV3",
    "V3Packet",
    "ReplayGuard",
    "PacketType",
    "FeromonV2Payload",
    "FeromonMergeBuffer",
    "PeersDB",
    "bootstrap_network",
    "NodeIdentity",
    "Peer",
    "PeerTable",
    "AddressBook",
    "PeerManager",
    "PeerReputation",
    "network_group",
    "WorkCoordinator",
    "WorkResult",
    "ResultReceipt",
    "WorkUnit",
    "ArtifactError",
    "ArtifactManifest",
    "ArtifactService",
    "ArtifactStore",
    "digest_file",
    "TRAINING_OPERATION",
    "TrainingWorker",
    "TrainingWorkError",
    "validate_gradient_artifact",
    "GradientAggregator",
    "GradientCandidate",
    "load_or_mine_identity_work",
    "verify_identity_work",
    "UsefulWorkCredit",
    "UsefulWorkLedger",
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
