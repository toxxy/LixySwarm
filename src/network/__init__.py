"""Network API; importing the transport does not load calculation engines."""

from importlib import import_module
from importlib.util import find_spec

_EXPORTS = {'bootstrap': ('PeersDB', 'bootstrap_network'),
 'lsp': ('LSPIdentity',),
 'lsp_v2': ('FeromonMergeBuffer', 'FeromonV2Payload', 'LSPNodeV2', 'PacketType'),
 'lsp_v3': ('LSPNodeV3', 'ReplayGuard', 'V3Packet'),
 'node': ('NodeIdentity', 'Peer', 'PeerTable'),
 'peer_manager': ('AddressBook', 'PeerManager', 'PeerReputation', 'network_group'),
 'work_protocol': ('ResultReceipt',
                   'WorkCancelledError',
                   'WorkCoordinator',
                   'WorkExecution',
                   'WorkResult',
                   'WorkUnit'),
 'artifact_store': ('ArtifactError',
                    'ArtifactManifest',
                    'ArtifactService',
                    'ArtifactStore',
                    'digest_file'),
 'training_worker': ('TRAINING_OPERATION',
                     'TrainingWorker',
                     'TrainingWorkError',
                     'validate_gradient_artifact'),
 'gradient_aggregation': ('GradientAggregator', 'GradientCandidate'),
 'identity_work': ('load_or_mine_identity_work', 'verify_identity_work'),
 'useful_work': ('UsefulWorkCredit', 'UsefulWorkLedger', 'verify_useful_work_bundle'),
 'scheduler_history': ('SchedulerHistory',),
 'messages': ('FeromonMessage', 'GossipMessage'),
 'swarm_network': ('SwarmNetwork',)}
_MODULES = {name: module for module, names in _EXPORTS.items() for name in names}
__all__ = list(_MODULES)
if find_spec("torch") is None:
    __all__ = [name for name in __all__ if name not in {"FeromonMessage", "GossipMessage", "SwarmNetwork"}]


def __getattr__(name):
    module = _MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value
