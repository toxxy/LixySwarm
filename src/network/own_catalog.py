"""App-selected own lineage anchor. Peers supply public bytes, never executable code."""
import platform
import sys

from src.learning.public_math import ARTIFACTS, digest
from src.learning.swarm_foundation_protocol import protocol_id
from src.learning.swarm_policy import state_id

COORDINATOR = 'c45094e5fd8c64e8f6f5be3aca703edb6527711a504fcdc747a0fba15c668255'
# This is a lineage authority chosen by the app, not a list of allowed workers.
ANCHOR = {
    'version': 1, 'family': 'lixyswarm-own-causal-byte-v1', 'initialization_protocol_id': protocol_id(),
    'model_id': 'cf59e348eeb1e2d2ac95d48266bbe53b45089a88b5845e98b47b18c6673528ce',
    'optimizer_id': 'd91781743f9345344f18b63863ca57a694438eb2a50cad4de2de580fbddd4ba5',
    'step': 1033, 'model_bytes': 7337088, 'optimizer_bytes': 12595275,
    'evaluation': {'nll': 1.8433024185456734, 'byte_accuracy': 0.46371864222817427,
                   'predicted_bytes': 57123, 'documents': 128},
    'provenance': 'Own initialization and accepted updates; own_network_native_20261016.json',
}
ANCHOR['state_id'] = state_id(ANCHOR['model_id'], ANCHOR['optimizer_id'], ANCHOR['step'])


def anchor_id():
    return digest(ANCHOR)


def dependencies():
    return [(ANCHOR['model_id'], ANCHOR['model_bytes']), (ANCHOR['optimizer_id'], ANCHOR['optimizer_bytes']),
            ARTIFACTS['train'], ARTIFACTS['evaluation']]


def own_engine():
    if sys.platform != 'linux' or platform.machine() != 'x86_64': return None
    try:
        if tuple(int(v) for v in platform.libc_ver()[1].split('.')[:2]) < (2,35): return None
    except ValueError:return None
    from .engine_catalog import EngineDownload
    return EngineDownload('https://github.com/toxxy/LixySwarm/releases/download/own-engine-0.4.0a26/LixySwarmOwnCPU-linux-x86_64.zip',
        '511dc2947755e551e484ed0fc43c3aafd8c28be9eb68ff6c6771ec8e88456022',267357394,
        'a23ebdceb73c18980a352fea05d63c066fe209178d44d8ebc73afa52be431ec8',840550573,
        profile='own-cpu-programs',regular_file_bytes=840550573)
