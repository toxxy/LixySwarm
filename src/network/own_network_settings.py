"""Local admission and an immutable curriculum contract for dynamic own learning."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from src.learning.public_math import digest
from src.learning.swarm_policy import IDENTIFIER, network_policy_id
from .telemetry import atomic_json


@dataclass
class OwnNetworkSettings:
    version: int = 2
    enabled: bool = False
    follow_enabled: bool = False
    owner_id: str = ''
    initial_state_id: str = ''
    initial_cursor: int = 0
    documents_per_worker: int = 64
    workers_per_window: int = 8
    maximum_rounds: int = 0  # Zero means no round limit; data and quality gates still apply.
    stop_after_rejections: int = 3
    parent_program_id: str = ''

    def validate(self):
        if (type(self.version) is not int or self.version != 2
                or type(self.enabled) is not bool or type(self.follow_enabled) is not bool
                or any(not isinstance(v, str) or v and not IDENTIFIER.fullmatch(v)
                       for v in [self.owner_id, self.initial_state_id, self.parent_program_id])
                or type(self.initial_cursor) is not int or not 0 <= self.initial_cursor < 4096
                or type(self.documents_per_worker) is not int or not 16 <= self.documents_per_worker <= 256
                or type(self.workers_per_window) is not int or not 1 <= self.workers_per_window <= 8
                or type(self.maximum_rounds) is not int or not 0 <= self.maximum_rounds <= 1000000
                or type(self.stop_after_rejections) is not int or not 1 <= self.stop_after_rejections <= 32):
            raise ValueError('Invalid own-network program controls')
        if self.follow_enabled and not self.owner_id:
            raise ValueError('Own following needs a locally admitted coordinator')
        if self.enabled and (not self.follow_enabled or not self.initial_state_id):
            raise ValueError('Own coordination needs local admission and a pinned initial state')

    def contract(self):
        return digest({k: v for k, v in asdict(self).items() if k not in {'enabled', 'follow_enabled'}} |
                      {'training_policy_id': network_policy_id()})

    def save(self, directory):
        self.validate(); atomic_json(Path(directory) / 'own-network-controls.json', asdict(self))

    @classmethod
    def load(cls, directory):
        path = Path(directory) / 'own-network-controls.json'
        if not path.exists(): return cls()
        if path.stat().st_size > 4096: raise ValueError('Own-network controls exceed their bound')
        value = json.loads(path.read_text())
        if not isinstance(value, dict) or set(value) != set(asdict(cls())):
            raise ValueError('Own-network controls have unknown fields')
        result = cls(**value); result.validate(); return result
