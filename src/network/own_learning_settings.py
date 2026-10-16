"""Local opt-in for bounded continuous learning by the two known operators."""
from dataclasses import asdict,dataclass
import json
from pathlib import Path

from src.learning.swarm_distributed import WORKERS,IDENTIFIER,policy_id
from src.learning.public_math import digest
from .telemetry import atomic_json


@dataclass
class OwnLearningSettings:
    version:int=1
    enabled:bool=False
    follow_enabled:bool=False
    owner_id:str=WORKERS[0]
    initial_state_id:str=''
    initial_cursor:int=256
    documents_per_worker:int=128
    maximum_rounds:int=10
    stop_after_rejections:int=3

    def validate(self):
        if (type(self.version) is not int or self.version!=1
                or type(self.enabled) is not bool or type(self.follow_enabled) is not bool
                or self.owner_id not in WORKERS
                or not isinstance(self.initial_state_id,str)
                or self.initial_state_id and not IDENTIFIER.fullmatch(self.initial_state_id)
                or type(self.initial_cursor) is not int or not 0<=self.initial_cursor<=4096
                or type(self.documents_per_worker) is not int or not 1<=self.documents_per_worker<=256
                or type(self.maximum_rounds) is not int or not 1<=self.maximum_rounds<=10
                or type(self.stop_after_rejections) is not int or not 1<=self.stop_after_rejections<=3):
            raise ValueError('Invalid bounded own-learning controls')
        if self.enabled and (not self.follow_enabled or not self.initial_state_id):
            raise ValueError('Own coordination requires a pinned initial state and local adoption permission')

    def contract(self):
        return digest({k:v for k,v in asdict(self).items() if k not in {'enabled','follow_enabled'}}|
            {'training_policy_id':policy_id()})

    def save(self,directory):
        self.validate();atomic_json(Path(directory)/'own-learning-controls.json',asdict(self))

    @classmethod
    def load(cls,directory):
        path=Path(directory)/'own-learning-controls.json'
        if not path.exists():return cls()
        if path.stat().st_size>4096:raise ValueError('Own learning controls exceed their bound')
        value=json.loads(path.read_text())
        if not isinstance(value,dict) or set(value)!=set(asdict(cls())):
            raise ValueError('Own learning controls have unknown fields')
        result=cls(**value);result.validate();return result
