"""Durable operator checkpoints for local, offline cold-trial recovery.

Callers hold the original/trial NodeStateLocks while capturing or restoring.
Only temporary compute controls may be changed by this checkpoint;
private banks, identity keys and learning ledgers are never copied from trials.
"""
import base64
import hashlib
import json
import os
import re
from pathlib import Path
import tempfile

from src.artifact_io import atomic_json
from src.network.artifact_store import digest_file
from .cold_recovery import COUNTERS,read_counter,newest_status,restore_counters
from src.network.telemetry import verify_status

CONTROLS=('controls.json','public-knowledge.json','language-engine.json','language-training.json',
    'language-follow-controls.json','language-learning-controls.json','learning-controls.json',
    'language-learning/ledger.json','learning/ledger.json','public-memory-controls.json')
PRIVATE=('identity.key','model-publisher.key','personal.enc')
OWNED_MUTABLE={'language-engine.json','language-training.json','controls.json'}


def fingerprint(path):
    path=Path(path)
    if not path.exists():return None
    if path.is_symlink() or not path.is_file():raise ValueError('Checkpoint only accepts regular local files')
    return digest_file(path)[0]


def atomic_bytes(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_symlink():raise ValueError('Refusing to replace a linked operator control')
    fd,name=tempfile.mkstemp(prefix='.cold-control-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(name,path)
    finally:Path(name).unlink(missing_ok=True)


class ColdCheckpoint:
    def __init__(self,path):
        self.path=Path(path)
        if self.path.is_symlink() or self.path.stat().st_size>4*1024**2:raise ValueError('Invalid recovery checkpoint')
        self.value=json.loads(self.path.read_text())
        if self.value.get('schema')!='lixyswarm-cold-checkpoint-v1':raise ValueError('Unknown recovery checkpoint')
        if (set(self.value.get('controls',{}))!=set(CONTROLS)
                or set(self.value.get('private_fingerprints',{}))!=set(PRIVATE)
                or set(self.value.get('counter_bases',{}))!=set(COUNTERS)
                or set(self.value.get('owned_hashes',{}))-OWNED_MUTABLE
                or not re.fullmatch('[0-9a-f]{64}',str(self.value.get('identity')))):
            raise ValueError('Recovery checkpoint exceeds its ownership scope')
        self.original=Path(self.value['original']);self.trial=Path(self.value['trial'])
        if (self.original.resolve()==self.trial.resolve() or self.original.resolve().is_relative_to(self.trial.resolve())
                or self.trial.resolve().is_relative_to(self.original.resolve())):
            raise ValueError('Original and trial state must be separate')

    @classmethod
    def capture(cls,original,trial,path):
        original=Path(original).resolve();trial=Path(trial).resolve();path=Path(path)
        if path.exists():raise ValueError('Recovery checkpoint already exists; resume it instead')
        if original==trial or original.is_relative_to(trial) or trial.is_relative_to(original):
            raise ValueError('Original and trial state must be separate')
        if trial.exists() and any(p.name!='node.lock' or p.is_symlink() or not p.is_file() for p in trial.iterdir()):
            raise ValueError('The initial trial directory must be empty apart from its instance lock')
        status=verify_status(json.loads((original/'status.json').read_text()))
        controls={};size=0
        for name in CONTROLS:
            p=original/name
            digest=fingerprint(p)
            raw=p.read_bytes() if digest is not None else None
            size+=len(raw or b'')
            if size>2*1024**2:raise ValueError('Operator checkpoint exceeds its bound')
            controls[name]={'sha256':digest,'data':base64.b64encode(raw).decode() if raw is not None else None}
        value={'schema':'lixyswarm-cold-checkpoint-v1','original':str(original),'trial':str(trial),
            'identity':status['node_id'],'phase':'captured','controls':controls,
            'private_fingerprints':{name:fingerprint(original/name) for name in PRIVATE},
            'counter_bases':{name:read_counter(original,name) for name in COUNTERS},'owned_hashes':{}}
        atomic_json(path,value);return cls(path)

    def record(self,phase,**values):
        self.value.update(values,phase=phase);atomic_json(self.path,self.value)

    def set_owned_control(self,name,data):
        if name not in OWNED_MUTABLE or not isinstance(data,bytes) or len(data)>64*1024:
            raise ValueError('Only temporary local compute controls may be changed')
        if name=='controls.json':
            # Recovery may stop compute, never enlarge or otherwise change consent.
            expected=json.loads(base64.b64decode(self.value['controls'][name]['data']))
            expected['policy'].update(cpu_percent=0,gpu_percent=0,max_cpu_jobs=0,max_gpu_jobs=0)
            if json.loads(data)!=expected:raise ValueError('Recovery can only suspend compute with all other permissions unchanged')
        before=self.value['controls'][name]['sha256'];current=fingerprint(self.original/name)
        accepted=self.value['owned_hashes'].get(name,[])
        if current not in [before,*accepted]:raise ValueError('Operator changed this control during the trial')
        digest=hashlib.sha256(data).hexdigest()
        if digest not in accepted:accepted.append(digest)
        self.value['owned_hashes'][name]=accepted
        # Write ahead: restore is safe if power fails before or after replacement.
        atomic_json(self.path,self.value);atomic_bytes(self.original/name,data)

    def suspend_compute(self):
        value=json.loads(base64.b64decode(self.value['controls']['controls.json']['data']))
        value['policy'].update(cpu_percent=0,gpu_percent=0,max_cpu_jobs=0,max_gpu_jobs=0)
        self.set_owned_control('controls.json',json.dumps(value,sort_keys=True,allow_nan=False).encode())

    def restore_controls(self):
        restored=[];external=[]
        for name,before in self.value['controls'].items():
            if name not in CONTROLS:raise ValueError('Checkpoint contains an unknown operator file')
            current=fingerprint(self.original/name)
            if current==before['sha256']:continue
            if name not in OWNED_MUTABLE:
                external.append(name);continue
            if current not in self.value['owned_hashes'].get(name,[]):
                external.append(name);continue
            if before['data'] is None:(self.original/name).unlink(missing_ok=True)
            else:
                raw=base64.b64decode(before['data'],validate=True)
                if hashlib.sha256(raw).hexdigest()!=before['sha256']:raise ValueError('Operator backup checksum changed')
                atomic_bytes(self.original/name,raw)
            restored.append(name)
        return {'restored':restored,'external_changes_preserved':external}

    def recover_metadata(self):
        """Idempotent across a power failure even after counters were written."""
        sequence=newest_status(self.original,self.trial,self.value['identity'])
        counters=restore_counters(self.original,self.trial,self.value['counter_bases'])
        controls=self.restore_controls()
        private={name:fingerprint(self.original/name)==digest for name,digest in self.value['private_fingerprints'].items()}
        report={'sequence':sequence,'counters':counters,'operator_controls':controls,'private_files_unchanged':private}
        self.record('metadata_recovered',metadata_report=report)
        return report
