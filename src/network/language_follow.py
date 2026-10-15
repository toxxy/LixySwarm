"""Follow verified public language versions on the existing node identity."""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict,dataclass,field
from pathlib import Path

from src.learning.public_bases import BASES
from src.release.language_versions import LanguageVersionRegistry,dependencies
from .client_state import ClientSettings,NodeStateLock
from .language_activation import activate_version
from .language_compute import LanguageSettings
from .language_replication import replicate_version
from .language_version_service import fetch_history
from .public_knowledge import managed_public_bytes
from .telemetry import atomic_json


@dataclass
class LanguageFollowSettings:
    version:int=1
    enabled:bool=False
    base_name:str='qwen3-1.7b'
    providers:list[str]=field(default_factory=list)
    select_updates:bool=False
    interval_s:int=30

    def validate(self):
        if (type(self.version) is not int or self.version!=1 or type(self.enabled) is not bool
                or self.base_name not in BASES or type(self.select_updates) is not bool
                or type(self.interval_s) is not int or not 30<=self.interval_s<=3600
                or not isinstance(self.providers,list) or len(self.providers)>8
                or any(not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value) for value in self.providers)
                or len(set(self.providers))!=len(self.providers) or self.enabled and not self.providers):
            raise ValueError('Invalid local language-follow controls')

    def save(self,directory):
        self.validate();atomic_json(Path(directory)/'language-follow-controls.json',asdict(self))

    @classmethod
    def load(cls,directory):
        path=Path(directory)/'language-follow-controls.json'
        if not path.exists():return cls()
        if path.stat().st_size>8192:raise ValueError('Language-follow controls exceed their bound')
        value=json.loads(path.read_text())
        if not isinstance(value,dict) or set(value)!=set(asdict(cls())):raise ValueError('Invalid language-follow schema')
        result=cls(**value);result.validate();return result


class LanguageVersionFollower:
    def __init__(self,knowledge,families,directory):
        self.knowledge,self.families,self.directory=knowledge,families,Path(directory)
        self.path=self.directory/'language-follow/status.json'
        self.settings=LanguageFollowSettings();self.cancel=threading.Event();self.changed=threading.Event()
        self.lock=threading.RLock();self.thread=None;self.closed=False;self.next_attempt=0
        self.verified=None;self.state={'state':'idle','release_id':None,'sequence':None,'error':None}
        if self.path.exists():
            if self.path.stat().st_size>64*1024:raise ValueError('Language-follow status exceeds its bound')
            self.state=json.loads(self.path.read_text())

    def snapshot(self):
        with self.lock:return {**self.state,'enabled':self.settings.enabled}

    def _save(self,**values):
        with self.lock:
            self.state.update(values,updated_at=time.time());atomic_json(self.path,self.state);self.changed.set()

    def update(self):
        try:self.settings=LanguageFollowSettings.load(self.directory)
        except (OSError,ValueError,TypeError):self.settings=LanguageFollowSettings()
        if not self.settings.enabled or not self.knowledge.allowed or self.closed:
            self.cancel.set()
            if (self.thread is None or not self.thread.is_alive()) and self.state.get('state') in {'checking','acquiring'}:
                self._save(state='paused')
            return
        if self.thread is not None and self.thread.is_alive():return
        if time.monotonic()<self.next_attempt:return
        self.cancel=threading.Event()
        self.thread=threading.Thread(target=self._run,args=(self.cancel,),daemon=True,name='lixy-language-follow')
        self.thread.start()

    def _fingerprint(self,chain):
        """Rehash after any object/manifest change, and always after a restart."""
        rows=[];seen=set()
        for bundle in chain:
            for identifier,size in dependencies(bundle):
                if identifier in seen:continue
                seen.add(identifier);manifest=self.families.store.manifest(identifier)
                stat=self.families.store.object_path(identifier).stat()
                if manifest.size!=size:raise ValueError('Language-follow object size changed')
                rows.append((identifier,manifest.to_dict(),stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns))
        return chain[-1]['release']['release_id'],rows

    def _run(self,cancel):
        configuration=asdict(self.settings);controls=None
        def allowed():
            try:
                return (not self.closed and not cancel.is_set() and self.knowledge.allowed
                    and asdict(LanguageFollowSettings.load(self.directory))==configuration
                    and asdict(ClientSettings.load(self.directory))==controls
                    and controls['connected'] and not controls['paused'] and controls['policy']['max_bandwidth_mbps']>0
                    and controls['policy']['max_disk_gb']>0)
            except (OSError,ValueError,TypeError):return False
        def check():
            if not allowed():cancel.set();raise RuntimeError('Language following paused or local controls changed')
        try:
            controls=asdict(ClientSettings.load(self.directory));selected=asdict(LanguageSettings.load(self.directory))
            check()
            with NodeStateLock(self.directory/'preparation-lock'):
                self._save(state='checking',error=None)
                registry=LanguageVersionRegistry(self.families.root,self.settings.base_name)
                head=registry.head();chain=registry.history(head['release']['release_id']);providers=[];newer=False
                routes=[];metadata_responses=0
                for provider in self.settings.providers:
                    check()
                    if provider==self.knowledge.node.identity.node_id_hex:continue
                    if provider not in self.knowledge.node.connected_node_ids:self.knowledge.node.connect_node(provider,timeout=12)
                    if provider not in self.knowledge.node.connected_node_ids:continue
                    providers.append(provider)
                    peer=next((row for row in self.knowledge.node.peers() if row['node_id']==provider),{})
                    routes.append({key:peer.get(key) for key in ['node_id','transport','encrypted','host','relay_host']})
                    try:remote=fetch_history(self.knowledge.work,self.settings.base_name,provider,cancel_event=cancel,timeout_s=30)
                    except (OSError,ValueError,RuntimeError):continue
                    metadata_responses+=1
                    common=min(len(chain),len(remote))-1
                    if chain[common]['release']['release_id']!=remote[common]['release']['release_id']:
                        raise ValueError('Language follower rejected a conflicting signed history')
                    newer|=len(remote)>len(chain)
                if not providers:raise RuntimeError('No configured language provider is reachable')
                try:unchanged=self.verified is not None and self.verified==self._fingerprint(chain)
                except (OSError,ValueError):unchanged=False
                acquired={}
                if newer or not unchanged:
                    check();self._save(state='acquiring')
                    def preflight(history):
                        check()
                        objects={identifier:size for row in history for identifier,size in dependencies(row)}
                        missing=sum(size for identifier,size in objects.items() if not self.families.store.has(identifier))
                        if managed_public_bytes(self.directory)+2*missing>controls['policy']['max_disk_gb']*1024**3:
                            raise ValueError('Language follower exceeds local storage consent')
                    last_progress=0
                    def progress(identifier,done,provider):
                        nonlocal last_progress
                        check()
                        now=time.monotonic()
                        if now-last_progress>=1:
                            if managed_public_bytes(self.directory)>controls['policy']['max_disk_gb']*1024**3:
                                cancel.set();raise ValueError('Language follower storage consent exceeded')
                            self._save(artifact_id=identifier,artifact_bytes=done,provider=provider);last_progress=now
                    result=replicate_version(self.knowledge.work,self.families.store,self.families.root,
                        self.settings.base_name,providers,cancel_event=cancel,receive_budget=self.knowledge.budget,
                        allowed=allowed,preflight=preflight,progress=progress)
                    acquired=result['acquired_objects'];head=registry.head();chain=registry.history(head['release']['release_id'])
                    self.verified=self._fingerprint(chain)
                check();selection='unchanged'
                if self.settings.select_updates:
                    engine=LanguageSettings.load(self.directory)
                    if asdict(engine)!=selected:selection='local_expert_changed'
                    elif not engine.enabled or engine.base_name!=self.settings.base_name:selection='expert_not_enabled'
                    elif engine.adapter_id!=head['release']['model_artifact_id']:
                        activate_version(self.directory,self.settings.base_name,head['release']['release_id'],expected_adapter_id=engine.adapter_id)
                        selection='selected'
                if newer or not unchanged:
                    record=self.directory/'language-follow/versions'/f"{head['release']['release_id']}.json"
                    if acquired or not record.exists():
                        atomic_json(record,{**result,'active_expert_changed':selection=='selected',
                            'selection':selection,'routes':routes,'verified_at':time.time(),
                            'physical_receiver':self.knowledge.node.identity.node_id_hex})
                self._save(state='current' if metadata_responses else 'waiting',release_id=head['release']['release_id'],sequence=head['release']['sequence'],
                    model_id=head['release']['model_artifact_id'],selection=selection,acquired_objects=acquired,
                    routes=routes,metadata_responses=metadata_responses,error=None if metadata_responses else 'Metadata unavailable; local version preserved',
                    last_verified_at=time.time() if newer or not unchanged else self.state.get('last_verified_at'))
        except (OSError,ValueError,RuntimeError,KeyError,TypeError) as error:
            self._save(state='waiting' if not cancel.is_set() else 'paused',error=str(error)[:200])
        finally:self.next_attempt=time.monotonic()+self.settings.interval_s

    def close(self):
        self.closed=True;self.cancel.set()
        if self.thread is not None:self.thread.join(timeout=10)
