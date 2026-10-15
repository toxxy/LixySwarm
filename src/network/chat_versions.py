"""Verify chat-version metadata without acquiring pretrained model weights."""
import hashlib
import json
from pathlib import Path

from src.release import language_family as genesis
from src.release.language_family import canonical
from src.release.language_versions import verify_chain,verify_evaluation,MAX_EVALUATION_BYTES,evidence_dependencies
from src.learning.public_math import MAX_TRACE_BYTES
from src.release.manifest import ReleaseError
from .telemetry import atomic_json
from .artifact_store import ArtifactStore,ArtifactService
from .language_version_service import fetch_history

MAX_CACHE_BYTES=20*1024**2


def _catalog_structure(value,name):
    if not isinstance(value,dict) or set(value)!={'chain','evaluations'}:raise ReleaseError('Invalid chat catalog')
    chain=value['chain'];target=verify_chain(chain)
    if chain[0]!=genesis.INITIAL_BUNDLES[name]:raise ReleaseError('Chat catalog belongs to another public family')
    reports=value['evaluations']
    expected={bundle['descriptor']['evaluation_id'] for bundle in chain[1:]}
    if not isinstance(reports,dict) or set(reports)!=expected:raise ReleaseError('Chat catalog evaluations are incomplete')
    for index,bundle in enumerate(chain[1:],1):
        identifier=bundle['descriptor']['evaluation_id'];report=reports[identifier];encoded=canonical(report)
        if (len(encoded)>MAX_EVALUATION_BYTES or len(encoded)!=bundle['descriptor']['evaluation_bytes']
                or hashlib.sha256(encoded).hexdigest()!=identifier):raise ReleaseError('Chat evaluation content identity mismatch')
    return target


def verify_catalog(value,name='qwen3-1.7b',*,store=None):
    target=_catalog_structure(value,name)
    for index,bundle in enumerate(value['chain'][1:],1):
        verify_evaluation(value['evaluations'][bundle['descriptor']['evaluation_id']],value['chain'][index-1],bundle,store=store)
    return target


class ChatVersions:
    def __init__(self,directory,name='qwen3-1.7b'):
        self.directory=Path(directory);self.name=name;self.path=self.directory/'catalog.json'
        self.store=ArtifactStore(self.directory/'artifacts',max_artifact_bytes=MAX_TRACE_BYTES,max_total_bytes=32*1024**2)
        self.cached=None;self.stamp=None

    def _validate(self,value):
        return verify_catalog(value,self.name,store=self.store)

    def _shell(self):
        """Only anchor local history while missing proof files are being repaired."""
        if not self.path.exists():return {'chain':[genesis.INITIAL_BUNDLES[self.name]],'evaluations':{}}
        if self.path.is_symlink() or self.path.stat().st_size>MAX_CACHE_BYTES:raise ReleaseError('Chat catalog exceeds its bound')
        value=json.loads(self.path.read_text());_catalog_structure(value,self.name)
        return value

    def read(self):
        if not self.path.exists():return {'chain':[genesis.INITIAL_BUNDLES[self.name]],'evaluations':{}}
        stat=self.path.stat();stamp=(stat.st_mtime_ns,stat.st_size,stat.st_ino)
        if self.path.is_symlink() or stat.st_size>MAX_CACHE_BYTES:raise ReleaseError('Chat catalog exceeds its bound')
        if stamp!=self.stamp:
            value=json.loads(self.path.read_text());self._validate(value)
            self.cached=value;self.stamp=stamp
        elif any(row['descriptor']['version']==5 for row in self.cached['chain']):
            self._validate(self.cached)
        return self.cached

    def head(self):return self.read()['chain'][-1]

    def install(self,chain,evaluations):
        value={'chain':chain,'evaluations':evaluations};target=self._validate(value);current=self._shell()['chain']
        common=min(len(current),len(chain))-1
        if current[common]['release']['release_id']!=chain[common]['release']['release_id']:
            raise ReleaseError('Chat catalog cannot fork')
        if len(chain)<len(current):return current[-1]
        if len(canonical(value))>MAX_CACHE_BYTES:raise ReleaseError('Chat catalog exceeds its bound')
        atomic_json(self.path,value);self.stamp=None
        return chain[-1]

    def synchronize(self,work,provider,*,cancel_event,preflight,receive_budget=None):
        chain=fetch_history(work,self.name,provider,cancel_event=cancel_event)
        old=self._shell();reports={};client=ArtifactService(work,self.store,namespace='language.artifact',
            register_handlers=False,receive_budget=receive_budget)
        for bundle in chain[1:]:
            identifier=bundle['descriptor']['evaluation_id'];size=bundle['descriptor']['evaluation_bytes']
            preflight(size*2+len(canonical(chain)))
            if identifier in old['evaluations']:reports[identifier]=old['evaluations'][identifier]
            else:
                path=client.fetch(identifier,peer_id=provider,expected_size=size,timeout_s=60,cancel_event=cancel_event)
                if self.store.manifest(identifier).kind!='evaluation':raise ReleaseError('Chat version has no evaluation artifact')
                reports[identifier]=json.loads(path.read_text())
            for identifier,size in evidence_dependencies(bundle):
                preflight(size*2+len(canonical(chain)))
                client.fetch(identifier,peer_id=provider,expected_size=size,timeout_s=120,cancel_event=cancel_event)
        preflight(len(canonical({'chain':chain,'evaluations':reports})))
        return self.install(chain,reports)
