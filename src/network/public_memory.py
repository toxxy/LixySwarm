"""Durable public-memory replication on the physical node, without private IO."""
from dataclasses import asdict,dataclass
from copy import deepcopy
import json
from pathlib import Path
import re
import threading
import time
import unicodedata

from src.contribution import ResourceRequirements
from src.release.public_memory import (PublicMemoryRegistry,SPACE_ID,MAX_DOCUMENT_BYTES,
    canonical,verify_chain,verify_document)
from .artifact_store import ArtifactService,ArtifactError
from .public_knowledge import managed_public_bytes
from .telemetry import atomic_json
from .work_protocol import ResultReceipt

CATALOG='public-memory.catalog.v1'


@dataclass
class PublicMemorySettings:
    version:int=1
    enabled:bool=True

    def save(self,directory):
        if type(self.version) is not int or self.version!=1 or type(self.enabled) is not bool:
            raise ValueError('Invalid public memory controls')
        atomic_json(Path(directory)/'public-memory-controls.json',asdict(self))

    @classmethod
    def load(cls,directory):
        path=Path(directory)/'public-memory-controls.json'
        if not path.exists():return cls()
        if path.is_symlink() or path.stat().st_size>1024:raise ValueError('Invalid bounded public memory controls')
        value=json.loads(path.read_text())
        if not isinstance(value,dict) or set(value)!={'version','enabled'} or type(value['version']) is not int or value['version']!=1 or type(value['enabled']) is not bool:
            raise ValueError('Invalid public memory controls')
        return cls(**value)


def words(text):
    folded=''.join(c for c in unicodedata.normalize('NFKD',text.casefold()) if not unicodedata.combining(c))
    return set(re.findall(r'[\w.-]+',folded))


def select_cards(document,question):
    """Selection reads only the current question, never personal history."""
    query=words(question)
    public_demo=('registro' in query and 'publico' in query) or {'public','registry'}<=query
    public_model=('catalogo' in query and 'publico' in query) or {'public','catalog'}<=query
    ranked=[]
    for row in document['cards']:
        if row['kind']=='synthetic-registry':
            if not public_demo:continue
            terms=words(row['text'])-{'registro','publico','ficticio','fictional','public','registry','en','este','ejemplo','in','this','example','la','el','es','the','is','for','del','de','of','code','clave','equipo','device','project','proyecto','nodo','etiqueta','label','correccion','correction'}
            score=len(query&terms)
        else:
            if not public_model or 'qwen3-1.7b' not in query:continue
            is_hash=bool(query&{'hash','adaptador','adapter'})
            if row['record_id']!=('language-adapter' if is_hash else 'language-sequence'):continue
            score=10
        if score:ranked.append((score,row['record_id'],row))
    return [deepcopy(row) for _,_,row in sorted(ranked,key=lambda item:(-item[0],item[1]))[:2]]


def augment_messages(messages,cards):
    """Add public evidence without removing personal turns or changing its scope.

    The original current question remains a verbatim suffix. When the common
    context budget is full, no public evidence is inserted or claimed as used.
    """
    if not cards:return deepcopy(messages),[]
    selected=[];identifiers=[];original=sum(len(row['content'].encode()) for row in messages)
    prefix='Datos de fuentes públicas verificadas / Verified public sources:\n'
    suffix='\n\nPregunta actual / Current question:\n'
    for row in cards:
        trial='\n'.join(selected+[row['text']])
        if original+len((prefix+trial+suffix).encode())>4096:continue
        selected.append(row['text']);identifiers.append(row['record_id'])
    result=deepcopy(messages)
    if selected:result[-1]['content']=prefix+'\n'.join(selected)+suffix+messages[-1]['content']
    return result,identifiers


class PublicMemoryService:
    def __init__(self,knowledge,directory):
        self.knowledge,self.directory=knowledge,Path(directory)
        self.registry=PublicMemoryRegistry(directory);self.controls=PublicMemorySettings()
        self.settings=None;self.closed=False;self.cancel=threading.Event();self.thread=None;self.next_attempt=0
        self.error=None;self.state='waiting';self.lock=threading.RLock()
        self.transfer=ArtifactService(knowledge.work,self.registry.store,namespace='public-memory.artifact',
            register_handlers=False,receive_budget=knowledge.budget)
        knowledge.work.register_handler(CATALOG,'artifact',self._guard(self._catalog),permit_cached=self._cached)
        for operation,handler in [('public-memory.artifact.describe.v1',self.transfer._describe),
            ('public-memory.artifact.read-chunk.v1',self.transfer._read_chunk),
            ('public-memory.artifact.read-chunk.v2',self.transfer._read_encoded_chunk)]:
            knowledge.work.register_handler(operation,'artifact',self._guard(handler,content=True),permit_cached=self._cached)

    def sharing(self):return self.controls.enabled and self.knowledge.allowed and not self.closed

    def receiving(self):
        return (not self.closed and self.controls.enabled and self.settings is not None and self.settings.connected
            and self.settings.policy.mode!='disabled' and self.settings.policy.max_bandwidth_mbps>0
            and self.settings.policy.max_disk_gb>0)

    def _charge(self,value,execution):
        self.knowledge.budget.wait(len(canonical(value))+2048,cancel_event=self.knowledge.cancel,deadline=execution.deadline)

    def _guard(self,handler,*,content=False):
        def guarded(payload,execution):
            execution.raise_if_cancelled()
            if not self.sharing():raise ArtifactError('Public memory sharing is disabled')
            if content:
                identifier=payload.get('artifact_id')
                matching=[row for row in self.registry.chain() if row['content_id']==identifier]
                if len(matching)!=1:raise ArtifactError('Only accepted public memory content may be served')
                self.registry.document(matching[0])
            value=handler(payload,execution);self._charge(value,execution);return value
        return guarded

    def _cached(self,execution,output):
        if not self.sharing():return False
        if execution.operation!=CATALOG:
            identifier=execution.payload.get('artifact_id')
            matching=[row for row in self.registry.chain() if row['content_id']==identifier]
            if len(matching)!=1:return False
            self.registry.document(matching[0])
        self._charge(output,execution);return self.sharing()

    def _catalog(self,payload,execution):
        if payload!={'space_id':SPACE_ID}:raise ValueError('Unknown public memory space')
        chain=self.registry.chain()
        if not chain:raise ArtifactError('No accepted public memory')
        return {'chain':chain}

    def update(self,settings):
        self.settings=settings
        try:self.controls=PublicMemorySettings.load(self.directory)
        except (OSError,ValueError):self.controls=PublicMemorySettings(enabled=False)
        if not self.receiving():self.cancel.set();self.state='disabled';return
        if self.thread is not None and self.thread.is_alive():return
        self.cancel=threading.Event()
        candidates=[peer['node_id'] for peer in self.knowledge.node.peers()
            if peer.get('capabilities',{}).get('public_memory_v1')]
        if candidates and time.monotonic()>=self.next_attempt:
            self.thread=threading.Thread(target=self._synchronize,args=(candidates,self.cancel),daemon=True,name='lixy-public-memory')
            self.thread.start()

    def _synchronize(self,providers,cancel):
        self.state='checking'
        try:
            for provider in providers[:8]:
                if cancel.is_set() or not self.receiving():return
                try:
                    self.acquire(provider,cancel)
                    self.error=None;self.state='available';return
                except (OSError,ValueError,RuntimeError) as error:self.error=type(error).__name__
            self.state='waiting'
        finally:self.next_attempt=time.monotonic()+30

    def acquire(self,provider,cancel):
        result=self.knowledge.work.submit(CATALOG,{'space_id':SPACE_ID},ResourceRequirements(kind='artifact'),
            peer_id=provider,timeout_s=30,cancel_event=cancel)
        if (result.status!='ok' or not ResultReceipt.from_dict(result.receipt).verify(result,
                expected_worker_id=provider,expected_requester_id=self.knowledge.node.identity.node_id_hex)
                or set(result.output)!={'chain'}):raise ValueError('Unverified public memory catalog response')
        chain=result.output['chain'];head=verify_chain(chain);current=self.registry.chain()
        if current:
            common=min(len(chain),len(current))-1
            if chain[common]['release_id']!=current[common]['release_id']:raise ValueError('Public memory provider forked')
            if len(chain)<len(current):chain=current
        def preflight(extra):
            if (cancel.is_set() or not self.receiving()
                    or managed_public_bytes(self.directory)+extra>self.settings.policy.max_disk_gb*1024**3):
                raise ValueError('Public memory reception exceeds current consent')
        with self.knowledge.metadata_budget(self.receiving) as budget:
            self.transfer.receive_budget=budget
            for manifest in chain:
                preflight(2*manifest['content_bytes']+len(canonical(chain)))
                path=self.transfer.fetch(manifest['content_id'],peer_id=provider,expected_size=manifest['content_bytes'],
                    timeout_s=60,cancel_event=cancel)
                value=json.loads(path.read_text());verify_document(value,manifest['sequence'])
                self.registry.store.import_file(path,kind='dataset',media_type='application/json')
            preflight(len(canonical(chain))*2)
            accepted=self.registry.install(chain)
        return accepted

    def retrieve(self,question):
        if not self.controls.enabled:return [],None
        try:
            chain=self.registry.chain()
            if not chain:return [],None
            document=self.registry.document();cards=select_cards(document,question)
            return cards,deepcopy(chain[-1]) if cards else None
        except (OSError,ValueError,TypeError):return [],None

    def snapshot(self):
        chain=[];count=0
        try:
            chain=self.registry.chain();document=self.registry.document()
            if document:count=len(document['cards'])
        except (OSError,ValueError,TypeError):pass
        return {'enabled':self.controls.enabled,'space_id':SPACE_ID,'release_id':chain[-1]['release_id'] if chain else None,
            'sequence':chain[-1]['sequence'] if chain else None,'records':count,'sharing':self.sharing() and bool(chain),
            'available':self.controls.enabled and count>0,'state':self.state,'error':self.error}

    def close(self):
        self.closed=True;self.cancel.set()
        if self.thread is not None:self.thread.join(timeout=2)
