"""Owner-requested, resumable repair of an already verified public version.

No tensor process runs and no expert is selected. The signed local history pins
all bytes; peers only supply immutable objects through the physical LSP node.
"""
from dataclasses import asdict
import json
from pathlib import Path
import re
import shutil
import threading
import time

from src.release.language_versions import LanguageVersionRegistry,dependencies
from src.learning.public_bases import BASES
from .client_state import ClientSettings,NodeStateLock
from .language_preparation import INITIAL_PROVIDERS,BUDGET_MARGIN_BYTES
from .language_replication import replicate_version
from .public_knowledge import managed_public_bytes,BandwidthBudget
from .telemetry import atomic_json


def repair_budget(store,chain):
    missing=0
    for identifier,size in {identifier:size for bundle in chain for identifier,size in dependencies(bundle)}.items():
        if store.has(identifier) and store.manifest(identifier).size==size:continue
        part=store.incoming/(identifier+'.part')
        if part.is_symlink():raise ValueError('La descarga pendiente no es un archivo local válido')
        present=min(size,part.stat().st_size) if part.is_file() else 0
        missing+=size-present
    return missing+BUDGET_MARGIN_BYTES


class LanguageCacheRepair:
    def __init__(self,node,knowledge,families,directory):
        self.node,self.knowledge,self.families=node,knowledge,families
        self.directory=Path(directory);self.intent=self.directory/'language-cache-repair-intent.json'
        self.path=self.directory/'language-cache-repair.json'
        self.lock=threading.RLock();self.thread=None;self.cancel=threading.Event();self.closed=False;self.next_attempt=0
        self.state={'state':'idle','message':'','done':0,'total':0}

    def snapshot(self):
        with self.lock:return dict(self.state)

    def _status(self,state,**values):
        with self.lock:
            self.state={**self.state,**values,'state':state,'updated_at':time.time()}
            atomic_json(self.path,self.state)

    def _eligible(self):
        settings=ClientSettings.load(self.directory)
        if (not settings.connected or settings.paused or not self.knowledge.allowed
                or settings.policy.max_disk_gb<=0 or settings.policy.max_bandwidth_mbps<=0):
            raise ValueError('Conecta y autoriza conservar conocimiento público para reparar sus archivos')
        return settings

    def _read_intent(self):
        if self.intent.stat().st_size>1024:raise ValueError('La solicitud local de reparación excede su límite')
        return json.loads(self.intent.read_text())

    def start(self,name):
        if self.closed:raise RuntimeError('El nodo se está cerrando')
        if not isinstance(name,str) or name not in BASES:raise ValueError('Familia pública desconocida')
        self._eligible()
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                if self.state.get('base_name')!=name:raise RuntimeError('Ya se está reparando otra familia pública')
                return self.snapshot()
            registry=LanguageVersionRegistry(self.families.root,name);head=registry.head()
            intent={'version':1,'requested':True,'base_name':name,'release_id':head['release']['release_id']}
            atomic_json(self.intent,intent)
            return self._begin(intent)

    def _begin(self,intent):
        if self.closed:raise RuntimeError('El nodo se está cerrando')
        settings=self._eligible();name=intent['base_name']
        registry=LanguageVersionRegistry(self.families.root,name)
        if registry.head()['release']['release_id']!=intent['release_id']:
            raise ValueError('La versión local cambió; vuelve a solicitar su reparación')
        chain=registry.history(intent['release_id']);required=repair_budget(self.families.store,chain)
        if managed_public_bytes(self.directory)+required>settings.policy.max_disk_gb*1024**3 or shutil.disk_usage(self.directory).free<required:
            raise ValueError('No hay espacio autorizado para reparar esta versión pública')
        self.cancel=threading.Event()
        self._status('starting',base_name=name,release_id=intent['release_id'],done=0,total=0,
            message='Recuperando los archivos de la versión pública verificada')
        self.thread=threading.Thread(target=self._run,args=(intent,settings),daemon=True,name='lixy-cache-repair')
        self.thread.start();return self.snapshot()

    def stop(self):
        self.cancel.set()
        if self.intent.exists():
            value=self._read_intent();value['requested']=False;atomic_json(self.intent,value)
        return self.snapshot()

    def update(self):
        if self.closed or self.thread is not None and self.thread.is_alive() or time.monotonic()<self.next_attempt:return
        try:
            value=self._read_intent()
            if (set(value)!={'version','requested','base_name','release_id'} or value['version']!=1
                    or type(value['requested']) is not bool or not isinstance(value['base_name'],str) or value['base_name'] not in BASES
                    or not re.fullmatch('[0-9a-f]{64}',str(value['release_id']))):return
            if not value['requested']:return
            with self.lock:
                if self.thread is not None and self.thread.is_alive():return
                self.next_attempt=time.monotonic()+30
                self._begin(value)
        except FileNotFoundError:
            return
        except (OSError,ValueError,TypeError,RuntimeError) as error:
            self.next_attempt=time.monotonic()+30
            self._status('waiting',message=str(error))

    def _run(self,intent,settings):
        permission=asdict(settings.policy);name=intent['base_name'];received={};last=0.;sizes={}
        def allowed():
            current=ClientSettings.load(self.directory)
            return (not self.closed and not self.cancel.is_set() and current.connected and not current.paused
                and self.knowledge.allowed and asdict(current.policy)==permission
                and self._read_intent()==intent
                and managed_public_bytes(self.directory)<=current.policy.max_disk_gb*1024**3)
        def preflight(chain):
            if chain[-1]['release']['release_id']!=intent['release_id']:
                raise ValueError('La reparación no puede seleccionar otra versión')
            sizes.update({identifier:size for bundle in chain for identifier,size in dependencies(bundle)})
            required=repair_budget(self.families.store,chain)
            if managed_public_bytes(self.directory)+required>settings.policy.max_disk_gb*1024**3 or shutil.disk_usage(self.directory).free<required:
                raise ValueError('El espacio autorizado cambió durante la reparación')
        def progress(identifier,done,_provider):
            nonlocal last
            received[identifier]=done
            if done==sizes.get(identifier) or time.monotonic()-last>=.5:
                last=time.monotonic();self._status('receiving',done=sum(received.values()),total=sum(sizes.values()))
        try:
            with NodeStateLock(self.directory/'preparation-lock'):
                providers=[]
                candidates=sorted(INITIAL_PROVIDERS)+[p['node_id'] for p in self.node.peers() if p.get('capabilities',{}).get('language_versions_v1')]
                for peer in list(dict.fromkeys(candidates))[:8]:
                    if not allowed():raise RuntimeError('La reparación quedó detenida con sus archivos conservados')
                    if peer!=self.node.identity.node_id_hex and self.node.connect_node(peer,timeout=12):providers.append(peer)
                if not providers:raise RuntimeError('Esperando una réplica pública disponible')
                report=replicate_version(self.knowledge.work,self.families.store,self.families.root,name,providers,
                    cancel_event=self.cancel,allowed=allowed,preflight=preflight,progress=progress,pinned_only=True,
                    receive_budget=BandwidthBudget(lambda:settings.policy.max_bandwidth_mbps/2 if allowed() else 0))
                if not allowed():raise RuntimeError('Los permisos cambiaron antes de terminar la reparación')
                current=self._read_intent();current['requested']=False;atomic_json(self.intent,current)
                self._status('complete',message='Archivos públicos recuperados y verificados; experto seleccionado sin cambios',
                    release_id=report['release_id'],model_id=report['model_id'],active_expert_changed=False)
        except Exception as error:
            self._status('stopped' if self.cancel.is_set() else 'waiting',message=str(error));self.next_attempt=time.monotonic()+30

    def close(self):
        self.closed=True;self.cancel.set()
        if self.thread is not None:self.thread.join(timeout=6)
