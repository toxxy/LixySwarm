"""Prepare a locally approved expert on the existing physical LSP node."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import threading
import time

from src.release.language_catalog import INITIAL_BUNDLES
from src.release.language_family import dependencies
from src.release.language_versions import LanguageVersionRegistry,dependencies as version_dependencies,validate_bundle
from .client_state import ClientSettings,NodeStateLock
from .engine_catalog import language_engine,engine_regular_bytes
from .engine_download import PreparationCancelled, prepare_engine
from .engine_package import verify_package
from .language_compute import LanguageSettings
from .language_install import materialize_version,model_hardlinks_available
from .language_replication import replicate_version
from .public_knowledge import managed_public_bytes
from .telemetry import atomic_json
from .compute_process import ComputeSupervisor
from .work_protocol import WorkExecution,WorkUnit
from src.contribution import ResourceRequirements
from src.learning.public_quality import quality_cases,correct_answer

# This adapter passed the public preservation gate. The smaller experimental
# adapter failed; being signed and downloadable does not make it selectable.
CHAT_FAMILY = 'qwen3-1.7b'
CHAT_ADAPTER = 'badef4394af180ab75ca25f0d9d47f2406b92809b53b5845ab65d451cf9c4b3c'
INITIAL_PROVIDERS = {'c45094e5fd8c64e8f6f5be3aca703edb6527711a504fcdc747a0fba15c668255',
    'ea914d97caafef310e50385034abae9a50ba86496b0e9cbf4679e6070e4c25b0'}
BUDGET_MARGIN_BYTES=8*1024**2


def preparation_budget(entry,*,hardlinks=False):
    bundle = INITIAL_BUNDLES[CHAT_FAMILY]
    model=sum(size for _,size in dependencies(bundle))*(1 if hardlinks else 2)
    regular=engine_regular_bytes(entry)
    return max(entry.bytes*(2 if entry.parts else 1),entry.bytes+regular,regular+model)+BUDGET_MARGIN_BYTES


def remaining_budget(directory, store, entry,*,chain=None,hardlinks=None):
    """Additional peak bytes: engine first, verified-cache release, then model."""
    directory = Path(directory)
    model_missing = 0
    required=dict(dependencies(INITIAL_BUNDLES[CHAT_FAMILY])) if chain is None else {
        identifier:size for bundle in chain for identifier,size in version_dependencies(bundle)}
    for identifier, size in required.items():
        try:
            if store.has(identifier) and store.manifest(identifier).size == size:
                continue
        except (OSError, ValueError):
            pass
        partial = store.incoming/f'{identifier}.part'
        present = min(size, partial.stat().st_size) if partial.is_file() else 0
        model_missing += size-present
    selected=INITIAL_BUNDLES[CHAT_FAMILY] if chain is None else chain[-1]
    selected_dependencies=dependencies(selected) if chain is None else version_dependencies(selected)
    if hardlinks is None:hardlinks=model_hardlinks_available(directory,store)
    model_copy = 0 if hardlinks or (directory/'language-models'/selected['release']['release_id']).is_dir() else sum(size for _, size in selected_dependencies)
    if (directory/'engines'/entry.manifest_sha256).is_dir():
        return model_missing+model_copy+BUDGET_MARGIN_BYTES
    download = directory/'engine-downloads'
    reclaim={}
    def present(identifier,maximum,suffixes):
        for suffix in suffixes:
            path=download/(identifier+suffix)
            if path.is_file() and not path.is_symlink():
                stat=path.stat();size=min(maximum,stat.st_size)
                reclaim[(stat.st_dev,stat.st_ino)]=size
                return size
        return 0
    complete=(download/(entry.sha256+'.zip')).is_file()
    if entry.parts:
        archive=present(entry.sha256,entry.bytes,['.zip'] if complete else ['.assembling'])
        pieces=sum(part.bytes-present(part.sha256,part.bytes,['.piece'] if complete else ['.piece','.part']) for part in entry.parts)
        downloading=0 if complete else pieces+entry.bytes-archive
    else:
        archive=present(entry.sha256,entry.bytes,['.zip','.part']);downloading=entry.bytes-archive
    cached=sum(reclaim.values());regular=engine_regular_bytes(entry)
    return max(0,downloading,entry.bytes+regular-cached,regular+model_missing+model_copy-cached)+BUDGET_MARGIN_BYTES


def eligible(settings, resources, sharing, entry):
    policy = settings.policy
    if entry is None:
        raise ValueError('Aún no hay un experto GPU publicado para este sistema')
    if not settings.connected or settings.paused or not sharing:
        raise ValueError('Conecta y autoriza conservar conocimiento público antes de preparar el experto')
    if (policy.mode not in {'balanced', 'maximum'} or policy.cpu_percent <= 0 or policy.max_cpu_jobs < 1
            or policy.max_ram_gb < 5 or policy.gpu_percent <= 0 or policy.max_gpu_jobs < 1
            or not resources.get('has_gpu')
            or resources.get('gpu_vram_gb', 0)*policy.gpu_percent/100/max(1, policy.max_gpu_jobs) < 4):
        raise ValueError('El experto necesita 5 GB de RAM y al menos 4 GB de GPU dentro de tus límites')
    if policy.max_bandwidth_mbps <= 0:
        raise ValueError('Autoriza conexión para preparar el experto')


class LanguagePreparation:
    def __init__(self, node, knowledge, families, directory, resources):
        self.node, self.knowledge, self.families = node, knowledge, families
        self.directory, self.resources = Path(directory), resources
        self.path = self.directory/'language-preparation.json'
        self.intent = self.directory/'language-preparation-intent.json'
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.thread = None
        self.closed = False
        self.next_attempt = 0
        self.state = {'state': 'idle', 'done': 0, 'total': 0, 'message': '', 'updated_at': time.time()}

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def _status(self, state, done=0, total=0, message=''):
        with self.lock:
            self.state = {'state': state, 'done': done, 'total': total, 'message': message, 'updated_at': time.time()}
            atomic_json(self.path, self.state)

    def start(self):
        if self.closed:
            raise RuntimeError('El nodo se está cerrando')
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                return self.snapshot()
            settings = ClientSettings.load(self.directory)
            entry = language_engine('cuda')
            eligible(settings, self.resources, self.knowledge.allowed, entry)
            LanguageSettings.load(self.directory).validate()
            required = remaining_budget(self.directory, self.families.store, entry)
            if managed_public_bytes(self.directory)+required > settings.policy.max_disk_gb*1024**3:
                raise ValueError(f'Autoriza al menos {(managed_public_bytes(self.directory)+required)/1024**3:.1f} GB de almacenamiento para esta preparación')
            if shutil.disk_usage(self.directory).free < required:
                raise ValueError('No hay suficiente espacio libre para preparar el experto')
            atomic_json(self.intent, {'version': 1, 'requested': True})
            self.cancel = threading.Event()
            self._status('starting', message='Preparando el experto con tus permisos')
            self.thread = threading.Thread(target=self._run, args=(settings, entry), daemon=True, name='lixy-language-preparation')
            self.thread.start()
            return self.snapshot()

    def stop(self):
        self.cancel.set()
        atomic_json(self.intent, {'version': 1, 'requested': False})
        return self.snapshot()

    def update(self):
        if self.closed or self.thread is not None and self.thread.is_alive():
            return
        try:
            if self.intent.stat().st_size > 1024:
                return
            requested = json.loads(self.intent.read_text()) == {'version': 1, 'requested': True}
        except (OSError, ValueError, TypeError):
            return
        if requested and time.monotonic() >= self.next_attempt:
            self.next_attempt = time.monotonic()+30
            try:
                self.start()
            except (OSError, ValueError, RuntimeError) as error:
                self._status('waiting', message=str(error))

    def _run(self, settings, entry):
        permission = asdict(settings.policy)
        previous = asdict(LanguageSettings.load(self.directory))
        def allowed():
            current = ClientSettings.load(self.directory)
            return (not self.closed and not self.cancel.is_set() and current.connected and not current.paused
                and self.knowledge.allowed and asdict(current.policy) == permission
                and asdict(LanguageSettings.load(self.directory)) == previous
                and managed_public_bytes(self.directory) <= current.policy.max_disk_gb*1024**3)
        guard=NodeStateLock(self.directory/'preparation-lock');acquired=False
        try:
            guard.__enter__();acquired=True
            required=remaining_budget(self.directory,self.families.store,entry)
            if managed_public_bytes(self.directory)+required>settings.policy.max_disk_gb*1024**3 or shutil.disk_usage(self.directory).free<required:
                raise ValueError('El almacenamiento disponible cambió; la preparación se conserva para reanudarla')
            if not allowed():
                raise PreparationCancelled('Preparación detenida; los archivos se conservan')
            self._status('engine', message='Preparando el motor de lenguaje')
            installed = self.directory/'engines'/entry.manifest_sha256
            if installed.is_dir():
                manifest = verify_package(installed, entry.manifest_sha256, profile=entry.profile)
                command = str(installed/manifest['entrypoint'])
            else:
                command = prepare_engine(entry, self.directory, allowed=allowed,
                    rate_mbps=lambda: settings.policy.max_bandwidth_mbps/2,
                    cancel_event=self.cancel, progress=lambda state, done, total: self._status('engine-'+state, done, total, 'Preparando el motor de lenguaje'))
                verify_package(Path(command).parent, entry.manifest_sha256, profile=entry.profile)
            if not allowed():raise PreparationCancelled('Los permisos cambiaron; el experto no se activó')
            self._status('discovering',message='Buscando la versión evaluada del experto público')
            release,base = self._prepare_version(settings,entry,allowed)
            adapter_id=release.model_artifact_id
            accepted_release_id=release.release_id
            unchanged_permissions=allowed
            def allowed():
                return (unchanged_permissions() and LanguageVersionRegistry(self.families.root,CHAT_FAMILY).head()['release']['release_id']==accepted_release_id)
            if not allowed():
                raise PreparationCancelled('Los permisos cambiaron; el experto no se activó')
            self._status('testing',message='Comprobando que el experto responde dentro de tus límites')
            self._probe(command,base,settings,allowed,adapter_id)
            if not allowed():
                raise PreparationCancelled('Los permisos cambiaron; el experto no se activó')
            LanguageSettings(enabled=True, command=[command], base_name=CHAT_FAMILY, base_directory=str(base),
                adapter_file=str(base/'adapter.safetensors'), adapter_id=adapter_id).save(self.directory)
            atomic_json(self.intent, {'version': 1, 'requested': False})
            self._status('complete', message='Experto preparado; responde y colabora dentro de tus permisos')
        except Exception as error:
            try:
                if asdict(LanguageSettings.load(self.directory)) != previous:
                    atomic_json(self.intent, {'version': 1, 'requested': False})
            except (OSError, ValueError, TypeError):
                atomic_json(self.intent, {'version': 1, 'requested': False})
            self._status('stopped' if self.cancel.is_set() else 'waiting', message=str(error))
            self.next_attempt = time.monotonic()+30
        finally:
            if acquired:guard.__exit__(None,None,None)

    def _prepare_version(self,settings,entry,allowed):
        registry=LanguageVersionRegistry(self.families.root,CHAT_FAMILY)
        candidates=[peer['node_id'] for peer in self.node.peers()
            if CHAT_FAMILY in peer.get('resources',{}).get('language_families_v1',[])
            or peer.get('capabilities',{}).get('language_versions_v1')]
        providers=[]
        for identifier in list(dict.fromkeys(candidates+sorted(INITIAL_PROVIDERS)))[:8]:
            if not allowed():raise PreparationCancelled('Los permisos cambiaron; el experto no se activó')
            if identifier!=self.node.identity.node_id_hex and self.node.connect_node(identifier,timeout=12):providers.append(identifier)
        sizes={};received={};last_progress=0
        def preflight(chain,hardlinks=None):
            sizes.update({identifier:size for bundle in chain for identifier,size in version_dependencies(bundle)})
            required=remaining_budget(self.directory,self.families.store,entry,chain=chain,hardlinks=hardlinks)
            if (managed_public_bytes(self.directory)+required>settings.policy.max_disk_gb*1024**3
                    or shutil.disk_usage(self.directory).free<required):
                raise ValueError('No hay espacio autorizado para la versión aceptada y su motor')
        def progress(identifier,done,_provider):
            nonlocal last_progress
            received[identifier]=done
            if done==sizes.get(identifier) or time.monotonic()-last_progress>=.5:
                last_progress=time.monotonic()
                self._status('versions',sum(received.values()),sum(sizes.values()),'Verificando la versión evaluada del experto')
        if providers:
            self._status('versions',message='Adquiriendo la versión evaluada del experto')
            from .public_knowledge import BandwidthBudget
            replicate_version(self.knowledge.work,self.families.store,self.families.root,CHAT_FAMILY,providers,
                cancel_event=self.cancel,allowed=allowed,preflight=preflight,
                receive_budget=BandwidthBudget(lambda:settings.policy.max_bandwidth_mbps/2 if allowed() else 0),
                progress=progress)
        head=registry.head();chain=registry.history(head['release']['release_id'])
        preflight(chain);registry.install(chain,self.families.store)
        release,descriptor=validate_bundle(head)
        if descriptor['base_name']!=CHAT_FAMILY:raise ValueError('La versión aceptada pertenece a otra familia')
        links=model_hardlinks_available(self.directory,self.families.store)
        # The immediately preceding preflight reserved copies if links are
        # unavailable. If links disappear later, never silently exceed that plan.
        preflight(chain,hardlinks=links)
        base=materialize_version(self.directory,CHAT_FAMILY,release.release_id,allowed=allowed,require_hardlinks=links)
        return release,base

    def _probe(self,command,base,settings,allowed,adapter_id=None):
        if adapter_id is None:adapter_id=CHAT_ADAPTER
        requirements=ResourceRequirements(kind='inference',gpu_required=True,ram_gb=5)
        lease,reason=self.knowledge.governor.acquire(requirements)
        if lease is None:raise RuntimeError('El cálculo no está disponible ahora: '+reason)
        with lease:
            execution=WorkExecution(WorkUnit.create(origin_node_id=self.node.identity.node_id_hex,
                operation='language.inference.v1',kind='inference',requirements=requirements,payload={},timeout_s=60),self.cancel)
            case=quality_cases()[0]
            request={'action':'inference','device':'cuda','base_name':CHAT_FAMILY,'base_directory':str(base),
                'adapter_file':str(base/'adapter.safetensors'),'adapter_id':adapter_id,
                'messages':[{'role':'user','content':case['prompt']}],'deadline':execution.deadline,
                'cpu_threads':max(1,min(8,int(self.resources.get('cpu_cores',1)*settings.policy.cpu_percent/100))),
                'gpu_fraction':settings.policy.gpu_percent/100/max(1,settings.policy.max_gpu_jobs)}
            result=ComputeSupervisor().run([command],request,execution,allowed=allowed,ram_gb=5)
            if (set(result)!={'model_id','answer','output_tokens'} or result['model_id']!=adapter_id
                    or type(result['output_tokens']) is not int or not 1<=result['output_tokens']<=128
                    or not isinstance(result['answer'],str) or len(result['answer'].encode())>8192
                    or not correct_answer(case,result['answer'])):
                raise ValueError('El experto no pasó la comprobación de funcionamiento; no se activó')

    def close(self):
        self.closed = True
        self.cancel.set()
        if self.thread is not None:
            self.thread.join(timeout=6)
