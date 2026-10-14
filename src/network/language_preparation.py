"""Prepare a locally approved expert on the existing physical LSP node."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import threading
import time

from src.release.language_catalog import INITIAL_BUNDLES
from src.release.language_family import dependencies, load_bundle, verify_bundle
from .client_state import ClientSettings,NodeStateLock
from .engine_catalog import language_engine
from .engine_download import PreparationCancelled, prepare_engine
from .engine_package import verify_package
from .language_compute import LanguageSettings
from .language_families import fetch_family
from .language_install import materialize_family
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
INITIAL_PROVIDERS = {'c45094e5fd8c64e8f6f5be3aca703edb6527711a504fcdc747a0fba15c668255'}


def preparation_budget(entry):
    bundle = INITIAL_BUNDLES[CHAT_FAMILY]
    # Two copies cover filesystems without hardlinks and staging installation.
    # Multipart archives retain their pieces plus the joined archive.
    return 2*sum(size for _, size in dependencies(bundle)) + 2*entry.installed_bytes + entry.bytes*(2 if entry.parts else 1)


def remaining_budget(directory, store, entry):
    """Reserve remaining stages, including resumable archives and staging."""
    directory = Path(directory)
    model_missing = 0
    for identifier, size in dependencies(INITIAL_BUNDLES[CHAT_FAMILY]):
        try:
            if store.manifest(identifier).size == size:
                continue
        except (OSError, ValueError):
            pass
        partial = store.incoming/f'{identifier}.part'
        present = min(size, partial.stat().st_size) if partial.is_file() else 0
        model_missing += size-present
    model_copy = 0 if (directory/'language-models'/INITIAL_BUNDLES[CHAT_FAMILY]['release']['release_id']).is_dir() else sum(size for _, size in dependencies(INITIAL_BUNDLES[CHAT_FAMILY]))
    if (directory/'engines'/entry.manifest_sha256).is_dir():
        return model_missing+model_copy
    download = directory/'engine-downloads'
    def present(identifier, maximum, suffixes):
        return max((min(maximum, (download/(identifier+suffix)).stat().st_size)
            for suffix in suffixes if (download/(identifier+suffix)).is_file()), default=0)
    archive = entry.bytes-present(entry.sha256, entry.bytes, ['.zip', '.assembling'] if entry.parts else ['.zip', '.part'])
    pieces = sum(part.bytes-present(part.sha256, part.bytes, ['.piece', '.part']) for part in entry.parts)
    return model_missing+model_copy+archive+pieces+2*entry.installed_bytes


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
            try:
                bundle = load_bundle(self.families.root, CHAT_FAMILY)
                if not all(self.families.store.has(identifier) for identifier, _ in dependencies(bundle)):
                    raise ValueError('Incomplete local family')
            except (OSError, ValueError):
                self._status('discovering', message='Buscando participantes con el experto público')
                provider = None
                candidates = [p['node_id'] for p in self.node.peers()
                    if CHAT_FAMILY in p.get('resources', {}).get('language_families_v1', [])]
                for identifier in list(dict.fromkeys(candidates + sorted(INITIAL_PROVIDERS))):
                    if not allowed():
                        raise PreparationCancelled('Preparación detenida; los archivos se conservan')
                    if self.node.connect_node(identifier, timeout=12):
                        provider = identifier
                        break
                if provider is None:
                    raise RuntimeError('Esperando un participante que conserve el experto')
                total = sum(size for _, size in dependencies(INITIAL_BUNDLES[CHAT_FAMILY]))
                self._status('model', 0, total, 'Descargando y verificando el experto por la red')
                received = 0
                last_progress = 0
                def progress(_identifier, size):
                    nonlocal received
                    received += size
                    self._status('model', received, total, 'Descargando y verificando el experto por la red')
                def chunk_progress(_identifier, done, size):
                    nonlocal last_progress
                    if done==size or time.monotonic()-last_progress>=.5:
                        last_progress=time.monotonic()
                        self._status('model', received+done, total, 'Descargando y verificando el experto por la red')
                from .public_knowledge import BandwidthBudget
                bundle = fetch_family(self.knowledge.work, self.families.store, self.families.root, CHAT_FAMILY, provider,
                    cancel_event=self.cancel, progress=progress,chunk_progress=chunk_progress,
                    receive_budget=BandwidthBudget(lambda: settings.policy.max_bandwidth_mbps/2 if allowed() else 0))
            release, descriptor = verify_bundle(bundle)
            if descriptor['base_name'] != CHAT_FAMILY or release.model_artifact_id != CHAT_ADAPTER:
                raise ValueError('Este experto no supera la política local de activación')
            self._status('materializing', message='Verificando los archivos del experto')
            base = materialize_family(self.directory, CHAT_FAMILY, allowed=allowed)
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
            if not allowed():
                raise PreparationCancelled('Los permisos cambiaron; el experto no se activó')
            self._status('testing',message='Comprobando que el experto responde dentro de tus límites')
            self._probe(command,base,settings,allowed)
            if not allowed():
                raise PreparationCancelled('Los permisos cambiaron; el experto no se activó')
            LanguageSettings(enabled=True, command=[command], base_name=CHAT_FAMILY, base_directory=str(base),
                adapter_file=str(base/'adapter.safetensors'), adapter_id=CHAT_ADAPTER).save(self.directory)
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

    def _probe(self,command,base,settings,allowed):
        requirements=ResourceRequirements(kind='inference',gpu_required=True,ram_gb=5)
        lease,reason=self.knowledge.governor.acquire(requirements)
        if lease is None:raise RuntimeError('El cálculo no está disponible ahora: '+reason)
        with lease:
            execution=WorkExecution(WorkUnit.create(origin_node_id=self.node.identity.node_id_hex,
                operation='language.inference.v1',kind='inference',requirements=requirements,payload={},timeout_s=60),self.cancel)
            case=quality_cases()[0]
            request={'action':'inference','device':'cuda','base_name':CHAT_FAMILY,'base_directory':str(base),
                'adapter_file':str(base/'adapter.safetensors'),'adapter_id':CHAT_ADAPTER,
                'messages':[{'role':'user','content':case['prompt']}],'deadline':execution.deadline,
                'cpu_threads':max(1,min(8,int(self.resources.get('cpu_cores',1)*settings.policy.cpu_percent/100))),
                'gpu_fraction':settings.policy.gpu_percent/100/max(1,settings.policy.max_gpu_jobs)}
            result=ComputeSupervisor().run([command],request,execution,allowed=allowed,ram_gb=5)
            if (set(result)!={'model_id','answer','output_tokens'} or result['model_id']!=CHAT_ADAPTER
                    or type(result['output_tokens']) is not int or not 1<=result['output_tokens']<=128
                    or not isinstance(result['answer'],str) or len(result['answer'].encode())>8192
                    or not correct_answer(case,result['answer'])):
                raise ValueError('El experto no pasó la comprobación de funcionamiento; no se activó')

    def close(self):
        self.closed = True
        self.cancel.set()
        if self.thread is not None:
            self.thread.join(timeout=6)
