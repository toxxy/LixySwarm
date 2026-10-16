"""Prepare the organism on this native identity using locally pinned public inputs."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import threading
import time
import uuid

from src.contribution import ResourceRequirements
from . import own_catalog as catalog
from .client_state import ClientSettings, NodeStateLock
from .compute_process import ComputeSupervisor
from .engine_catalog import engine_regular_bytes
from .engine_download import prepare_engine, PreparationCancelled
from .engine_package import verify_package
from .own_network_settings import OwnNetworkSettings
from .own_swarm import OwnSwarmSettings
from .public_knowledge import managed_public_bytes
from .report_privacy import public_route
from .telemetry import atomic_json
from .work_protocol import WorkExecution, WorkUnit

RESERVE_BYTES = 160 * 1024**2


def storage_guard(directory, maximum, authorized, *, clock=time.monotonic):
    """Check consent every time; amortize scans under the full peak reservation."""
    next_scan=0.;used=None
    def allowed():
        nonlocal next_scan,used
        try:
            if not authorized():return False
            now=clock()
            if used is None or now>=next_scan:
                used=managed_public_bytes(directory);next_scan=now+.5
            return used<=maximum
        except (OSError,ValueError,TypeError):return False
    return allowed


def eligible(settings, sharing, entry):
    p = settings.policy
    if entry is None: raise ValueError('El motor propio todavía no está publicado para este sistema')
    if not settings.connected or settings.paused or not sharing:
        raise ValueError('Conecta y autoriza conservar conocimiento público para preparar el organismo')
    if (p.mode not in {'balanced', 'maximum'} or p.cpu_percent <= 0 or p.max_cpu_jobs < 1
            or p.max_ram_gb < 4 or p.max_bandwidth_mbps <= 0):
        raise ValueError('El organismo necesita CPU autorizada, conexión y 4 GB de RAM dentro de tus límites')


class OwnPreparation:
    def __init__(self, service):
        self.service = service; self.knowledge = service.knowledge; self.directory = service.directory
        self.path = self.directory / 'own-preparation.json'; self.intent = self.directory / 'own-preparation-intent.json'
        self.plan = self.directory / 'own-preparation-plan.json'
        self.lock = threading.RLock(); self.changed = threading.Event(); self.cancel = threading.Event()
        self.thread = None; self.closed = False; self.next_attempt = 0; self.permission = None
        self.state = {'state': 'idle', 'done': 0, 'total': 0, 'message': '', 'anchor_id': catalog.anchor_id()}

    def snapshot(self):
        with self.lock: return dict(self.state)

    def _status(self, state, done=0, total=0, message=''):
        with self.lock:
            self.state = {'state': state, 'done': done, 'total': total, 'message': message,
                'anchor_id': catalog.anchor_id(), 'updated_at': time.time()}
            atomic_json(self.path, self.state); self.changed.set()

    def _network_controls(self):
        return OwnNetworkSettings(follow_enabled=True, owner_id=catalog.COORDINATOR,
            initial_state_id=catalog.ANCHOR['state_id'])

    def _remaining(self, entry):
        missing = sum(size for identifier, size in catalog.dependencies() if not self.service.store.has(identifier))
        engine = self.directory / 'engines' / entry.manifest_sha256
        return missing + RESERVE_BYTES + (0 if engine.is_dir() else entry.bytes + engine_regular_bytes(entry))

    def _preflight(self, settings, entry):
        needed = self._remaining(entry)
        if (managed_public_bytes(self.directory) + needed > settings.policy.max_disk_gb * 1024**3
                or shutil.disk_usage(self.directory).free < needed):
            raise ValueError('El almacenamiento autorizado no alcanza para el motor y el estado propio')

    def start(self):
        with self.lock:
            if self.closed: raise RuntimeError('El nodo se está cerrando')
            if self.thread is not None and self.thread.is_alive(): return self.snapshot()
            settings = ClientSettings.load(self.directory); entry = catalog.own_engine()
            eligible(settings, self.knowledge.allowed, entry)
            if (self.directory / 'own-swarm.json').exists():
                if self.service.ready:
                    self._status('complete', message='El organismo ya está preparado en este equipo')
                    atomic_json(self.intent, {'version': 1, 'requested': False}); return self.snapshot()
                raise ValueError('Ya existe una selección propia; se conserva sin reemplazarla por el ancla inicial')
            expected = {'version': 1, 'anchor_id': catalog.anchor_id(),
                'network_controls': asdict(self._network_controls())}
            if self.plan.exists():
                if self.plan.stat().st_size > 4096 or json.loads(self.plan.read_text()) != expected:
                    raise ValueError('La preparación anterior conserva otro contrato')
            if (self.directory / 'own-network-controls.json').exists():
                if not self.plan.exists() or asdict(OwnNetworkSettings.load(self.directory)) != expected['network_controls']:
                    raise ValueError('La configuración de colaboración propia ya existe; no se reemplaza')
            self._preflight(settings, entry); atomic_json(self.plan, expected)
            self.permission = asdict(settings.policy)
            atomic_json(self.intent, {'version': 1, 'requested': True})
            self.cancel = threading.Event(); self._status('starting', message='Preparando el organismo propio')
            self.thread = threading.Thread(target=self._run, args=(settings, entry), name='lixy-own-preparation', daemon=True)
            self.thread.start(); return self.snapshot()

    def stop(self):
        self.cancel.set(); atomic_json(self.intent, {'version': 1, 'requested': False})
        return self.snapshot()

    def update(self):
        if self.closed: return
        if self.thread is not None and self.thread.is_alive():
            try:
                current = ClientSettings.load(self.directory)
                if (not self.knowledge.allowed or not current.connected or current.paused
                        or asdict(current.policy) != self.permission): self.cancel.set()
            except (OSError, ValueError, TypeError): self.cancel.set()
            return
        try:
            if self.intent.stat().st_size > 1024: return
            requested = json.loads(self.intent.read_text()) == {'version': 1, 'requested': True}
        except (OSError, ValueError, TypeError): return
        try:
            current = ClientSettings.load(self.directory)
            if not current.connected or current.paused or not self.knowledge.allowed: return
        except (OSError, ValueError, TypeError): return
        if requested and time.monotonic() >= self.next_attempt:
            self.next_attempt = time.monotonic() + 30
            try: self.start()
            except (OSError, ValueError, RuntimeError) as error: self._status('waiting', message=str(error))

    def _run(self, settings, entry):
        permission = asdict(settings.policy); transfers = []; failures=[]
        attempt=self.directory/'public/own/preparation-attempts'/(uuid.uuid4().hex+'.json')
        started=time.time()
        def authorized():
            current = ClientSettings.load(self.directory)
            return (not self.closed and not self.cancel.is_set() and self.knowledge.allowed
                and current.connected and not current.paused and asdict(current.policy) == permission
                and not (self.directory / 'own-swarm.json').exists()
                and json.loads(self.intent.read_text()) == {'version': 1, 'requested': True}
                and (not (self.directory / 'own-network-controls.json').exists()
                    or asdict(OwnNetworkSettings.load(self.directory)) == asdict(self._network_controls())))
        allowed=storage_guard(self.directory,settings.policy.max_disk_gb*1024**3,authorized)
        def check():
            if not allowed(): raise PreparationCancelled('Preparación detenida; el progreso se conserva')
        try:
            with NodeStateLock(self.directory / 'preparation-lock'):
                check(); self._preflight(settings, entry)
                installed = self.directory / 'engines' / entry.manifest_sha256
                if installed.is_dir():
                    self._status('engine-verifying',message='Verificando el motor conservado; la descarga anterior se reutiliza')
                    manifest = verify_package(installed, entry.manifest_sha256, profile=entry.profile, check=check)
                    command = str(installed / manifest['entrypoint'])
                else:
                    command = prepare_engine(entry, self.directory, allowed=allowed,
                        rate_mbps=lambda: settings.policy.max_bandwidth_mbps / 2, cancel_event=self.cancel,
                        progress=lambda stage, done, total: self._status('engine-' + stage, done, total, 'Preparando el motor propio'))
                check(); self._status('discovering', message='Buscando participantes que conservan el estado propio')
                candidates = [row['node_id'] for row in self.knowledge.node.peers()
                    if row.get('capabilities', {}).get('own_swarm_v2')]
                providers = list(dict.fromkeys(candidates + [catalog.COORDINATOR]))[:16]
                self._status('receiving', total=sum(size for _, size in catalog.dependencies()),
                    message='Adquiriendo pesos, optimizador y datos públicos del organismo')
                done = 0
                for identifier, size in catalog.dependencies():
                    check()
                    if not self.service.store.has(identifier):
                        for peer in providers:
                            try:
                                self._fetch(identifier, size, peer, check, transfers,
                                    progress=lambda offset,_size:self._status('receiving',done+offset,
                                        sum(n for _,n in catalog.dependencies()),'Recibiendo y verificando el estado propio'))
                                break
                            except PreparationCancelled: raise
                            except (OSError, ValueError, RuntimeError, StopIteration) as error:
                                failures.append({'provider':peer,'artifact_id':identifier,'type':type(error).__name__,
                                    'error':str(error)[:180],'at':time.time()})
                                continue
                        else: raise RuntimeError('No hay un proveedor público disponible del estado propio fijado')
                    from .artifact_store import digest_file
                    if digest_file(self.service.store.object_path(identifier)) != (identifier, size):
                        raise ValueError('El estado propio adquirido no coincide con el ancla verificada')
                    done += size; self._status('receiving', done, sum(n for _, n in catalog.dependencies()),
                        'Conservando el estado de aprendizaje propio')
                check(); self._status('validating', message='Comprobando pesos propios y continuidad del optimizador')
                audit = self._validate(command, settings, allowed)
                check(); atomic_json(self.directory / 'own-preparation-validation.json', audit)
                with NodeStateLock(self.directory / 'own-state-selection'):
                    check()
                    controls = self._network_controls(); controls.save(self.directory)
                    own = OwnSwarmSettings(version=2, enabled=True, command=[command],
                        model_id=catalog.ANCHOR['model_id'], optimizer_id=catalog.ANCHOR['optimizer_id'],
                        step=catalog.ANCHOR['step'], requesters=sorted({catalog.COORDINATOR, self.knowledge.node.identity.node_id_hex}))
                    own.save(self.directory)
                atomic_json(self.directory / 'own-preparation-result.json', {'success': True,
                    'anchor_id': catalog.anchor_id(), 'state_id': catalog.ANCHOR['state_id'],
                    'engine_manifest_sha256': entry.manifest_sha256, 'transfers': transfers, 'audit': audit})
                atomic_json(self.intent, {'version': 1, 'requested': False})
                self._status('complete', message='Organismo preparado; disponible para colaborar con la red')
                atomic_json(attempt,{'started_at':started,'finished_at':time.time(),'success':True,
                    'anchor_id':catalog.anchor_id(),'provider_failures':failures})
        except PreparationCancelled as error:
            self._status('stopped', message=str(error))
            atomic_json(attempt,{'started_at':started,'finished_at':time.time(),'success':False,
                'cancelled':True,'error':str(error)[:180],'provider_failures':failures})
        except Exception as error:
            self.next_attempt=time.monotonic()+30
            self._status('stopped' if self.cancel.is_set() else 'waiting',
                message='Preparación detenida; el progreso se conserva' if self.cancel.is_set() else type(error).__name__ + ': ' + str(error)[:180])
            atomic_json(attempt,{'started_at':started,'finished_at':time.time(),'success':False,
                'type':type(error).__name__,'error':str(error)[:180],'provider_failures':failures})

    def _fetch(self, identifier, size, peer, check, transfers, *, progress=lambda *_:None):
        node = self.knowledge.node; check()
        if peer not in node.connected_node_ids and not node.connect_node(peer, timeout=12):
            raise RuntimeError('Proveedor propio sin ruta disponible')
        row = next(p for p in node.peers() if p['node_id'] == peer)
        route = public_route({k: row[k] for k in ['host', 'relay_host', 'transport', 'encrypted'] if k in row})
        if not route['encrypted'] or not route['public_internet']:
            raise ValueError('El estado propio debe adquirirse por Internet público cifrado')
        settings = ClientSettings.load(self.directory)
        if managed_public_bytes(self.directory) + 2*size + RESERVE_BYTES > settings.policy.max_disk_gb*1024**3:
            raise ValueError('La adquisición excede el almacenamiento autorizado')
        began = time.monotonic(); chunks = []
        self.service.transfer.fetch(identifier, peer_id=peer, expected_size=size, timeout_s=600,
            cancel_event=self.cancel, observer=chunks.append,progress=progress)
        check(); transfers.append({'artifact_id': identifier, 'bytes': size, 'provider': peer, 'route': route,
            'seconds': time.monotonic() - began, 'encoded_data_bytes': sum(v['encoded_data_bytes'] for v in chunks)})

    def _validate(self, command, settings, allowed):
        from src.learning.public_math import ARTIFACTS
        requirements = ResourceRequirements(kind='training', ram_gb=4, disk_gb=.08)
        lease, reason = self.knowledge.governor.acquire(requirements)
        if lease is None: raise RuntimeError('Esperando recursos para comprobar el organismo: ' + reason)
        with lease:
            execution = WorkExecution(WorkUnit.create(origin_node_id=self.knowledge.node.identity.node_id_hex,
                operation='swarm.validate.v1', kind='training', requirements=requirements, payload={}, timeout_s=180), self.cancel)
            request = {k: catalog.ANCHOR[k] for k in ['model_id', 'optimizer_id', 'step', 'state_id', 'initialization_protocol_id']}
            request.update(action='own-validate', model_file=str(self.service.store.object_path(catalog.ANCHOR['model_id'])),
                optimizer_file=str(self.service.store.object_path(catalog.ANCHOR['optimizer_id'])),
                evaluation_file=str(self.service.store.object_path(ARTIFACTS['evaluation'][0])), device='cpu',
                cpu_threads=max(1, min(8, int(self.service.resources['cpu_cores']*settings.policy.cpu_percent/100))),
                deadline=execution.deadline)
            result = ComputeSupervisor().run([command], request, execution, allowed=allowed, ram_gb=4)
        expected = catalog.ANCHOR['evaluation']; observed = result.get('evaluation', {})
        if (result.get('success') is not True or result.get('state_id') != catalog.ANCHOR['state_id']
                or result.get('model_id') != catalog.ANCHOR['model_id'] or result.get('optimizer_id') != catalog.ANCHOR['optimizer_id']
                or result.get('initialization_protocol_id') != catalog.ANCHOR['initialization_protocol_id']
                or result.get('optimizer_step') != catalog.ANCHOR['step'] or result.get('optimizer_states') != 175
                or result.get('saved_state_unchanged') is not True or result.get('new_gradient_computed') is not False
                or result.get('cuda_initialized') is not False or result.get('causality', {}).get('passed') is not True
                or observed.get('documents') != expected['documents'] or observed.get('predicted_bytes') != expected['predicted_bytes']
                or any(type(observed.get(k)) not in (int,float) or not math.isfinite(observed[k]) for k in ['nll','byte_accuracy'])
                or abs(observed.get('nll', 0) - expected['nll']) > 1e-5
                or abs(observed.get('byte_accuracy', 0) - expected['byte_accuracy']) > 1e-12):
            raise ValueError('El organismo adquirido no pasó la comprobación de continuidad y funcionamiento')
        return result

    def close(self):
        self.closed = True; self.cancel.set()
        if self.thread is not None: self.thread.join(timeout=6)
