"""Cold own preparation preserves consent, existing progress and typed activation."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import urllib.error

import pytest

from src.contribution import ContributionPolicy
from src.network import own_preparation as module
from src.network.own_preparation import OwnPreparation
from src.network.own_network_settings import OwnNetworkSettings
from src.network.own_swarm import OwnSwarmSettings
from src.network.client_state import ClientSettings
from src.network.artifact_store import ArtifactStore
from src.learning.swarm_policy import state_id
from src.network.lsp import LSPIdentity


def fixture(tmp_path, monkeypatch):
    own = LSPIdentity.generate().node_id_hex; provider = LSPIdentity.generate().node_id_hex
    data = [b'own weights fixture', b'continued own optimizer fixture', b'public training', b'public evaluation']
    objects = {hashlib.sha256(raw).hexdigest(): raw for raw in data}; identifiers = list(objects)
    anchor = {**deepcopy(module.catalog.ANCHOR), 'model_id': identifiers[0], 'model_bytes': len(data[0]),
        'optimizer_id': identifiers[1], 'optimizer_bytes': len(data[1])}
    anchor['state_id'] = state_id(anchor['model_id'], anchor['optimizer_id'], anchor['step'])
    monkeypatch.setattr(module.catalog, 'ANCHOR', anchor); monkeypatch.setattr(module.catalog, 'COORDINATOR', provider)
    monkeypatch.setattr(module.catalog, 'dependencies', lambda: [(k, len(v)) for k, v in objects.items()])
    entry = SimpleNamespace(manifest_sha256='f'*64, bytes=1024, installed_bytes=1024, profile='own-cpu-bootstrap')
    monkeypatch.setattr(module.catalog, 'own_engine', lambda: entry)
    policy = ContributionPolicy.for_mode('maximum'); policy.max_disk_gb = 2; policy.max_ram_gb = 4
    ClientSettings(policy=policy).save(tmp_path)
    store = ArtifactStore(tmp_path/'public/own/artifacts'); cancelled = []
    node = SimpleNamespace(identity=SimpleNamespace(node_id_hex=own), connected_node_ids={provider},
        peers=lambda: [{'node_id': provider, 'capabilities': {'own_swarm_v2': True},
            'transport': 'relay', 'relay_host': '31.97.9.54', 'encrypted': True}],
        connect_node=lambda *_a, **_kw: True)
    def fetch(identifier, **options):
        cancelled.append(options['cancel_event'])
        path = tmp_path/'fixture-source'; path.write_bytes(objects[identifier]); return store.import_file(path).artifact_id
    service = SimpleNamespace(directory=tmp_path, knowledge=SimpleNamespace(node=node, allowed=True),
        store=store, ready=False, resources={'cpu_cores': 4}, transfer=SimpleNamespace(fetch=fetch))
    def prepare(_entry, directory, **kw):
        assert kw['allowed']()
        command = Path(directory)/'engines'/_entry.manifest_sha256/'engine'; command.parent.mkdir(parents=True)
        command.write_text('source-independent code fixture'); return str(command)
    monkeypatch.setattr(module, 'prepare_engine', prepare)
    monkeypatch.setattr(module, 'verify_package', lambda *_a, **kw: {'entrypoint': 'engine'})
    p = OwnPreparation(service)
    report = {'success': True, 'state_id': anchor['state_id'], 'optimizer_states': 175,
        'initialization_protocol_id': anchor['initialization_protocol_id'],
        'model_id': anchor['model_id'], 'optimizer_id': anchor['optimizer_id'], 'optimizer_step': anchor['step'],
        'causality': {'passed': True}, 'new_gradient_computed': False, 'cuda_initialized': False,
        'saved_state_unchanged': True, 'evaluation': deepcopy(anchor['evaluation'])}
    monkeypatch.setattr(p, '_validate', lambda *_a, **_kw: deepcopy(report))
    return p, report, cancelled, anchor, provider


def finish(p):
    p.thread.join(timeout=5)
    assert not p.thread.is_alive()
    return p.snapshot()


def test_cold_acquisition_selects_only_after_validation_and_keeps_optimizer_progress(tmp_path, monkeypatch):
    p, report, calls, anchor, provider = fixture(tmp_path, monkeypatch)
    before = (tmp_path/'controls.json').read_bytes()
    try:
        p.start(); assert finish(p)['state'] == 'complete'
        own = OwnSwarmSettings.load(tmp_path); network = OwnNetworkSettings.load(tmp_path)
        assert own.version == 2 and own.enabled and own.step == 1033 and own.model_id == anchor['model_id']
        assert network.follow_enabled and not network.enabled and network.owner_id == provider
        assert (tmp_path/'controls.json').read_bytes() == before and len(calls) == 4
        proof = json.loads((tmp_path/'own-preparation-result.json').read_text())
        assert proof['audit'] == report and all(t['route']['public_internet'] and t['route']['encrypted'] for t in proof['transfers'])
        assert not json.loads(p.intent.read_text())['requested']
    finally: p.close()


def test_existing_selected_learning_is_never_replaced_by_bootstrap_anchor(tmp_path, monkeypatch):
    p, *_ = fixture(tmp_path, monkeypatch)
    OwnSwarmSettings().save(tmp_path); before = (tmp_path/'own-swarm.json').read_bytes()
    with pytest.raises(ValueError, match='selección propia'): p.start()
    assert (tmp_path/'own-swarm.json').read_bytes() == before and not p.intent.exists()


def test_pause_revokes_inflight_transfer_and_keeps_unselected_progress(tmp_path, monkeypatch):
    p, *_ = fixture(tmp_path, monkeypatch); entered = threading.Event()
    def fetch(*_a, **kw):
        entered.set(); assert kw['cancel_event'].wait(3)
        raise RuntimeError('Cancelled public transfer')
    p.service.transfer.fetch = fetch
    try:
        p.start(); assert entered.wait(3)
        settings = ClientSettings.load(tmp_path); settings.paused = True; settings.save(tmp_path); p.update()
        assert finish(p)['state'] == 'stopped' and p.cancel.is_set()
        assert not (tmp_path/'own-swarm.json').exists() and p.plan.is_file()
    finally: p.close()


def test_private_transport_cannot_activate(tmp_path, monkeypatch):
    p, *_ = fixture(tmp_path, monkeypatch)
    p.knowledge.node.peers = lambda: [{'node_id': module.catalog.COORDINATOR,
        'capabilities': {'own_swarm_v2': True}, 'transport': 'direct', 'host': '100.84.3.4', 'encrypted': True}]
    try:
        p.start(); assert finish(p)['state'] == 'waiting'
        assert not (tmp_path/'own-swarm.json').exists()
    finally: p.close()


def test_nonfinite_validation_cannot_activate_even_with_a_success_flag(tmp_path, monkeypatch):
    from contextlib import nullcontext
    p, report, *_ = fixture(tmp_path, monkeypatch)
    report['evaluation']['nll'] = float('nan')
    p.knowledge.governor = SimpleNamespace(acquire=lambda req: (nullcontext(), ''))
    monkeypatch.setattr(module.ComputeSupervisor, 'run', lambda *_a, **_kw: report)
    with pytest.raises(ValueError, match='comprobación'):
        OwnPreparation._validate(p, '/installed/engine', ClientSettings.load(tmp_path), lambda: True)
    assert not (tmp_path/'own-swarm.json').exists()


def test_preparation_ipc_rejects_foreign_origins_and_peer_selected_programs(tmp_path, monkeypatch):
    from test_chat_service import broker_fixture, call
    from src.network.chat_service import LocalChatAPI
    broker, _ = broker_fixture(monkeypatch); calls = []
    preparation = SimpleNamespace(start=lambda: calls.append(True) or {'state': 'starting'},
        stop=lambda: {'state': 'stopped'}, snapshot=lambda: {'state': 'idle'})
    api = LocalChatAPI(broker, tmp_path, own_preparation=preparation); api.start()
    try:
        for options in [{'token': 'wrong'}, {'origin': 'https://foreign.invalid'}]:
            with pytest.raises(urllib.error.HTTPError) as e: call(api, '/own-preparation/start', {}, **options)
            assert e.value.code == 403
        with pytest.raises(urllib.error.HTTPError) as e: call(api, '/own-preparation/start', {'command': '/untrusted'})
        assert e.value.code == 400 and not calls
        assert call(api, '/own-preparation/start', {})['state'] == 'starting' and calls == [True]
    finally: api.close()
