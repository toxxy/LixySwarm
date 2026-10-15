"""Live recovery must survive changed controls without rolling back owner state."""
import json
import time

import pytest

from experiments import recover_native_cache as recovery
from src.network.client_state import ClientSettings, NodeStateLock
from src.network.language_cache_repair import LanguageCacheRepair
from src.network.public_knowledge import KnowledgeSettings
from src.network.telemetry import atomic_json
from src.release.language_versions import LanguageVersionRegistry
from test_language_cache_repair import repair_fixture, finish


def fixture(monkeypatch, tmp_path):
    d, model, store, node, knowledge, families, failures, calls = repair_fixture(monkeypatch, tmp_path)
    folder = tmp_path/'case'; trial = folder/'node'; trial.mkdir(parents=True)
    migration = folder/'live-recovery-migration'; migration.mkdir()
    (d/'identity.key').write_bytes(b'original identity; never replace')
    (d/'personal.enc').write_bytes(b'current private memory')
    for name in recovery.COUNTERS:
        atomic_json(d/name, {'completed': 47})
    atomic_json(d/'language-preparation-intent.json', {'version': 1, 'requested': False})
    atomic_json(migration/'language-preparation-intent.json', {'version': 1, 'requested': True})
    atomic_json(migration/'migration.json', {
        'preparation_stopped_hash': recovery.fingerprint(d/'language-preparation-intent.json'),
        'private_fingerprints': {'identity.key': recovery.fingerprint(d/'identity.key')},
        'counter_before': {name: 40 for name in recovery.COUNTERS}})
    atomic_json(folder/'checkpoint.json', {'owned_hashes': {}})
    registry = LanguageVersionRegistry(families.root, 'qwen3-0.6b')
    head = registry.head()['release']['release_id']
    session = {'physical_id': node.identity.node_id_hex,
        'planned_inventory': {'heads': {'qwen3-0.6b': head},
            'weights': {model: registry.head()['descriptor']['adapter_bytes']}}}
    monkeypatch.setattr(recovery, 'load', lambda path: (session, d, trial, folder))
    monkeypatch.setattr(recovery, 'extras', lambda: [])
    monkeypatch.setattr(recovery, 'FAMILIES', ('qwen3-0.6b',))
    KnowledgeSettings(enabled=True, consented_at=time.time()).save(d)
    service = LanguageCacheRepair(node, knowledge, families, d)
    class Controller:
        def __init__(self, directory, **kw): self.directory = directory
        def snapshot(self):
            return {'running': self.directory == d, 'fresh': self.directory == d,
                'status': {'node_id': node.identity.node_id_hex}}
        def cache_repair_status(self): return service.snapshot()
        def repair_language_cache(self, name): return service.start(name)
    monkeypatch.setattr(recovery, 'NodeController', Controller)
    return recovery.NativeCacheRecovery(folder/'session.json'), service, d, folder, calls, model


def test_changed_current_policy_private_memory_and_new_counters_are_preserved(monkeypatch, tmp_path):
    helper, service, d, folder, calls, model = fixture(monkeypatch, tmp_path)
    settings = ClientSettings.load(d); settings.policy.max_bandwidth_mbps = 17; settings.save(d)
    protected = {name: (d/name).read_bytes() for name in ['controls.json', 'language-engine.json',
        'language-training.json', 'identity.key', 'personal.enc', *recovery.COUNTERS]}
    try:
        assert helper.tick()['phase'] == 'receiving'
        finish(service)
        assert helper.tick()['phase'] == 'restoring'
        # Restart from the persisted marker, reverify and finish with no transfer.
        helper = recovery.NativeCacheRecovery(folder/'session.json')
        result = helper.tick()
        assert result['phase'] == 'complete' and result['original_knowledge_verified']
        assert result['trial_success'] is False and calls == [model]
        assert {name: (d/name).read_bytes() for name in protected} == protected
        assert json.loads((d/'language-preparation-intent.json').read_text())['requested']
        assert (folder/'recovered.json').exists()
        view = d/'language-models'/helper.session['planned_inventory']['heads']['qwen3-0.6b']/'adapter.safetensors'
        assert view.stat().st_ino == helper.store.object_path(model).stat().st_ino
    finally: service.close()


def test_paused_owner_and_other_requested_family_prevent_network_calls(monkeypatch, tmp_path):
    helper, service, d, folder, calls, model = fixture(monkeypatch, tmp_path)
    try:
        settings = ClientSettings.load(d); settings.paused = True; settings.save(d)
        assert helper.tick()['phase'] == 'waiting' and service.thread is None and not calls
        settings.paused = False; settings.save(d)
        atomic_json(d/'language-cache-repair-intent.json', {'version': 1, 'requested': True,
            'base_name': 'qwen3-1.7b', 'release_id': 'f'*64})
        assert 'priority' in helper.tick()['message'] and service.thread is None and not calls
    finally: service.close()


def test_preparation_lock_prevents_publishing_incomplete_views(monkeypatch, tmp_path):
    helper, service, d, folder, calls, model = fixture(monkeypatch, tmp_path)
    try:
        helper.tick(); finish(service)
        with NodeStateLock(d/'preparation-lock'):
            with pytest.raises(RuntimeError, match='already uses'): helper.tick()
        assert not (d/'language-models').exists()
        assert helper.tick()['phase'] == 'restoring'
    finally: service.close()


def test_owner_preparation_edit_is_not_replaced_at_completion(monkeypatch, tmp_path):
    helper, service, d, folder, calls, model = fixture(monkeypatch, tmp_path)
    try:
        helper.tick(); finish(service); helper.tick()
        (d/'language-preparation-intent.json').write_text('{"requested":false,"version":1}\n')
        current = (d/'language-preparation-intent.json').read_bytes()
        assert helper.tick()['preparation_intent'] == 'owner_changed'
        assert (d/'language-preparation-intent.json').read_bytes() == current
    finally: service.close()


def test_corrupted_recovered_model_is_not_accepted_after_a_restart(monkeypatch, tmp_path):
    helper, service, d, folder, calls, model = fixture(monkeypatch, tmp_path)
    try:
        helper.tick(); finish(service); helper.tick()
        helper.store.object_path(model).write_bytes(b'corrupt recovered model')
        helper = recovery.NativeCacheRecovery(folder/'session.json')
        with pytest.raises(ValueError): helper.tick()
        assert not (folder/'recovered.json').exists()
    finally: service.close()


def test_owner_stop_is_not_automatically_reversed(monkeypatch, tmp_path):
    helper, service, d, folder, calls, model = fixture(monkeypatch, tmp_path)
    try:
        helper.tick(); finish(service)
        service._status('stopped'); service.stop()
        assert 'Owner stopped' in helper.tick()['message']
        assert calls == [model] and not (folder/'recovered.json').exists()
    finally: service.close()


def test_native_restart_after_download_uses_verified_persisted_completion(monkeypatch, tmp_path):
    helper, service, d, folder, calls, model = fixture(monkeypatch, tmp_path)
    try:
        helper.tick(); finish(service)
        monkeypatch.setattr(helper.controller, 'cache_repair_status', lambda: {'state': 'idle'})
        assert helper.tick()['phase'] == 'restoring'
        assert helper.tick()['phase'] == 'complete' and calls == [model]
    finally: service.close()
