"""Finish an interrupted pilot recovery through the normal native node.

The owner API transfers public bytes through LSP. This supervisor does not stop
the node, roll back operator files, copy private banks or restore old counters.
Run after the participant service, never as a prerequisite for starting it.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from desktop.backend import NodeController
from src.network.artifact_store import ArtifactStore
from src.network.client_state import ClientSettings, NodeStateLock
from src.network.public_knowledge import KnowledgeSettings
from src.network.telemetry import atomic_json
from src.release.language_versions import LanguageVersionRegistry
from .cold_native_trial import load, read, extras
from .cold_recovery import COUNTERS, accounted_bytes, checked_file, read_counter, repair_view


FAMILIES = ('qwen3-1.7b', 'qwen3-0.6b')


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def restore_family(directory, store, registry, release_id):
    """Verify the signed chain before adding links, publishing the adapter last."""
    chain = registry.history(release_id)
    registry.verify_available(chain, store)
    for bundle in chain:
        descriptor, release = bundle['descriptor'], bundle['release']
        declared = {row['name']: (row['artifact_id'], row['bytes']) for row in descriptor['files']}
        declared['adapter.safetensors'] = (release['model_artifact_id'], descriptor['adapter_bytes'])
        repair_view(directory, store, release['release_id'], declared)
    return [bundle['release']['release_id'] for bundle in chain]


def restore_preparation_intent(directory, migration):
    """Undo only our temporary suspension, preserving a subsequent owner edit."""
    record = read(migration / 'migration.json')
    current = directory / 'language-preparation-intent.json'
    previous = migration / 'language-preparation-intent.json'
    if fingerprint(current) != record['preparation_stopped_hash']:
        return 'owner_changed'
    if not previous.exists():
        return 'no_previous_request'
    value = read(previous, 1024)
    if set(value) != {'version', 'requested'} or value['version'] != 1 or type(value['requested']) is not bool:
        raise ValueError('Previous preparation intent is invalid')
    atomic_json(current, value)
    return 'restored'


class NativeCacheRecovery:
    def __init__(self, session):
        self.session, self.directory, self.trial, self.folder = load(Path(session))
        self.checkpoint = read(self.folder / 'checkpoint.json')
        self.migration = self.folder / 'live-recovery-migration'
        self.record = read(self.migration / 'migration.json')
        if self.checkpoint['owned_hashes']:
            raise ValueError('Restore owned temporary controls before adopting a live recovery')
        if any(read_counter(self.trial, name) for name in COUNTERS):
            raise ValueError('Trial contributions need an offline counter reconciliation')
        self.controller = NodeController(self.directory)
        self.store = ArtifactStore(self.directory / 'public/language/artifacts', max_artifact_bytes=4*1024**3)
        self.path = self.folder / 'live-recovery.json'
        self.state = read(self.path) if self.path.exists() else {'version': 1, 'families': {}, 'phase': 'waiting'}
        self.validated = set()

    def save(self, **values):
        self.state.update(values, updated_at=time.time())
        atomic_json(self.path, self.state)
        return dict(self.state)

    def tick(self):
        snapshot = self.controller.snapshot()
        if NodeController(self.trial, port=0).snapshot()['running']:
            raise ValueError('The trial identity is still running')
        if (not snapshot['fresh'] or snapshot['status']['node_id'] != self.session['physical_id']):
            raise RuntimeError('Waiting for the original native participant')
        if fingerprint(self.directory / 'identity.key') != self.record['private_fingerprints']['identity.key']:
            raise ValueError('The original identity changed')
        settings = ClientSettings.load(self.directory)
        if not settings.connected or settings.paused or not KnowledgeSettings.load(self.directory).enabled:
            return self.save(phase='waiting', message='Waiting for the current owner permission')
        used = accounted_bytes([self.directory, self.trial], extras())
        if used > settings.policy.max_disk_gb*1024**3:
            return self.save(phase='waiting', message='Waiting for authorized physical storage', aggregate_managed_bytes=used)
        for name in FAMILIES:
            registry = LanguageVersionRegistry(self.directory / 'public/language', name)
            pinned = self.session['planned_inventory']['heads'][name]
            if registry.head()['release']['release_id'] != pinned:
                raise ValueError('Public head changed; preserve the newer selection for review')
            if name in self.validated:
                continue
            if name in self.state['families']:
                # A restarted supervisor rechecks real bytes, not just its marker.
                with NodeStateLock(self.directory / 'preparation-lock'):
                    restore_family(self.directory, self.store, registry, pinned)
                self.validated.add(name)
                continue
            status = self.controller.cache_repair_status()
            # The native worker begins idle after a restart, while its last
            # completed download remains on disk. Verify bytes before using it.
            if status.get('state') == 'idle':
                persisted = self.directory / 'language-cache-repair.json'
                previous = read(persisted) if persisted.exists() else {}
                if previous.get('state') == 'complete':
                    status = previous
            if (status.get('state') == 'complete' and status.get('base_name') == name
                    and status.get('release_id') == pinned):
                with NodeStateLock(self.directory / 'preparation-lock'):
                    releases = restore_family(self.directory, self.store, registry, pinned)
                self.validated.add(name)
                self.state['families'][name] = {'head': pinned, 'verified_releases': releases}
                return self.save(phase='restoring', message='Verified public family and restored model links')
            intent_path = self.directory / 'language-cache-repair-intent.json'
            intent = read(intent_path, 1024) if intent_path.exists() else {}
            if intent.get('requested'):
                if intent.get('base_name') != name or intent.get('release_id') != pinned:
                    return self.save(phase='waiting', message='Another owner cache repair has priority')
            elif self.state.get('requested_family') == name:
                return self.save(phase='waiting', message='Owner stopped this cache repair; waiting for a new request')
            else:
                self.controller.repair_language_cache(name)
            return self.save(phase='receiving', base_name=name, requested_family=name, done=status.get('done', 0),
                total=status.get('total', 0), aggregate_managed_bytes=used,
                message='Native node resumes its signed public cache through Internet LSP')
        for identifier, size in self.session['planned_inventory']['weights'].items():
            checked_file(self.store.object_path(identifier), identifier, size)
        counters = {name: read_counter(self.directory, name) for name in COUNTERS}
        if any(counters[name] < self.record['counter_before'][name] for name in COUNTERS):
            raise ValueError('A native counter regressed; do not overwrite it')
        preparation = restore_preparation_intent(self.directory, self.migration)
        result = self.save(phase='complete', recovered=True, original_knowledge_verified=True,
            physical_id=self.session['physical_id'], counters_observed=counters,
            preparation_intent=preparation, aggregate_managed_bytes=used,
            controls_rolled_back=False, trial_success=False,
            message='Original public knowledge restored without rolling back current owner controls')
        # The old offline recovery recognizes this completion marker and exits.
        atomic_json(self.folder / 'recovered.json', result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    args = parser.parse_args()
    recovery = NativeCacheRecovery(args.session)
    while True:
        try:
            result = recovery.tick()
        except (OSError, RuntimeError, ValueError) as error:
            result = recovery.save(phase='waiting', message=str(error))
        if result['phase'] == 'complete':
            print(json.dumps(result), flush=True)
            return
        time.sleep(5)


if __name__ == '__main__':
    main()
