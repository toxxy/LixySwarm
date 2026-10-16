"""Native dynamic-membership own learning, with bounded work and durable retries.

Membership is discovered, never a source-code list. A locally admitted owner
coordinates one lineage; this is not leader election or open-network consensus.
"""
from copy import deepcopy
import json
import threading
import time
import uuid

from src.learning.swarm_policy import network_policy_id, definition, verify_contributions, verdict
from .own_learning import OwnLearningCoordinator
from .own_network_settings import OwnNetworkSettings
from .own_network_plan import choose_window, allocate, replacement
from .own_replication import HEAD_OPERATION, ADOPT_OPERATION
from .own_state import verify_transition
from .telemetry import atomic_json


class OwnNetworkCoordinator(OwnLearningCoordinator):
    def __init__(self, service, local_work):
        self.service = service; self.local_work = local_work; self.knowledge = service.knowledge
        self.settings = OwnNetworkSettings(); self.error = None
        self.root = service.directory / 'own-network'; self.path = self.root / 'ledger.json'
        self.lock = threading.RLock(); self.changed = threading.Event(); self.cancel = threading.Event()
        self.thread = None; self.closed = False; self.ledger = None; self.maintenance_at = 0
        if self.path.exists():
            if self.path.stat().st_size > 4 * 1024**2: raise ValueError('Own-network ledger exceeds its bound')
            value = json.loads(self.path.read_text())
            if (value.get('version') != 2 or value.get('training_policy_id') != network_policy_id()
                    or type(value.get('round')) is not int or not 0 <= value['round'] <= 1000000
                    or type(value.get('cursor')) is not int or not 0 <= value['cursor'] <= 4096
                    or not isinstance(value.get('workers'), dict) or len(value['workers']) > 4096):
                raise ValueError('Invalid own-network ledger; preserve it for inspection')
            self.ledger = value

    def snapshot(self):
        with self.lock:
            value = {k: self.ledger.get(k) for k in ['status', 'round', 'cursor', 'accepted', 'rejections', 'state_id', 'error', 'retry_at', 'replicas', 'ready_workers']} if self.ledger else {
                'status': 'idle', 'round': 0, 'cursor': self.settings.initial_cursor, 'accepted': 0, 'rejections': 0}
            return {**value, 'enabled': self.settings.enabled, 'follow_enabled': self.settings.follow_enabled,
                'training_policy_id': network_policy_id(), 'program_error': self.error,
                'known_workers': len(self.ledger['workers']) if self.ledger else 0,
                'window_limit': self.settings.workers_per_window,
                'pending': self.ledger.get('pending') is not None if self.ledger else False}

    def _allowed(self, contract):
        return (not self.closed and self.settings.enabled and self.settings.contract() == contract
            and self.service.ready and self.knowledge.allowed and self.service.engine.version == 2
            and self.knowledge.node.identity.node_id_hex == self.settings.owner_id)

    def update(self):
        try:
            self.settings = OwnNetworkSettings.load(self.service.directory)
            if self.ledger and self.ledger['contract_id'] != self.settings.contract():
                raise ValueError('Own-network program changed; durable progress was preserved')
            self.error = None
        except (OSError, ValueError, TypeError) as error:
            self.error = str(error); self.cancel.set(); return
        contract = self.settings.contract()
        if (self.settings.follow_enabled and self.service.engine.version == 2 and self.service.ready
                and self.knowledge.allowed and self.settings.owner_id != self.knowledge.node.identity.node_id_hex):
            # Every admitted worker maintains a route to its selected lineage,
            # even when the ordinary four outbound peer slots are already full.
            self.cancel.set()
            if self.thread is not None and self.thread.is_alive(): return
            if time.time() >= self.maintenance_at:
                self.maintenance_at = time.time() + 30
                self.thread = threading.Thread(target=self._connect_owner, args=(self.settings.owner_id,),
                    name='lixy-own-network-follow', daemon=True)
                self.thread.start()
            return
        if not self._allowed(contract): self.cancel.set(); return
        if self.thread is not None and self.thread.is_alive(): return
        if self.ledger is None:
            if self.service.snapshot()['state_id'] != self.settings.initial_state_id:
                self.error = 'Own-network initial state is not the selected verified state'; return
            self.ledger = {'version': 2, 'contract_id': contract, 'training_policy_id': network_policy_id(),
                'parent_program_id': self.settings.parent_program_id,
                'initial_state_id': self.settings.initial_state_id, 'state_id': self.settings.initial_state_id,
                'round': 0, 'cursor': self.settings.initial_cursor, 'accepted': 0, 'rejections': 0,
                'workers': {}, 'ready_workers': 0, 'pending': None, 'status': 'ready',
                'error': None, 'retry_at': 0, 'replicas': 1}
            self._save()
        if self.ledger['pending'] is None:
            status = ('complete' if self.settings.maximum_rounds and self.ledger['round'] >= self.settings.maximum_rounds
                else 'awaiting_new_curriculum' if self.ledger['cursor'] >= 4096
                else 'no_improvement' if self.ledger['rejections'] >= self.settings.stop_after_rejections else None)
            if status:
                if self.ledger['status'] != status: self._save(status=status)
                # New or returning replicas still catch up when training has
                # exhausted its current food or reached a quality stop.
                if time.time() >= self.maintenance_at:
                    self.maintenance_at = time.time() + 30; self.cancel = threading.Event()
                    self.thread = threading.Thread(target=self._maintain, args=(contract,),
                        name='lixy-own-network-replicas', daemon=True)
                    self.thread.start()
                return
        if time.time() < self.ledger.get('retry_at', 0): return
        self.cancel = threading.Event()
        self.thread = threading.Thread(target=self._run, args=(contract,), name='lixy-own-network', daemon=True)
        self.thread.start()

    def _connect_owner(self, owner):
        try:
            if owner not in self.knowledge.node.connected_node_ids:
                self.knowledge.node.connect_node(owner, timeout=12)
        except (OSError, ValueError, RuntimeError):
            pass

    def _discover(self, contract):
        """Signed heads decide readiness; announcements only nominate candidates."""
        own = self.settings.owner_id; heads = {own: self.service.snapshot()}
        peers = [row['node_id'] for row in self.knowledge.node.peers()
                 if row.get('capabilities', {}).get('own_swarm_v2') and row['node_id'] != own]
        history = self.ledger['workers']
        # Probe bounded pages and rotate them; joining a huge colony must not
        # spawn N threads or make every update wait for N round trips.
        peers = sorted(set(peers), key=lambda w: (history.get(w, {}).get('last_probe', 0), w))[:16]
        for worker in peers:
            self._check(contract)
            try:
                head = self._rpc(worker, HEAD_OPERATION, {}, contract)['output']
                if (head.get('ready') and head.get('training_policy_id') == network_policy_id()
                        and own in head.get('requesters', [])):
                    heads[worker] = head
            except InterruptedError: raise
            except Exception: pass
            if worker not in history and len(history) >= 4096: continue
            history.setdefault(worker, {})['last_probe'] = time.time()
        self._save(workers=history, ready_workers=len(heads))
        return heads

    def _sync_peer(self, worker, contract):
        head = self._rpc(worker, HEAD_OPERATION, {}, contract)['output']
        if not head.get('ready') or head.get('training_policy_id') != network_policy_id():
            raise RuntimeError('Own worker is not ready for this policy')
        following = self.service.snapshot()['state_id']; chain = []
        while following != head['state_id']:
            if len(chain) >= 4096: raise ValueError('Own lineage exceeds bounded recovery history')
            path = self.service.root / 'history' / (following + '.json')
            if not path.is_file(): raise ValueError('Own worker follows another or unavailable lineage')
            transition = json.loads(path.read_text()); chain.append(transition['candidate_result'])
            following = transition['parent_state_id']
        for result in reversed(chain):
            self._rpc(worker, ADOPT_OPERATION, {'candidate_result': result}, contract)
        return head

    def _maintain(self, contract):
        try:
            heads = self._discover(contract); own = self.settings.owner_id; replicas = 1
            for worker in self._replication_targets(heads):
                try:
                    self._sync_peer(worker, contract)
                    head = self._rpc(worker, HEAD_OPERATION, {}, contract)['output']
                    replicas += int(head['state_id'] == self.service.snapshot()['state_id'])
                except InterruptedError: raise
                except Exception: pass
                self.ledger['workers'].setdefault(worker, {})['last_replication'] = time.time()
            self._save(replicas=replicas)
        except (InterruptedError, OSError, ValueError, RuntimeError):
            pass

    def _replication_targets(self, heads):
        return sorted((w for w in heads if w != self.settings.owner_id),
            key=lambda w: (self.ledger['workers'].get(w, {}).get('last_replication', 0), w))[:2]

    def _validate_pending(self, pending):
        if (pending['round'] != self.ledger['round'] or pending['cursor'] != self.ledger['cursor']
                or pending['state_id'] != self.ledger['state_id'] or not 1 <= len(pending['assignments']) <= 8):
            raise ValueError('Own pending window contradicts durable progress')
        cursor = pending['cursor']; keys = set()
        for row in pending['assignments']:
            if row['start'] != cursor or row['shard_id'] != str(cursor) or not 1 <= row['count'] <= 256:
                raise ValueError('Own pending ranges are not disjoint and contiguous')
            cursor += row['count']; keys.add(row['shard_id'])
        if cursor > 4096 or not set(pending['contributions']) <= keys:
            raise ValueError('Own pending contribution coverage is invalid')
        for key, record in pending['contributions'].items():
            row = next(r for r in pending['assignments'] if r['shard_id'] == key)
            output = verify_contributions([record], model_id=pending['model_id'], training_state_id=pending['state_id'],
                requester_id=self.settings.owner_id, training_policy_id=network_policy_id())[0]
            if any(output[k] != row[k] for k in ['start', 'count']) or record['worker_id'] != row['worker_id']:
                raise ValueError('Own contribution differs from its durable assignment')

    def _wait_jobs(self, assignments, contract, pending, ready):
        active = {}; failed = set(); synced = {self.settings.owner_id}
        try:
            while len(pending['contributions']) < len(assignments):
                self._check(contract)
                for row in assignments:
                    key = row['shard_id']; worker = row['worker_id']
                    if key in pending['contributions'] or key in active: continue
                    if worker not in ready or worker in failed:
                        target = replacement(ready, assignments, pending['contributions'], failed)
                        if target is None: continue
                        pending['reassignments'].append({'shard_id': key, 'from': worker, 'to': target, 'at': time.time()})
                        row['worker_id'] = worker = target
                        self._save(pending=deepcopy(pending))
                    try:
                        if worker not in synced: self._sync_peer(worker, contract); synced.add(worker)
                        request = uuid.uuid4().hex
                        payload = {k: pending[k] for k in ['state_id', 'model_id']}
                        payload.update(dataset_id=definition(network_policy_id())['training_dataset_id'],
                            training_policy_id=network_policy_id(), start=row['start'], count=row['count'])
                        self.local_work.start({'request_id': request, 'worker_id': worker,
                            'operation': definition(network_policy_id())['gradient_operation'], 'payload': payload})
                        active[key] = request
                    except InterruptedError: raise
                    except Exception: failed.add(worker)
                for key, request in list(active.items()):
                    job = self.local_work.status(request)
                    row = next(r for r in assignments if r['shard_id'] == key)
                    if job['state'] == 'complete':
                        pending['contributions'][key] = {'worker_id': row['worker_id'], 'result': job['result']}
                        pending['work'][key] = job
                        self._validate_pending(pending)
                        self._save(pending=deepcopy(pending)); active.pop(key)
                    elif job['state'] != 'running':
                        failed.add(row['worker_id']); active.pop(key)
                if not active and len(pending['contributions']) < len(assignments):
                    remaining = [r for r in assignments if r['shard_id'] not in pending['contributions']]
                    if all(r['worker_id'] in failed or r['worker_id'] not in ready for r in remaining) and replacement(
                            ready, assignments, pending['contributions'], failed) is None:
                        raise RuntimeError('No ready own worker can complete the preserved window')
                if active: self.cancel.wait(.2)
        finally:
            for request in active.values(): self.local_work.cancel(request)

    def _candidate(self, pending, contract):
        self._check(contract); request = uuid.uuid4().hex
        payload = {k: pending[k] for k in ['state_id', 'model_id']}
        payload.update(dataset_id=definition(network_policy_id())['training_dataset_id'], training_policy_id=network_policy_id(),
            contributions=[pending['contributions'][r['shard_id']] for r in pending['assignments']])
        self.local_work.start({'request_id': request, 'worker_id': self.settings.owner_id,
            'operation': definition(network_policy_id())['candidate_operation'], 'payload': payload})
        try:
            while True:
                self._check(contract); row = self.local_work.status(request)
                if row['state'] == 'complete': return row
                if row['state'] != 'running': raise RuntimeError('Own network candidate failed: ' + row['state'])
                self.cancel.wait(.2)
        finally: self.local_work.cancel(request)

    def _run(self, contract):
        pending = None
        try:
            self._check(contract); heads = self._discover(contract)
            pending = deepcopy(self.ledger['pending'])
            if pending is None:
                selected = self.service.snapshot()
                if selected['state_id'] != self.ledger['state_id']:
                    raise ValueError('Own selected state moved outside durable network progress')
                window = choose_window(list(heads), self.ledger['workers'], limit=self.settings.workers_per_window)
                pending = {'round': self.ledger['round'], 'cursor': self.ledger['cursor'],
                    'state_id': selected['state_id'], 'model_id': selected['model_id'],
                    'assignments': allocate(window, heads, self.ledger['workers'], start=self.ledger['cursor'], base_count=self.settings.documents_per_worker),
                    'contributions': {}, 'work': {}, 'reassignments': [], 'candidate': None,
                    'adoptions': {}, 'attempts': 0, 'started_at': time.time()}
                self._save(pending=deepcopy(pending))
            self._validate_pending(pending)
            pending['attempts'] += 1
            self._save(pending=deepcopy(pending), status='gradients', error=None)
            if pending['candidate'] is None:
                if self.service.snapshot()['state_id'] != pending['state_id']:
                    raise ValueError('Own pending gradients belong to another selected state')
                self._wait_jobs(pending['assignments'], contract, pending, heads)
                self._validate_pending(pending); self._save(status='evaluating')
                pending['candidate'] = self._candidate(pending, contract); self._save(pending=deepcopy(pending))
            result = pending['candidate']['result']; output = result['output']
            if (output['state_id'] != pending['state_id'] or output['model_id'] != pending['model_id']
                    or output['training_policy_id'] != network_policy_id()
                    or output['accepted'] != verdict(output['before'], output['after'])):
                raise ValueError('Own candidate contradicts its pending state or quality policy')
            selected = pending['state_id']; replicas = self.ledger.get('replicas', 1)
            if output['accepted']:
                replicas = 1
                transition = verify_transition(result, self.service.store.object_path(output['evaluation_id']).read_bytes())
                selected = transition['state_id']; self._save(status='replicating')
                # Local replay/selection is mandatory. Remote catch-up is retried
                # before a worker's next contribution, not an all-colony barrier.
                own = self.settings.owner_id
                adopted = self._rpc(own, ADOPT_OPERATION, {'candidate_result': result}, contract)
                if adopted['output']['state_id'] != selected: raise ValueError('Own local replica selected another state')
                pending['adoptions'][own] = adopted; self._save(pending=deepcopy(pending))
                for worker in self._replication_targets(heads):
                    try:
                        self._sync_peer(worker, contract)
                        head = self._rpc(worker, HEAD_OPERATION, {}, contract)['output']
                        if head['state_id'] == selected:
                            pending['adoptions'][worker] = {'state_id': selected, 'head': head}; replicas += 1
                            self._save(pending=deepcopy(pending))
                    except InterruptedError: raise
                    except Exception: pass
                    self.ledger['workers'].setdefault(worker, {})['last_replication'] = time.time()
            elif self.service.snapshot()['state_id'] != selected:
                raise ValueError('Rejected own attempt changed selected parameters')
            self._check(contract)
            finished = {**pending, 'accepted': output['accepted'], 'resulting_state_id': selected,
                'finished_at': time.time(), 'autonomous': True, 'replicas_observed': replicas}
            atomic_json(self.root / 'rounds' / f"{pending['round']:08d}.json", finished)
            history = deepcopy(self.ledger['workers'])
            for row in pending['assignments']:
                worker = row['worker_id']; job = pending['work'][row['shard_id']]
                profile = history.setdefault(worker, {})
                seconds = job.get('seconds', 0)
                if type(seconds) in (float, int) and 0 < seconds < 7200:
                    measured = row['count'] / seconds
                    previous = profile.get('documents_per_second', measured)
                    profile.update(documents_per_second=.5 * previous + .5 * measured,
                        samples=min(1000000, profile.get('samples', 0) + 1))
                profile['last_round'] = pending['round']
            self._save(round=self.ledger['round'] + 1, cursor=pending['cursor'] + sum(r['count'] for r in pending['assignments']),
                accepted=self.ledger['accepted'] + int(output['accepted']), rejections=0 if output['accepted'] else self.ledger['rejections'] + 1,
                state_id=selected, pending=None, workers=history, replicas=replicas, status='ready', error=None, retry_at=0)
        except InterruptedError:
            if self.ledger: self._save(status='paused')
        except Exception as error:
            if self.ledger: self._save(status='waiting', error=type(error).__name__ + ': ' + str(error)[:160],
                retry_at=time.time() + min(300, 15 * max(1, (pending or {}).get('attempts', 1))))
