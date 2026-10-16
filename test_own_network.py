"""Dynamic membership tests use synthetic identities, never count them as PCs."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from src.learning.public_math import canonical
from src.learning import swarm_distributed as legacy
from src.learning.swarm_policy import network_policy_id, definition, state_id, verify_contributions
from src.network.lsp import LSPIdentity
from src.network.work_protocol import WorkResult, ResultReceipt
from src.network.artifact_store import ArtifactStore
from src.network.own_network import OwnNetworkCoordinator
from src.network.own_network_settings import OwnNetworkSettings
from src.network.own_network_plan import choose_window, allocate
from src.network.own_replication import HEAD_OPERATION, ADOPT_OPERATION
from src.network.own_state import verify_transition
from src.network.own_swarm import OwnSwarmSettings


def sign(output, worker, requester, job):
    result = WorkResult(job_id=hashlib.sha256(str(job).encode()).hexdigest(), status='ok', output=output)
    return replace(result, receipt=ResultReceipt.create(result, worker, requester.node_id_hex).to_dict()).to_dict()


def setup(tmp_path, n):
    identities = [LSPIdentity.generate() for _ in range(n)]; owner = identities[0]
    workers = {w.node_id_hex: w for w in identities}; policy = network_policy_id()
    initial = state_id('a' * 64, 'b' * 64, 1024)
    selected = {'state_id': initial, 'model_id': 'a' * 64, 'ready': True,
        'training_policy_id': policy, 'requesters': [owner.node_id_hex]}
    live = set(workers); failed = set(); heads = {w: deepcopy(selected) for w in workers}
    node = SimpleNamespace(identity=owner, peers=lambda: [{'node_id': w, 'capabilities': {'own_swarm_v2': True}}
        for w in live if w != owner.node_id_hex])
    service = SimpleNamespace(directory=tmp_path, root=tmp_path / 'public/own', ready=True,
        engine=SimpleNamespace(version=2), knowledge=SimpleNamespace(node=node, allowed=True),
        snapshot=lambda: deepcopy(selected), store=ArtifactStore(tmp_path / 'public/own/artifacts'))
    settings = OwnNetworkSettings(enabled=True, follow_enabled=True, owner_id=owner.node_id_hex,
        initial_state_id=initial, maximum_rounds=1)
    settings.save(tmp_path)
    class Work:
        def __init__(self): self.jobs = {}; self.started = []; self.cancelled = []
        def start(self, value):
            self.started.append(deepcopy(value)); payload = value['payload']; worker = value['worker_id']
            if worker in failed:
                self.jobs[value['request_id']] = {'state': 'failed'}; return
            if value['operation'] == definition(policy)['gradient_operation']:
                output = {**payload, 'tokens': payload['count'] * 128, 'loss': 2.,
                    'gradient_id': hashlib.sha256(value['request_id'].encode()).hexdigest(), 'gradient_bytes': 100}
            else:
                before = {'nll': 2., 'byte_accuracy': .4, 'predicted_bytes': 57123, 'documents': 128}
                output = {k: payload[k] for k in ['training_policy_id', 'state_id', 'model_id']}
                output.update(accepted=True, before=before, after={**before, 'nll': 1.99}, optimizer_id='b' * 64,
                    previous_optimizer_step=1024, candidate_optimizer_step=1025,
                    prediction_bytes=sum(r['result']['output']['tokens'] for r in payload['contributions']),
                    candidate_model_id='c' * 64, candidate_model_bytes=100,
                    candidate_optimizer_id='d' * 64, candidate_optimizer_bytes=100)
                evaluation = {**output, 'candidate_state_id': state_id('c' * 64, 'd' * 64, 1025),
                    'contributions': payload['contributions']}
                path = tmp_path / 'evaluation'; path.write_bytes(canonical(evaluation))
                artifact = service.store.import_file(path, kind='evaluation')
                output.update(evaluation_id=artifact.artifact_id, evaluation_bytes=artifact.size)
            result = sign(output, workers[worker], owner, value['request_id'])
            self.jobs[value['request_id']] = {'state': 'complete', 'result': result, 'seconds': 1.}
        def status(self, key): return self.jobs[key]
        def cancel(self, key): self.cancelled.append(key)
    work = Work(); c = OwnNetworkCoordinator(service, work); c.settings = settings
    c.ledger = {'version': 2, 'contract_id': settings.contract(), 'training_policy_id': policy,
        'initial_state_id': initial, 'state_id': initial, 'parent_program_id': '', 'round': 0, 'cursor': 0,
        'accepted': 0, 'rejections': 0, 'workers': {}, 'ready_workers': 0, 'replicas': 1,
        'pending': None, 'status': 'ready', 'error': None, 'retry_at': 0}
    c._save()
    fault = {'after_local_adoption': False}
    def rpc(worker, operation, payload, contract):
        c._check(contract)
        if worker not in live: raise ConnectionError('Synthetic worker left')
        if operation == HEAD_OPERATION:
            return {'output': deepcopy(selected if worker == owner.node_id_hex else heads[worker])}
        assert operation == ADOPT_OPERATION
        result = payload['candidate_result']; output = result['output']
        transition = verify_transition(result, service.store.object_path(output['evaluation_id']).read_bytes())
        target = selected if worker == owner.node_id_hex else heads[worker]
        target.update(state_id=transition['state_id'], model_id=transition['model_id'])
        if worker == owner.node_id_hex:
            path = service.root / 'history' / (transition['state_id'] + '.json')
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(transition))
            if fault['after_local_adoption']:
                fault['after_local_adoption'] = False; raise InterruptedError('Synthetic restart boundary')
        return {'output': {'state_id': transition['state_id']}}
    c._rpc = rpc
    return c, work, workers, live, failed, selected, fault, rpc


@pytest.mark.parametrize('n', [1, 2, 3, 8])
def test_arbitrary_n_signed_contributions_survive_candidate_and_catchup(tmp_path, n):
    c, work, workers, live, failed, selected, fault, rpc = setup(tmp_path, n)
    c._run(c.settings.contract())
    assert c.ledger['error'] is None and c.ledger['round'] == 1 and c.ledger['accepted'] == 1
    assert c.ledger['cursor'] == 64 * n and c.ledger['pending'] is None
    record = json.loads((c.root / 'rounds/00000000.json').read_text())
    outputs = verify_contributions(list(record['contributions'].values()), model_id='a' * 64,
        training_state_id=c.settings.initial_state_id, requester_id=c.settings.owner_id,
        training_policy_id=network_policy_id())
    assert len(outputs) == n and len({r['worker_id'] for r in record['contributions'].values()}) == n
    assert record['replicas_observed'] == min(n, 3)
    assert not any(w in legacy.WORKERS for w in workers)


def test_disconnected_worker_shard_moves_without_losing_completed_work(tmp_path):
    c, work, workers, live, failed, *_ = setup(tmp_path, 3)
    lost = next(w for w in workers if w != c.settings.owner_id); failed.add(lost)
    c._run(c.settings.contract())
    assert c.ledger['round'] == 1 and c.ledger['cursor'] == 192
    record = json.loads((c.root / 'rounds/00000000.json').read_text())
    assert record['reassignments'][0]['from'] == lost
    assert len(record['contributions']) == 3 and len({v['worker_id'] for v in record['contributions'].values()}) == 2
    successful = [v for v in work.started if v['operation'].endswith('gradient.v2') and v['worker_id'] != lost]
    assert len(successful) == 3  # Completed shards were not retrained.


def test_restart_after_selection_keeps_exact_candidate_and_progress(tmp_path):
    c, work, workers, live, failed, selected, fault, rpc = setup(tmp_path, 3)
    fault['after_local_adoption'] = True; c._run(c.settings.contract())
    assert c.ledger['status'] == 'paused' and c.ledger['cursor'] == 0
    candidate = deepcopy(c.ledger['pending']['candidate']); started = len(work.started)
    resumed = OwnNetworkCoordinator(c.service, work); resumed.settings = c.settings
    # The RPC boundary uses c's permission check, as a real native RPC would.
    resumed._rpc = rpc; resumed._run(resumed.settings.contract())
    assert resumed.ledger['round'] == 1 and resumed.ledger['cursor'] == 192
    assert len(work.started) == started
    record = json.loads((resumed.root / 'rounds/00000000.json').read_text())
    assert record['candidate'] == candidate


def test_late_members_catch_up_after_curriculum_stop_without_new_training(tmp_path):
    c, work, workers, live, failed, selected, fault, rpc = setup(tmp_path, 8)
    live.clear(); live.add(c.settings.owner_id)
    c._run(c.settings.contract()); started = len(work.started)
    assert c.ledger['round'] == 1 and c.ledger['cursor'] == 64
    live.update(workers)
    for _ in range(4): c._maintain(c.settings.contract())
    # Reading every signed head below proves rotating replication reached all
    # late identities, without inventing another training completion.
    for worker in workers:
        assert rpc(worker, HEAD_OPERATION, {}, c.settings.contract())['output']['state_id'] == selected['state_id']
    assert len(work.started) == started and c.ledger['cursor'] == 64


def test_large_membership_rotates_bounded_windows_and_uses_observed_capacity():
    workers = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(64)]; history = {}; seen = set()
    for round_number in range(8):
        window = choose_window(workers, history)
        assert len(window) == 8 and not seen.intersection(window)
        seen.update(window)
        for worker in window: history[worker] = {'last_round': round_number}
    assert seen == set(workers)
    history[workers[0]].update(samples=2, documents_per_second=4.)
    history[workers[1]].update(samples=2, documents_per_second=1.)
    plan = allocate(workers[:3], {}, history, start=0)
    assert [r['count'] for r in plan] == [256, 64, 64]
    tail = allocate(workers[:8], {}, history, start=4090)
    assert tail == [{'shard_id': '4090', 'worker_id': workers[0], 'start': 4090, 'count': 6}]


def test_admission_and_legacy_policy_cannot_be_silently_reinterpreted(tmp_path):
    own = LSPIdentity.generate().node_id_hex
    OwnSwarmSettings(version=2, requesters=[own]).validate()
    with pytest.raises(ValueError): OwnSwarmSettings(version=1, requesters=[own]).validate()
    assert legacy.policy_id() == '57052db9cad9b02f9b5bad64ce37f6e71c7ad7f597ee341320af266af091cb8c'
    c, work, *_ = setup(tmp_path, 3)
    c._run(c.settings.contract()); record = json.loads((c.root / 'rounds/00000000.json').read_text())
    with pytest.raises(ValueError):
        legacy.verify_contributions(list(record['contributions'].values()), model_id='a' * 64,
            training_state_id=c.settings.initial_state_id, requester_id=c.settings.owner_id)


def test_network_rejects_replayed_overlapping_foreign_state_and_policy_receipts(tmp_path):
    c, work, workers, *_ = setup(tmp_path, 3)
    c._run(c.settings.contract()); record = json.loads((c.root / 'rounds/00000000.json').read_text())
    rows = list(record['contributions'].values())
    options = dict(model_id='a' * 64, training_state_id=c.settings.initial_state_id,
        requester_id=c.settings.owner_id, training_policy_id=network_policy_id())
    with pytest.raises(ValueError): verify_contributions([rows[0], rows[0]], **options)
    with pytest.raises(ValueError): verify_contributions(rows, **{**options, 'training_state_id': 'f' * 64})
    with pytest.raises(ValueError): verify_contributions(rows, **{**options, 'requester_id': 'f' * 64})
    output = deepcopy(rows[1]['result']['output']); output['start'] = rows[0]['result']['output']['start']
    overlap = {'worker_id': rows[1]['worker_id'], 'result': sign(output,
        workers[rows[1]['worker_id']], workers[c.settings.owner_id], 'overlap')}
    with pytest.raises(ValueError, match='overlap'): verify_contributions([rows[0], overlap], **options)
    output['start'] = 128; output['training_policy_id'] = legacy.policy_id()
    mixed = {'worker_id': rows[1]['worker_id'], 'result': sign(output,
        workers[rows[1]['worker_id']], workers[c.settings.owner_id], 'mixed')}
    with pytest.raises(ValueError, match='policy'): verify_contributions([rows[0], mixed], **options)
