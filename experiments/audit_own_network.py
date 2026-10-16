"""Audit native dynamic windows without inventing workers or resetting v1 history."""
import json
from pathlib import Path

from src.learning.swarm_policy import network_policy_id, state_id, verify_contributions, verdict
from src.network.artifact_store import ArtifactStore, digest_file
from src.network.own_network_settings import OwnNetworkSettings
from src.network.own_state import verify_transition
from src.network.work_protocol import WorkResult, ResultReceipt


def signed_output(value, worker, owner):
    result = WorkResult.from_dict(value)
    if (result.status != 'ok' or not result.receipt or not ResultReceipt.from_dict(result.receipt).verify(
            result, expected_worker_id=worker, expected_requester_id=owner)):
        raise ValueError('Invalid native physical receipt')
    return result.output


def audit(directory, protocol, laptop):
    node = Path(directory); root = node / 'own-network'
    settings = OwnNetworkSettings.load(node); ledger = json.loads((root / 'ledger.json').read_text())
    if (ledger['contract_id'] != settings.contract() or settings.initial_state_id != protocol['initial_state_id']
            or settings.initial_cursor != protocol['initial_cursor'] or ledger['pending'] is not None
            or ledger['training_policy_id'] != network_policy_id() or settings.parent_program_id != protocol['parent_program_id']):
        raise ValueError('Native program changed or still has pending work')
    store = ArtifactStore(node / 'public/own/artifacts', max_artifact_bytes=16*1024**2, max_total_bytes=1024**3)
    owner = settings.owner_id; parent = settings.initial_state_id; cursor = settings.initial_cursor
    step = protocol['initial_optimizer_step']; accepted = 0; rejections = 0; jobs = set()
    records = []; reports = {}; audits = {}; transfers = []; reassigned = 0
    for index in range(ledger['round']):
        row = json.loads((root / 'rounds' / f'{index:08d}.json').read_text())
        if (row['round'] != index or row['cursor'] != cursor or row['state_id'] != parent
                or not row['autonomous'] or row['finished_at'] < row['started_at']
                or not 1 <= len(row['assignments']) <= settings.workers_per_window):
            raise ValueError('Dynamic window changed its durable data range or parent')
        contributions = [row['contributions'][r['shard_id']] for r in row['assignments']]
        outputs = verify_contributions(contributions, model_id=row['model_id'], training_state_id=parent,
            requester_id=owner, training_policy_id=network_policy_id())
        end = cursor
        for assignment, contribution, output in zip(row['assignments'], contributions, outputs):
            key = assignment['shard_id']; worker = contribution['worker_id']; result = contribution['result']
            if (assignment['start'] != end or assignment['shard_id'] != str(end)
                    or assignment['worker_id'] != worker or any(output[k] != assignment[k] for k in ['start', 'count'])
                    or result['job_id'] in jobs or row['work'][key]['result'] != result):
                raise ValueError('Completed shard duplicates work or differs from its assignment')
            jobs.add(result['job_id']); end += assignment['count']
            if digest_file(store.object_path(output['gradient_id'])) != (output['gradient_id'], output['gradient_bytes']):
                raise ValueError('Native gradient content differs from its signed receipt')
            route = row['work'][key]['route']
            if worker != owner and (not route['encrypted'] or not route['public_internet']):
                raise ValueError('Remote gradient used a non-public route')
        for event in row['reassignments']:
            if event['from'] == event['to'] or event['shard_id'] not in row['contributions']:
                raise ValueError('Invalid native reassignment evidence')
        reassigned += len(row['reassignments'])
        result = row['candidate']['result']; output = signed_output(result, owner, owner)
        if (output['training_policy_id'] != network_policy_id() or output['state_id'] != parent
                or state_id(output['model_id'], output['optimizer_id'], step) != parent
                or output['previous_optimizer_step'] != step or output['candidate_optimizer_step'] != step + 1
                or output['accepted'] != row['accepted'] or row['accepted'] != verdict(output['before'], output['after'])
                or output['prediction_bytes'] != sum(v['tokens'] for v in outputs)):
            raise ValueError('Native candidate contradicts its policy, parent or optimizer continuity')
        path = store.object_path(output['evaluation_id']); raw = path.read_bytes()
        if digest_file(path) != (output['evaluation_id'], output['evaluation_bytes']):
            raise ValueError('Native evaluation differs from its candidate receipt')
        evaluation = json.loads(raw); reports[output['evaluation_id']] = raw.decode()
        if evaluation['contributions'] != contributions or any(evaluation[k] != output[k]
                for k in ['accepted', 'before', 'after', 'state_id', 'model_id', 'optimizer_id', 'training_policy_id']):
            raise ValueError('Native evaluator used another contribution set')
        if row['accepted']:
            transition = verify_transition(result, raw); parent = transition['state_id']; step += 1
            accepted += 1; rejections = 0
            adoption = signed_output(row['adoptions'][owner], owner, owner)
            if adoption['state_id'] != parent: raise ValueError('Local adoption selected another state')
            local_audit = json.loads((node / 'public/own/audits' / (parent + '.json')).read_text())
            remote_audit = laptop['audits'].get(parent)
            for checked in [local_audit, remote_audit]:
                if (not checked or not checked['success'] or checked['state_id'] != parent
                        or checked['training_policy_id'] != network_policy_id()
                        or checked['model_id'] != transition['model_id'] or checked['optimizer_id'] != transition['optimizer_id']
                        or checked['optimizer_step'] != step or checked['optimizer_states'] != 175
                        or checked['cuda_initialized'] or not checked['gradient_replay_verified']
                        or not checked['saved_state_unchanged'] or checked['new_gradient_computed']):
                    raise ValueError('An accepted state lacks independent physical CPU replay')
            if adoption['audit'] != local_audit: raise ValueError('Local replay report differs from its signed adoption')
            audits[parent] = {'coordinator': local_audit, 'returned_worker': remote_audit}
            remote_adoption = laptop['adoptions'][parent]
            for transfer in remote_adoption['transfers']:
                if not transfer['route']['encrypted'] or not transfer['route']['public_internet']:
                    raise ValueError('Returned worker acquired learned state over a non-public route')
                transfers.append(transfer)
            for name in ['candidate_model', 'candidate_optimizer']:
                if digest_file(store.object_path(output[name + '_id'])) != (output[name + '_id'], output[name + '_bytes']):
                    raise ValueError('Accepted learned state was lost or changed')
        else:
            rejections += 1
            if row['adoptions']: raise ValueError('Rejected candidate was adopted')
        if row['resulting_state_id'] != parent: raise ValueError('Window recorded another resulting state')
        cursor = end; records.append(row)
    selected = json.loads((node / 'own-swarm.json').read_text())
    if (ledger['cursor'] != cursor or ledger['accepted'] != accepted or ledger['rejections'] != rejections
            or ledger['state_id'] != parent or state_id(selected['model_id'], selected['optimizer_id'], selected['step']) != parent
            or selected['step'] != step or laptop['own_swarm']['state_id'] != parent):
        raise ValueError('Completed durable progress or physical replicas disagree')
    for name in ['model_id', 'optimizer_id']:
        identifier = selected[name]; remote = laptop['selected_objects'][identifier]
        if remote['sha256'] != identifier or digest_file(store.object_path(identifier)) != (identifier, remote['bytes']):
            raise ValueError('Returned physical replica did not retain the exact learned content')
    return {'verified_cycles': len(records), 'accepted': accepted, 'rejected': len(records) - accepted,
        'consecutive_rejections': rejections, 'cursor': cursor, 'state_id': parent, 'optimizer_step': step,
        'status': ledger['status'], 'training_policy_id': network_policy_id(), 'coordinator_contract_id': settings.contract(),
        'reassignments': reassigned, 'records': records, 'evaluation_reports_raw': reports,
        'independent_cpu_audits': audits, 'public_replica_transfers': transfers,
        'distinct_workers_per_window': [len({v['worker_id'] for v in r['contributions'].values()}) for r in records],
        'remote_audit_evidence': 'Public replay reports collected over administrative SSH; remote adoption receipts verified live but not retained by v2',
        'larger_physical_network_tested': False, 'linear_speedup_proved': False,
        'clean_app_own_bootstrap_proved': False, 'leader_election_or_open_consensus_proved': False}
