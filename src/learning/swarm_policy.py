"""Versioned own-learning policies. Published v1 receipts keep their meaning."""
import math

from . import swarm_distributed as legacy
from .public_math import digest

IDENTIFIER = legacy.IDENTIFIER
state_id = legacy.state_id
WORKERS = legacy.WORKERS  # Only the immutable v1 compatibility path uses this.
GRADIENT_OPERATION = legacy.GRADIENT_OPERATION
CANDIDATE_OPERATION = legacy.CANDIDATE_OPERATION
NETWORK_GRADIENT_OPERATION = 'swarm.gradient.v2'
NETWORK_CANDIDATE_OPERATION = 'swarm.candidate.v2'
FOOD_GRADIENT_OPERATION = 'swarm.gradient.v3'
FOOD_CANDIDATE_OPERATION = 'swarm.candidate.v3'


def network_definition():
    return {**legacy.definition(), 'version': 2,
        'gradient_operation': NETWORK_GRADIENT_OPERATION,
        'candidate_operation': NETWORK_CANDIDATE_OPERATION,
        'minimum_physical_workers': 1, 'maximum_contributions_per_update': 8,
        'membership': 'Locally admitted coordinator; dynamic compatible workers, no baked-in identities',
        'scheduling': 'Bounded rotating windows; disjoint shards can be reassigned before acceptance',
        'scope': 'Own causal byte-language learning with dynamic membership; not open consensus or general chat'}


def network_policy_id():
    return digest(network_definition())


def definition(identifier=legacy.policy_id()):
    if not isinstance(identifier,str):raise ValueError('Unknown own training policy')
    if identifier == legacy.policy_id():
        return legacy.definition()
    if identifier == network_policy_id():
        return network_definition()
    from .own_food_catalog import program_for_policy, policy_definition
    return policy_definition(program_for_policy(identifier))


def policy_id():
    """Retain the historical default for old callers and saved settings."""
    return legacy.policy_id()


def validate_payload(payload, *, candidate=False):
    if not isinstance(payload, dict):
        raise ValueError('Own work must be a typed public payload')
    if payload.get('training_policy_id') == legacy.policy_id():
        return legacy.validate_payload(payload, candidate=candidate)
    p = definition(payload.get('training_policy_id', ''))
    fields = {'state_id', 'model_id', 'dataset_id', 'training_policy_id'}
    fields |= {'contributions'} if candidate else {'start', 'count'}
    if (set(payload) != fields or payload['dataset_id'] != p['training_dataset_id']
            or any(not isinstance(payload[k], str) or not IDENTIFIER.fullmatch(payload[k])
                   for k in ['state_id', 'model_id'])):
        raise ValueError('Own work has incompatible state, dataset or fields')
    if candidate:
        if not isinstance(payload['contributions'], list) or not 1 <= len(payload['contributions']) <= p['maximum_contributions_per_update']:
            raise ValueError('Own update exceeds its bounded contribution window')
    elif (type(payload['start']) is not int or type(payload['count']) is not int
            or not 1 <= payload['count'] <= p['maximum_documents_per_shard']
            or not 0 <= payload['start'] < p.get('training_units',4096)
            or payload['start'] + payload['count'] > p.get('training_units',4096)):
        raise ValueError('Own training document range is invalid')


def verify_contributions(records, *, model_id, training_state_id, requester_id,
                         workers=None, training_policy_id=legacy.policy_id()):
    identifier = training_policy_id
    if identifier == legacy.policy_id():
        return legacy.verify_contributions(records, model_id=model_id,
            training_state_id=training_state_id, requester_id=requester_id,
            workers=legacy.WORKERS if workers is None else workers)
    from src.network.work_protocol import ResultReceipt, WorkResult
    p = definition(identifier)
    if not isinstance(records, list) or not 1 <= len(records) <= p['maximum_contributions_per_update']:
        raise ValueError('Own update exceeds its bounded contribution window')
    jobs = set(); ranges = []; outputs = []
    for item in records:
        if not isinstance(item, dict) or set(item) != {'worker_id', 'result'}:
            raise ValueError('Own contribution record is invalid')
        worker = item['worker_id']; result = WorkResult.from_dict(item['result'])
        if (not isinstance(worker, str) or not IDENTIFIER.fullmatch(worker)
                or workers is not None and worker not in workers
                or result.job_id in jobs or result.status != 'ok' or not result.receipt
                or not ResultReceipt.from_dict(result.receipt).verify(result,
                    expected_worker_id=worker, expected_requester_id=requester_id)):
            raise ValueError('Own contribution signature, worker or job is invalid')
        output = result.output
        required = {'state_id', 'model_id', 'dataset_id', 'training_policy_id', 'start', 'count',
                    'tokens', 'loss', 'gradient_id', 'gradient_bytes'}
        if set(output) != required or output['training_policy_id'] != identifier:
            raise ValueError('Own contribution has incompatible fields or policy')
        validate_payload({k: output[k] for k in ['state_id', 'model_id', 'dataset_id', 'training_policy_id', 'start', 'count']})
        if (output['state_id'] != training_state_id or output['model_id'] != model_id
                or type(output['tokens']) is not int or not 1 <= output['tokens'] <= output['count'] * 128
                or type(output['gradient_bytes']) is not int or not 0 < output['gradient_bytes'] <= p['maximum_artifact_bytes']
                or not isinstance(output['gradient_id'], str) or not IDENTIFIER.fullmatch(output['gradient_id'])
                or type(output['loss']) not in (int, float) or not math.isfinite(output['loss']) or output['loss'] < 0):
            raise ValueError('Own gradient belongs to another state or token budget')
        jobs.add(result.job_id); ranges.append((output['start'], output['start'] + output['count'])); outputs.append(output)
    ranges.sort()
    if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:])):
        raise ValueError('Own productive shards overlap')
    return outputs


def verdict(before, after, identifier=legacy.policy_id()):
    # v2 changes participation, not the previously fixed mathematical gate.
    p=definition(identifier)
    if p['version'] < 3:return legacy.verdict(before, after)
    for value in [before,after]:
        if not isinstance(value,dict) or set(value)!={'nll','byte_accuracy','predicted_bytes','documents','retention'}:
            raise ValueError('Own food evaluation is incomplete')
        primary={k:v for k,v in value.items() if k!='retention'}
        # Reuse the historical metric type checks without changing its receipt
        # schema or gate, substituting only this program's pinned byte count.
        if (primary['predicted_bytes']!=p['evaluation_prediction_bytes']
                or primary['documents']!=p['evaluation_documents']):
            raise ValueError('Own food evaluation changed its held-out coverage')
        legacy.verdict({**primary,'predicted_bytes':57123}, {**primary,'predicted_bytes':57123})
        legacy.verdict(value['retention'],value['retention'])
    accuracy_ok=(p.get('quality_selection')=='proper-log-score-v1' or
        after['byte_accuracy']>=before['byte_accuracy'] and
        after['retention']['byte_accuracy']>=before['retention']['byte_accuracy'])
    return (before['nll']-after['nll']>=max(p['minimum_absolute_nll_gain'],before['nll']*p['minimum_relative_nll_gain'])
        and accuracy_ok and after['retention']['nll']<=before['retention']['nll'])
