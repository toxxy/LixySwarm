"""Immutable policy and receipt checks for productive own-swarm contributions."""
import hashlib
import math
import re

from .public_math import ARTIFACTS, canonical, digest
from .swarm_foundation_protocol import protocol_id as initialization_protocol_id

GRADIENT_OPERATION = 'swarm.gradient.v1'
CANDIDATE_OPERATION = 'swarm.candidate.v1'
WORKERS = ('c45094e5fd8c64e8f6f5be3aca703edb6527711a504fcdc747a0fba15c668255',
           'ea914d97caafef310e50385034abae9a50ba86496b0e9cbf4679e6070e4c25b0')
IDENTIFIER = re.compile('[0-9a-f]{64}\\Z')


def definition():
    return {'version': 1, 'family': 'lixyswarm-own-causal-byte-v1',
        'initialization_protocol_id': initialization_protocol_id(),
        'training_dataset_id': ARTIFACTS['train'][0], 'evaluation_dataset_id': ARTIFACTS['evaluation'][0],
        'gradient_operation': GRADIENT_OPERATION, 'candidate_operation': CANDIDATE_OPERATION,
        'maximum_documents_per_shard': 256, 'batch_size': 16, 'maximum_document_bytes': 129,
        'gradient_dtype': 'float32', 'aggregation': 'prediction-byte-weighted-mean-before-clipping',
        'learning_rate': .0002, 'weight_decay': .01, 'clip_norm': 1.,
        'minimum_relative_nll_gain': .0001, 'minimum_absolute_nll_gain': 1e-6,
        'maximum_aggregate_byte_accuracy_drop': 0., 'minimum_physical_workers': 2,
        'ram_gb': 4., 'gradient_disk_gb': .02, 'candidate_disk_gb': .06,
        'maximum_seconds': 180, 'maximum_artifact_bytes': 16*1024**2,
        'evaluation_role': 'Repeated development validation; not a fresh blinded test',
        'scope': 'Own causal byte-language learning; not general chat or math answer certification'}


def policy_id():
    return digest(definition())


def state_id(model_id, optimizer_id, step):
    if (any(not isinstance(v,str) or not IDENTIFIER.fullmatch(v) for v in [model_id,optimizer_id])
            or type(step) is not int or not 1<=step<2**53):
        raise ValueError('Own training state requires exact weights, optimizer and step')
    return digest({'initialization_protocol_id': initialization_protocol_id(),
                   'model_id':model_id,'optimizer_id':optimizer_id,'step':step})


def validate_payload(payload, *, candidate=False):
    fields={'state_id','model_id','dataset_id','training_policy_id'}
    fields |= {'contributions'} if candidate else {'start','count'}
    if (not isinstance(payload,dict) or set(payload)!=fields
            or payload['training_policy_id']!=policy_id() or payload['dataset_id']!=ARTIFACTS['train'][0]
            or any(not isinstance(payload[k],str) or not IDENTIFIER.fullmatch(payload[k]) for k in ['state_id','model_id'])):
        raise ValueError('Only typed public own-swarm work is permitted')
    if not candidate and (type(payload['start']) is not int or type(payload['count']) is not int
            or not 1<=payload['count']<=definition()['maximum_documents_per_shard']
            or not 0<=payload['start']<4096 or payload['start']+payload['count']>4096):
        raise ValueError('Own training document range is invalid')
    if candidate and (not isinstance(payload['contributions'],list) or len(payload['contributions'])!=2):
        raise ValueError('The physical pilot requires exactly two disjoint workers')


def verify_contributions(records, *, model_id, training_state_id, requester_id, workers=WORKERS):
    from src.network.work_protocol import ResultReceipt, WorkResult
    if not isinstance(records,list) or len(records)!=2:
        raise ValueError('Two physical own-swarm contributions are required')
    seen_workers=set();seen_jobs=set();ranges=[];outputs=[]
    for item in records:
        if not isinstance(item,dict) or set(item)!={'worker_id','result'}:
            raise ValueError('Own contribution record is invalid')
        worker=item['worker_id'];result=WorkResult.from_dict(item['result'])
        if (worker not in workers or worker in seen_workers or result.job_id in seen_jobs
                or result.status!='ok' or not result.receipt
                or not ResultReceipt.from_dict(result.receipt).verify(result,
                    expected_worker_id=worker,expected_requester_id=requester_id)):
            raise ValueError('Own contribution signature, physical worker or job is invalid')
        output=result.output
        required={'state_id','model_id','dataset_id','training_policy_id','start','count',
                  'tokens','loss','gradient_id','gradient_bytes'}
        if set(output)!=required:
            raise ValueError('Own gradient result has unexpected fields')
        validate_payload({k:output[k] for k in ['state_id','model_id','dataset_id','training_policy_id','start','count']})
        if (output['model_id']!=model_id or output['state_id']!=training_state_id
                or type(output['tokens']) is not int or not 1<=output['tokens']<=output['count']*128
                or type(output['gradient_bytes']) is not int or not 0<output['gradient_bytes']<=definition()['maximum_artifact_bytes']
                or not isinstance(output['gradient_id'],str) or not IDENTIFIER.fullmatch(output['gradient_id'])
                or type(output['loss']) not in (float,int) or not math.isfinite(output['loss']) or output['loss']<0):
            raise ValueError('Own gradient belongs to another model, state or token budget')
        seen_workers.add(worker);seen_jobs.add(result.job_id)
        ranges.append((output['start'],output['start']+output['count']));outputs.append(output)
    ranges.sort()
    if ranges[0][1]>ranges[1][0]:raise ValueError('Own productive shards overlap')
    return outputs


def verdict(before, after):
    p=definition()
    for value in [before,after]:
        if (set(value)!={'nll','byte_accuracy','predicted_bytes','documents'}
                or type(value['nll']) not in (float,int) or not math.isfinite(value['nll']) or value['nll']<=0
                or type(value['byte_accuracy']) not in (float,int) or not 0<=value['byte_accuracy']<=1
                or value['predicted_bytes']!=57123 or value['documents']!=128):
            raise ValueError('Own learning evaluation is incomplete or incompatible')
    return (before['nll']-after['nll']>=max(p['minimum_absolute_nll_gain'],before['nll']*p['minimum_relative_nll_gain'])
            and after['byte_accuracy']>=before['byte_accuracy']-p['maximum_aggregate_byte_accuracy_drop'])
