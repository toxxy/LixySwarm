"""Prospective public TRAIN answer calibration; no held-out labels or optimizer."""
import hashlib
import math

from .language_retention import PARENT_MODEL, PARENT_RELEASE
from .language_specialist_answers import SPECIALIST_ID
from .public_math import ARTIFACTS, canonical, final_number, gate_definition


def definition():
    return {'version':1, 'action':'public-language-outcome-calibration-v1',
        'parent_model_id':PARENT_MODEL, 'parent_release_id':PARENT_RELEASE, 'specialist_id':SPECIALIST_ID,
        'training_artifact_id':ARTIFACTS['train'][0], 'training_indices':list(range(128)),
        'partition':'physical shard 0 receives even indices; physical shard 1 receives odd indices',
        'shards':2, 'experts':['accepted-parent','research-specialist'],
        'specialist_update_partition':'TRAIN[1024:4096], disjoint from calibration',
        'compute_dtype':'float32', 'streamed_base':True, 'seed':20261015,
        'decoding':gate_definition()['decoding'], 'maximum_new_tokens':384,
        'ram_gb':10, 'disk_reservation_gb':.01, 'maximum_seconds':1800,
        'outcomes':['strict_numeric_correctness','valid_final_format','output_tokens','generation_seconds','teacher_forced_nll'],
        'evaluation_data_used':False, 'optimizer_updates':False, 'native_publication_authorized':False,
        'scope':'Two existing PCs collect frozen-expert TRAIN outcomes; not a speedup, learned-router or promotion claim'}


def protocol_id():
    return hashlib.sha256(canonical(definition())).hexdigest()


def partition(snapshot, shard):
    if type(shard) is not int or shard not in (0,1) or snapshot.get('split') != 'train':
        raise ValueError('Outcome calibration only accepts a fixed TRAIN shard')
    indices = definition()['training_indices'][shard::2]
    rows = [snapshot['rows'][i] for i in indices]
    if len(rows) != 64 or len({r['case'] for r in rows}) != 64:
        raise ValueError('Public outcome calibration has incomplete or duplicate TRAIN cases')
    return {'split':'train', 'rows':rows}, indices


def score_training_answers(answers, selected):
    if selected.get('split') != 'train': raise ValueError('Outcome scorer only accepts TRAIN rows')
    if (not isinstance(answers,list) or len(answers) != len(selected['rows'])
            or any(not isinstance(a,dict) or set(a) != {'case','answer'} or not isinstance(a['answer'],str)
                   or len(a['answer'].encode())>8192 for a in answers)
            or [a['case'] for a in answers] != [r['case'] for r in selected['rows']]):
        raise ValueError('Training outcomes have missing, reordered or unknown cases')
    return {a['case']:final_number(a['answer']) == r['final_number'] for a,r in zip(answers,selected['rows'])}


def verify_outcomes(value, snapshot):
    if (value.get('protocol_id') != protocol_id() or value.get('expert_ids') != [PARENT_MODEL,SPECIALIST_ID]
            or value.get('evaluation_data_used') is not False or value.get('optimizer_updates') is not False):
        raise ValueError('Outcome calibration provenance or frozen expert binding changed')
    selected, indices = partition(snapshot,value.get('shard'))
    if value.get('training_indices') != indices or set(value.get('experts',{})) != {'accepted-parent','research-specialist'}:
        raise ValueError('Outcome calibration changed its fixed TRAIN partition')
    summaries = {}
    for name,result in value['experts'].items():
        answers = result['answers']; tokens = result['output_tokens']; seconds = result['generation_seconds']
        correct = score_training_answers(answers,selected)
        if (len(tokens) != 64 or len(seconds) != 64 or any(type(n) is not int or not 1<=n<=384 for n in tokens)
                or any(type(t) not in (int,float) or not math.isfinite(t) or t<=0 for t in seconds)
                or type(result['nll']) not in (int,float) or not math.isfinite(result['nll']) or result['nll'] < 0
                or type(result['tokens']) is not int or result['tokens'] <= 0
                or sum(correct.values()) != result['correct'] or result['total'] != 64
                or result['at_output_limit'] != sum(n==384 for n in tokens)):
            raise ValueError('Outcome answers, exact scores, token counts or finite costs differ')
        summaries[name] = {'correct':sum(correct.values()), 'total':64,
            'valid_format':sum(final_number(row['answer']) is not None for row in answers),
            'output_tokens':sum(tokens), 'generation_seconds':sum(seconds), 'nll':result['nll']}
    return {'protocol_id':protocol_id(), 'shard':value['shard'], 'examples':64, 'experts':summaries,
        'public_training_only':True, 'native_publication_authorized':False}
