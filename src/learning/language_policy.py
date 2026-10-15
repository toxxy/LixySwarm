"""Immutable public batch policy; legacy evaluation and data remain unchanged."""
import hashlib
import json

from .public_conversation import public_corpus, corpus_digest

BATCH_CLIENT_VERSION = '0.4.0a14'
BATCH_POLICY = {
    'version': 2,
    'dataset_id': corpus_digest(public_corpus()),
    'maximum_examples': 64,
    'maximum_answer_tokens': 64*192,
    'maximum_answer_tokens_per_example': 192,
    'supervised_normalization': 'answer-token-mean',
    'retention': 'one-public-anchor-per-eight-examples-weighted-by-group-answer-tokens',
    'retention_anchor_index': '(start // 8 + group_index) modulo public_anchor_count',
    'gradient_clipping': 'one-global-l2-clip-at-one-after-mean-supervision-and-retention',
    'gradient_format': 'lixyswarm-language-gradient-v1-float32',
}
BATCH_POLICY_ID = hashlib.sha256(json.dumps(BATCH_POLICY, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
HALF_CLIENT_VERSION = '0.4.0a17'
HALF_POLICY = {**BATCH_POLICY, 'version': 3,
    'gradient_format': 'lixyswarm-language-gradient-v2-float16',
    'quantization': 'ieee754-binary16-round-nearest-even',
    'maximum_relative_l2_error': .001,
    'decoded_clipping': 'global-l2-clip-at-one-after-float32-reconstruction'}
HALF_POLICY_ID = hashlib.sha256(json.dumps(HALF_POLICY, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def policy_id(version):
    limits(version)
    return {1: None, 2: BATCH_POLICY_ID, 3: HALF_POLICY_ID}[version]


def policy_from_id(identifier):
    for version in [1, 2, 3]:
        if identifier == policy_id(version):return version
    raise ValueError('Unknown immutable language training policy identity')


def descriptor_policy(descriptor):
    version=descriptor.get('version')
    if type(version) is not int or version not in {1,2,3,4}:raise ValueError('Unknown language descriptor policy')
    return max(1,version-1)


def gradient_operation(version):
    limits(version);return f'language.gradient.v{version+1}'


def candidate_operation(version):
    limits(version);return f'language.candidate.v{version}'


def work_policy(operation):
    for version in [1,2,3]:
        if operation in {gradient_operation(version),candidate_operation(version)}:return version
    raise ValueError('Unknown language work operation')


def enabled_policy(engine,version):
    limits(version)
    return (version==1 or getattr(engine,'batching_enabled',False) is True
        and (version==2 or getattr(engine,'half_precision_enabled',False) is True))


def limits(version):
    if type(version) is not int or version not in {1, 2, 3}:
        raise ValueError('Unknown immutable language training policy')
    if version==1:return (8,1536)
    selected=BATCH_POLICY if version==2 else HALF_POLICY
    return selected['maximum_examples'],selected['maximum_answer_tokens']


def retention_plan(start, counts):
    """Every distinct group gets an anchor; a larger batch cannot dilute it."""
    if (type(start) is not int or start < 0 or not isinstance(counts, list) or not 1 <= len(counts) <= limits(2)[0]
            or start+len(counts) > len(public_corpus()['train'])
            or any(type(n) is not int or not 1 <= n <= 192 for n in counts)):
        raise ValueError('Invalid public batch retention inputs')
    total = sum(counts)
    return [((start//8+offset//8) % len(public_corpus()['retention']), sum(counts[offset:offset+8])/total)
            for offset in range(0, len(counts), 8)]
