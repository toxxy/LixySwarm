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


def limits(version):
    if type(version) is not int or version not in {1, 2}:
        raise ValueError('Unknown immutable language training policy')
    return (8, 1536) if version == 1 else (BATCH_POLICY['maximum_examples'], BATCH_POLICY['maximum_answer_tokens'])


def retention_plan(start, counts):
    """Every distinct group gets an anchor; a larger batch cannot dilute it."""
    if (type(start) is not int or start < 0 or not isinstance(counts, list) or not 1 <= len(counts) <= limits(2)[0]
            or start+len(counts) > len(public_corpus()['train'])
            or any(type(n) is not int or not 1 <= n <= 192 for n in counts)):
        raise ValueError('Invalid public batch retention inputs')
    total = sum(counts)
    return [((start//8+offset//8) % len(public_corpus()['retention']), sum(counts[offset:offset+8])/total)
            for offset in range(0, len(counts), 8)]
