"""Retrospective, training-only diagnostic; never a promotion policy."""
import hashlib
import json

PARENT_RELEASE = 'df231f9b93f6300ba7242da793d457880a416448f00d0e9b914e6a3f7b1e533a'
PARENT_MODEL = '506317cc9f40da9dea40e6d14a94910ac6625ae478b5f2d0248cb1bbbb93a6c3'
SOURCE_REPORT = 'b5d9358df1d49b4d14c3991c781aef9319cf10e3c100b8233e00744b3309eea8'
GRADIENTS = (
    ('18c219146676d9e8f99fc303c03cd906e910184e7ecffef7e2c2bc619d18a580', 3223824, 0, 64, 5759),
    ('12ec2a4311f6b253b4518f7d2ccf1809520fb7ea33cf1be7141ad7819721db80', 3223824, 64, 64, 5103),
    ('b942d0362da0f8f7d21348da792eb93df78bf97f5b45441b226eb1cbc95cfa30', 3223832, 128, 64, 5717),
)


def definition():
    from .language_policy import MATH_POLICY_ID, dataset_id
    from .public_math import ARTIFACTS
    return {'version': 1, 'action': 'public-retention-diagnostic-v1',
        'parent_release_id': PARENT_RELEASE, 'model_id': PARENT_MODEL,
        'source_report_id': SOURCE_REPORT, 'source_trial': 'offset0-p4-distributed',
        'training_policy_id': MATH_POLICY_ID, 'math_training_artifact_id': ARTIFACTS['train'][0],
        'historical_dataset_id': dataset_id(1), 'math_range': [0, 192],
        'historical_training_prefix': 64, 'historical_training_full': 200,
        'gradient_metadata': [list(row) for row in GRADIENTS], 'learning_rate': .001,
        'measurements': ['baseline-training-loss-and-reference-gradients',
            'decoded-signed-directions-vs-prefix-and-full-training-gradients',
            'reconstructed-source-vs-saved-half-precision-first-desktop-shard',
            'finite-sgd-training-loss-change-and-first-order-residual', 'restore-parent-parameters'],
        'held_out_data_used_in_compute': False, 'generation_performed': False,
        'scope': 'Retrospective diagnosis of one fixed rejected update; not a fresh quality test or training-method selection',
        'new_model_publication_authorized': False,
        'reference': 'https://arxiv.org/abs/1812.00420v2',
        'reference_scope': 'Reference-gradient projection motivates dot-product diagnostics; not a reproduction of A-GEM benchmarks'}


def protocol_id():
    return hashlib.sha256(json.dumps(definition(), sort_keys=True, separators=(',', ':')).encode()).hexdigest()
