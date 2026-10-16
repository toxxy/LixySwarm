"""Prospective own-weight foundation experiment; no tensor imports."""
from .public_math import ARTIFACTS, digest


def definition():
    from .swarm_foundation import FOUNDATION_CONFIG
    return {'version': 1, 'action': 'own-causal-foundation-v1',
        'model_config': dict(FOUNDATION_CONFIG), 'initialization_seed': 20261015,
        'sample_seed': 20261016, 'pretrained_weights': False, 'teacher_used_in_this_run': False,
        'training_artifact': list(ARTIFACTS['train']), 'evaluation_artifact': list(ARTIFACTS['evaluation']),
        'document_format': 'Question: {question}\nAnswer: {answer}\n\n',
        'encoding': 'utf8-bytes-256-v1', 'steps': 1024, 'batch_size': 16,
        'block_size': 128, 'optimizer': 'AdamW', 'learning_rate': .0006,
        'weight_decay': .01, 'gradient_clip_norm': 1., 'warmup_steps': 32,
        'evaluation_maximum_bytes_per_document': 512,
        'evaluation_schedule': 'initial-and-final-only-no-selection',
        'causality_atol': 1e-6, 'causality_rtol': 1e-5,
        'minimum_relative_heldout_nll_gain': .01,
        'maximum_seconds': 1200, 'ram_gb': 4., 'disk_reservation_gb': .08,
        'scope': 'Own causal byte-language pretraining baseline, not math reasoning or chat certification',
        'teachers_permitted_in_future': True, 'distributed_training_performed': False,
        'model_promotion_authorized': False, 'native_cycles_added': 0}


def protocol_id():
    return digest(definition())
