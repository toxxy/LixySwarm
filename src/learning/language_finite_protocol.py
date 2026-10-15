"""Prospective TRAIN-only finite-step experiment, separate from native policies."""
import hashlib
import json
from .language_retention import definition as retention_definition

KINDS=('arithmetic','context','exact','explanation','extraction','sorting')


def definition(precision='bfloat16'):
    prior=retention_definition(precision)
    return {**{key:prior[key] for key in ['parent_release_id','model_id','source_report_id','source_trial',
        'training_policy_id','math_training_artifact_id','historical_dataset_id','math_range','gradient_metadata',
        'historical_training_prefix','historical_training_full','model_compute_dtype','float32_matmul_precision','cuda_tf32','ram_gb']},
        'version':1,'action':'public-finite-step-v1','target':'math_train192',
        'protected':['old_prefix64','old_full200',*['old_kind_'+kind for kind in KINDS]],
        'rates':[.001/(2**index) for index in range(12)],'rate_selection':'first measured safe rate in descending fixed schedule',
        'absolute_tolerance':1e-7,'relative_tolerance':1e-6,'minimum_absolute_gain':1e-5,'minimum_relative_gain':1e-4,
        'execution':'train-with-autograd in every loss pass; identical no-update control before search',
        'rollback':'restore parent before every trial; exceptions and exhaustion restore; experimental accepted step restored before exit',
        'held_out_data_used_in_compute':False,'generation_performed':False,'new_model_publication_authorized':False,
        'scope':'Prospective finite safeguard for one saved two-PC update on public training groups; not new held-out quality or a native learning cycle'}


def protocol_id(precision='bfloat16'):
    return hashlib.sha256(json.dumps(definition(precision),sort_keys=True,separators=(',',':')).encode()).hexdigest()
