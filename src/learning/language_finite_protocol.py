"""Prospective TRAIN-only finite-step experiment, separate from native policies."""
import hashlib
import json
from .language_retention import definition as retention_definition

KINDS=('arithmetic','context','exact','explanation','extraction','sorting')


def definition(precision='bfloat16',*,group_direction=False,streamed_base=False):
    if streamed_base and (precision!='float32' or not group_direction):
        raise ValueError('Streamed base requires FP32 computation and group constraints')
    prior=retention_definition(precision)
    value={**{key:prior[key] for key in ['parent_release_id','model_id','source_report_id','source_trial',
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
    if group_direction:
        value.update(version=2,action='public-group-finite-step-v2',
            direction_preparation='Normalize each protected TRAIN gradient; bounded dual coordinate halfspace projection of saved two-PC direction',
            maximum_projection_sweeps=512,projection_relative_tolerance=1e-7,
            reference='https://papers.neurips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf',
            reference_scope='Separate first-order task constraints from GEM equations 7-8; bounded local solver and different model/data, not a benchmark reproduction',
            projection_failure='No finite-step search when returned representation violates a group constraint')
    if streamed_base:
        value.update(version=3,action='public-streamed-group-finite-step-v3',ram_gb=6,maximum_seconds=1200,
            base_matrix_storage_dtype='bfloat16',base_execution='frozen-bf16-storage-fp32-linear-recompute-v1',
            precision_conversion='Original public checkpoint tensors are BF16; exact FP32 conversion per linear, FP32 embedding output and small frozen parameters; no extra quantization',
            backward_storage='Save original frozen BF16 matrices only; recompute temporary FP32 matrices during first-order backward',
            memory_acceptance='Measure actual peak allocation in each physical GPU; no presumed laptop fit',
            implementation_reference='https://docs.pytorch.org/docs/2.12/notes/extending.html#extending-torch-autograd')
    return value


def protocol_id(precision='bfloat16',*,group_direction=False,streamed_base=False):
    return hashlib.sha256(json.dumps(definition(precision,group_direction=group_direction,streamed_base=streamed_base),sort_keys=True,separators=(',',':')).encode()).hexdigest()
