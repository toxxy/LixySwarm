"""Prospective four-step TRAIN selection followed by unchanged held-out gates."""
import hashlib,json
from .language_finite_protocol import definition as finite_definition


def definition():
    from .public_math import ARTIFACTS,gate_id,gate_definition
    value=finite_definition('float32',group_direction=True,streamed_base=True)
    value.update(version=1,action='public-block-quality-v1',maximum_steps=4,maximum_seconds=3600,
        projection_relative_margin=.01,maximum_projected_direction_norm=1.,disk_reservation_gb=.024,
        direction_source='Fixed saved three-shard two-PC direction from parent 2; projected again against current TRAIN groups each microstep',
        parent_preservation='Each trial preserves current groups and the initial-parent ceiling with the same numerical tolerances; no cumulative tolerance drift',
        quality_selection='No held-out measurement changes steps, rates, direction or hyperparameters; assess parent and final TRAIN-selected state once',
        rollback='Restore each step-start before its trial; exceptions and final assessment restore the initial parent; no model files written',
        evaluation_artifact_id=ARTIFACTS['evaluation'][0],evaluation_gate_id=gate_id(),evaluation_gate=gate_definition(),
        held_out_data_used_in_compute=True,held_out_data_used_for_training_or_selection=False,generation_performed=True,
        model_files_written=False,new_model_publication_authorized=False,
        scope='One bounded research block on a fixed shared direction, not four new network contributions or native learning cycles')
    return value


def protocol_id():
    return hashlib.sha256(json.dumps(definition(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
