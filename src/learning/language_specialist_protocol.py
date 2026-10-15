"""Prospective complementary specialist plus immutable ordered expert selection."""
import hashlib
import json
import math

from .language_selector_protocol import definition as selector_definition


def definition():
    return {**selector_definition(),'action':'public-language-specialist-v1',
        'experts':['accepted-parent-adapter','new-public-math-specialist-adapter'],
        'maximum_seconds':3600,'disk_reservation_gb':.2,'ram_gb':10,
        'math_training_examples':1024,'specialist_training_start':1024,'specialist_training_examples':3072,
        'specialist_epochs':1,'specialist_batch_size':64,'specialist_replay_examples':8,'specialist_replay_weight':.25,
        'specialist_learning_rate':.00002,'specialist_warmup_steps':4,'specialist_betas':[.9,.999],
        'specialist_epsilon':1e-8,'specialist_optimizer':'AdamW; zero weight decay; global clip L2 1 after token-normalized math and replay',
        'specialist_math_order':'seeded permutation of TRAIN[1024:4096], each exactly once',
        'specialist_replay_order':'old public TRAIN[(step*8+i)%200], i in [0,8)',
        'calibration_math_partition':'TRAIN[:1024], disjoint from all specialist math updates',
        'memory_feature_space':'lixyswarm-public-language-adapter-selection-v2',
        'comparison_expert_name':'trained-math-specialist','choice_count_field':'specialist_choices',
        'primary_selector':'public','loss_screen':'Original math and historical NLL improvement thresholds, no changes to trained state',
        'evaluation':'Freeze all learned states, screen all fixed NLL; if primary passes, run every original answer gate against FP32 and BF16 parent',
        'primary_selected_before_heldout':True,'full_answer_gates_required':True,
        'native_publication_authorized':False,'new_remote_contributions':0,
        'scope':'Single physical machine research, no distributed-learning credit or native activation; parent expert and original gates retained'}


def protocol_id():
    return hashlib.sha256(json.dumps(definition(),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def necessary_loss_gate(before,after):
    from .public_math import gate_definition
    gate=gate_definition()
    values={}
    for group,relative,absolute in [('math',gate['minimum_relative_math_nll_improvement'],gate['minimum_absolute_math_nll_improvement']),
            ('historical',gate['historical_nll_relative_improvement'],gate['historical_nll_absolute_improvement'])]:
        values[group]=after[group]['nll']<before[group]['nll']-max(absolute,before[group]['nll']*relative)
    return {'math_nll_gate_passed':values['math'],'historical_nll_gate_passed':values['historical'],
        'necessary_loss_gates_passed':all(values.values()),'native_publication_authorized':False}


def verify_specialist_report(report):
    from .language_selector_protocol import verify_loss_report
    p=definition();proof=verify_loss_report(report,protocol=p,identifier=protocol_id())
    trained=report['specialist_training'];batch=p['specialist_batch_size'];count=p['specialist_training_examples']
    order=trained['math_order'];steps=trained['steps_trace'];replay=p['specialist_replay_examples']
    if (trained['steps']!=count//batch or trained['math_examples']!=count or len(steps)!=count//batch
            or any(type(i) is not int for i in order) or sorted(order)!=list(range(count))
            or trained['public_training_only'] is not True or trained['evaluation_used'] is not False
            or trained['generalist_file_updated'] is not False or report['generalist_files_written'] is not False
            or report['updated_model_files_written'] is not True or report['new_remote_contributions']!=0
            or report['binding']['expert_ids']!=[p['parent_model_id'],report['selector_artifacts']['specialist.safetensors']['sha256']]):
        raise ValueError('Specialist training coverage or immutable expert binding changed')
    for i,row in enumerate(steps):
        rate=p['specialist_learning_rate']*min(1.,(i+1)/p['specialist_warmup_steps'])
        if (row['step']!=i+1 or row['math_indices']!=order[i*batch:(i+1)*batch]
                or row['replay_indices']!=[(i*replay+j)%p['historical_training_examples'] for j in range(replay)]
                or any(type(row[k]) not in (int,float) or not math.isfinite(row[k]) or row[k]<0 for k in ['math_nll','replay_nll','gradient_norm_before_clip'])
                or type(row['math_answer_tokens']) is not int or not 1<=row['math_answer_tokens']<=batch*512
                or type(row['replay_answer_tokens']) is not int or not 1<=row['replay_answer_tokens']<=replay*512
                or not math.isclose(row['learning_rate'],rate,rel_tol=1e-15,abs_tol=0.)):
            raise ValueError('Specialist step data, token cost or optimizer schedule changed')
    screen=necessary_loss_gate(report['evaluations']['accepted-parent'],report['evaluations'][p['primary_selector']])
    if (screen!=report['loss_screen'] or report['generation_performed'] is not screen['necessary_loss_gates_passed']
            or report['full_generation_gates_evaluated'] is not report['generation_performed']):
        raise ValueError('Specialist screening does not use the predetermined primary selector')
    return {**proof,'complete_math_training_coverage_verified':True,'replay_schedule_verified':True,
        'ordered_expert_binding_verified':True,'new_remote_contributions':0}
