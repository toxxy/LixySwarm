"""Prospective comparison of two responses to an invalid numeric answer.

No selector is fitted and no rejected specialist becomes a native expert.
This tests whether cooperation offers value beyond another parent call.
"""
import hashlib
import math
from pathlib import Path

from .language_retention import PARENT_MODEL, PARENT_RELEASE
from .language_specialist_answers import SPECIALIST_ID
from .public_math import (canonical, final_number, SOURCE_REPOSITORY, SOURCE_REVISION,
    SOURCES, SELECTION, _source_file, _source_rows)

SNAPSHOT_ID = '1c7873583f47330e5cfc0b0b4047d491dc2deb1359d1a5e1c57b4adfeb81708c'
SNAPSHOT_BYTES = 79509
REPAIR_INSTRUCTION = ('Review your previous solution. Correct any calculation errors. '
    'End with exactly one line containing #### followed by the final number. '
    'Do not use #### anywhere else in your new answer.')


def build_snapshot(source_directory):
    license_text = _source_file(source_directory, 'LICENSE').decode()
    test, test_questions = _source_rows(_source_file(source_directory, 'test.jsonl'), 'test')
    _, train_questions = _source_rows(_source_file(source_directory, 'train.jsonl'), 'train')
    if train_questions & test_questions or len(test) < 256:
        raise ValueError('Prospective rows must be outside the public training split')
    return {'version': 1, 'private_inputs': False, 'source_repository': SOURCE_REPOSITORY,
        'source_revision': SOURCE_REVISION, 'source_sha256': SOURCES['test.jsonl'][0],
        'license': {'spdx': 'MIT', 'sha256': SOURCES['LICENSE'][0], 'text': license_text},
        'selection': SELECTION, 'sorted_test_start': 128, 'sorted_test_stop': 256,
        'pretraining_contamination_excluded': False, 'rows': test[128:256]}


def load_snapshot(path):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size != SNAPSHOT_BYTES:
        raise ValueError('Prospective comparison snapshot size differs from its pin')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SNAPSHOT_ID:
        raise ValueError('Prospective comparison snapshot differs from its pin')
    import json
    value = json.loads(raw)
    if canonical(value) != raw or len(value['rows']) != 128:
        raise ValueError('Invalid prospective comparison rows')
    return value


def definition():
    return {'version': 1, 'action': 'public-format-recovery-comparison-v1',
        'parent_model_id': PARENT_MODEL, 'parent_release_id': PARENT_RELEASE,
        'specialist_id': SPECIALIST_ID, 'snapshot_id': SNAPSHOT_ID, 'snapshot_bytes': SNAPSHOT_BYTES,
        'selection': 'pinned sorted upstream TEST rows 128:256; original evaluation rows 0:128 excluded',
        'rule': 'keep the shared first parent answer if its final numeric line is valid; otherwise make one second call',
        'arms': ['parent', 'parent-repair', 'specialist-fallback'],
        'repair_instruction': REPAIR_INSTRUCTION, 'second_call_order': 'repair first on even row index; specialist first on odd',
        'maximum_input_tokens_per_call': 1024, 'maximum_new_tokens_per_call': 384,
        'maximum_calls_per_policy_per_case': 2, 'maximum_messages_bytes_per_call': 4096,
        'compute_dtype': 'float32', 'streamed_base': True, 'seed': 20261015, 'ram_gb': 10,
        'disk_reservation_gb': .02, 'maximum_seconds': 1800,
        'component_gate': {'minimum_new_correct_over_parent': 4, 'maximum_regressions': 0,
            'must_exceed_parent_repair_correct': True, 'maximum_time_ratio_to_parent_repair': 1.1},
        'training_performed': False, 'neural_routing_evaluated': False,
        'native_publication_authorized': False, 'original_global_gates_unchanged': True,
        'scope': 'Prospective separate math component: parser-aware specialist cooperation versus same-parent repair; no promotion'}


def protocol_id():
    return hashlib.sha256(canonical(definition())).hexdigest()


def needs_second_call(answer):
    # This function deliberately accepts no reference answer or case target.
    return final_number(answer) is None


def repair_messages(question, draft):
    from .public_math import prompt
    return [{'role': 'user', 'content': prompt({'question': question})},
        {'role': 'assistant', 'content': draft}, {'role': 'user', 'content': REPAIR_INSTRUCTION}]


def verify_report(report, snapshot):
    p = definition()
    if (report['protocol_id'] != protocol_id() or report['snapshot_id'] != SNAPSHOT_ID
            or report.get('training_performed') is not False or report.get('model_promoted') is not False):
        raise ValueError('Comparison report has another protocol or dataset')
    records = report['records']
    if len(records) != len(snapshot['rows']) or len({row['case'] for row in snapshot['rows']}) != len(records):
        raise ValueError('Comparison evidence is incomplete')
    totals = {arm: {'correct': 0, 'new_correct': 0, 'regressions': 0, 'calls': 0,
        'input_tokens': 0, 'output_tokens': 0, 'generation_seconds': 0.} for arm in p['arms']}
    second_count = 0
    for index, (row, record) in enumerate(zip(snapshot['rows'], records)):
        if record['case'] != row['case'] or record['second_order'] != (['repair','specialist'] if index%2 == 0 else ['specialist','repair']):
            raise ValueError('Comparison case or predetermined execution order changed')
        parent = record['parent']; second = needs_second_call(parent['answer'])
        if record['second_requested'] is not second or set(record['responses']) != ({'repair','specialist'} if second else set()):
            raise ValueError('Comparison routing used something other than the parent format')
        second_count += second
        for result in [parent, *record['responses'].values()]:
            if (set(result) != {'answer','output_tokens','input_tokens','generation_seconds'}
                    or not isinstance(result['answer'],str) or len(result['answer'].encode()) > 8192
                    or type(result['input_tokens']) is not int or not 1 <= result['input_tokens'] <= p['maximum_input_tokens_per_call']
                    or type(result['output_tokens']) is not int or not 1 <= result['output_tokens'] <= p['maximum_new_tokens_per_call']
                    or type(result['generation_seconds']) not in (int,float) or not math.isfinite(result['generation_seconds']) or result['generation_seconds'] < 0):
                raise ValueError('Comparison generation evidence exceeds its declared bounds')
        correct_parent = final_number(parent['answer']) == row['final_number']
        for arm in p['arms']:
            calls = [parent]
            if second and arm != 'parent': calls.append(record['responses']['repair' if arm == 'parent-repair' else 'specialist'])
            result = calls[-1]; correct = final_number(result['answer']) == row['final_number']
            metrics = totals[arm]
            metrics['correct'] += correct; metrics['new_correct'] += correct and not correct_parent
            metrics['regressions'] += correct_parent and not correct; metrics['calls'] += len(calls)
            for field in ['input_tokens','output_tokens','generation_seconds']:
                metrics[field] += sum(call[field] for call in calls)
    fallback = totals['specialist-fallback']; repair = totals['parent-repair']
    checks = {'gain_at_least_four': fallback['new_correct'] >= 4,
        'no_regressions': fallback['regressions'] == 0,
        'exceeds_same_parent_repair': fallback['correct'] > repair['correct'],
        'within_declared_time_budget': fallback['generation_seconds'] <= repair['generation_seconds']*1.1}
    return {'total': len(records), 'invalid_parent_answers': second_count, 'policies': totals,
        'component_conditions': checks, 'component_passed': all(checks.values()),
        'actual_fresh_generation_calls': len(records)+2*second_count,
        'counterfactual_generation_not_counted_as_primary_policy_cost': True,
        'neural_routing_evaluated': False, 'model_promoted': False}
