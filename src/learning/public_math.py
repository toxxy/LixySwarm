"""Pinned GSM8K curriculum and a stronger, separate research acceptance gate.

This module imports no tensor runtime. It never discovers local text sources:
only exact upstream bytes or app-pinned derived snapshots can be loaded.
Historical language policies, datasets and gates are unchanged.
"""
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re

VERSION = 'lixyswarm-public-gsm8k-v1'
SOURCE_REPOSITORY = 'https://github.com/openai/grade-school-math'
SOURCE_REVISION = '3101c7d5072418e28b9008a6636bde82a006892c'
SOURCES = {
    'LICENSE': ('86bbb73e855821d7c401912fd4bf82e34313e6e3b6fd6f909f2b6cc9e209a53b', 1062),
    'train.jsonl': ('17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465', 4166206),
    'test.jsonl': ('3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14', 749738),
}
COUNTS = {'train': 4096, 'evaluation': 128}
UPSTREAM_SPLITS = {'train': 'train', 'evaluation': 'test'}
ARTIFACTS = {
    'train': ('a1a36bc0cea6f5a8dbf81a5828a01da3ca0674c534431ef51583f3f46a915be6', 2485927),
    'evaluation': ('188812577d2a7834eff5fc0ac0fad5f09f121b3587b34b7ba5d88ba37537be50', 83241),
}
SELECTION = 'sort-sha256(version:upstream-split:exact-question)-then-take-count-v1'
HISTORICAL_GATE_ID = 'cb33111cc44098bc845ffff8eb51888527d6667fd872f03407a55928b9b8020b'
MAX_SNAPSHOT_BYTES = 8*1024**2
MAX_TRACE_BYTES = 4*1024**2
MATH_REPORT_FIELDS = {'math_before', 'math_after', 'math_trace_id', 'math_trace_bytes'}
PROMPT_PREFIX = ('Solve this word problem. Explain the calculations briefly, then end with '
                 'a line containing #### followed by the final number.\n\n')
NUMBER = r'-?(?:0|[1-9][0-9]*|[1-9][0-9]{0,2}(?:,[0-9]{3})+)(?:\.[0-9]+)?'
FINAL = re.compile(r'(?:^|\n)####[ \t]+('+NUMBER+r')[ \t]*\Z')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def final_number(answer):
    """Only an explicit final numeric line is a result; never evaluate expressions."""
    if not isinstance(answer, str) or len(answer.encode()) > 8192:
        return None
    matches = list(FINAL.finditer(answer.rstrip()))
    if len(matches) != 1 or len(matches[0][1]) > 64 or answer.count('####') != 1:
        return None
    try:
        value = Decimal(matches[0][1].replace(',', ''))
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    text = format(value, 'f')
    return '0' if value == 0 else text.rstrip('0').rstrip('.') if '.' in text else text


def prompt(row):
    return PROMPT_PREFIX + row['question']


def _source_file(directory, name):
    path = Path(directory)/name
    expected, size = SOURCES[name]
    if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
        raise ValueError('Public math source is missing or has an unexpected size')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Public math source differs from the pinned upstream revision')
    return raw


def _source_rows(raw, split):
    rows = []
    questions = set()
    for line in raw.decode('utf-8').splitlines():
        value = json.loads(line)
        if (not isinstance(value, dict) or set(value) != {'question', 'answer'}
                or any(not isinstance(value[k], str) or not value[k] for k in value)):
            raise ValueError('Invalid bounded public math source row')
        question, answer = value['question'], re.sub(r'<<[^<>]*>>', '', value['answer'])
        if question in questions:
            raise ValueError('Repeated upstream math question')
        questions.add(question)
        target = final_number(answer)
        if target is None:
            raise ValueError('Upstream math answer has no unambiguous numeric target')
        if (len(question.encode()) > 1000 or len(answer.encode()) > 1200
                or len((question+value['answer']).encode()) > 1800):
            continue
        key = hashlib.sha256((VERSION+':'+split+':'+question).encode()).hexdigest()
        rows.append({'case': key, 'question': question, 'answer': answer, 'final_number': target})
    return sorted(rows, key=lambda row: row['case']), questions


def build_snapshots(source_directory):
    """Reproduce derived artifacts from exact public bytes, including their license."""
    license_text = _source_file(source_directory, 'LICENSE').decode('utf-8')
    train, train_questions = _source_rows(_source_file(source_directory, 'train.jsonl'), 'train')
    test, test_questions = _source_rows(_source_file(source_directory, 'test.jsonl'), 'test')
    if train_questions & test_questions:
        raise ValueError('Upstream training and evaluation questions overlap')
    snapshots = {}
    for split, rows in [('train', train), ('evaluation', test)]:
        if len(rows) < COUNTS[split]:
            raise ValueError('Too few bounded public math examples')
        upstream = UPSTREAM_SPLITS[split]
        snapshots[split] = {
            'version': VERSION, 'split': split, 'private_inputs': False,
            'scope': 'English public math word problems; not general conversation certification',
            'source': {'repository': SOURCE_REPOSITORY, 'revision': SOURCE_REVISION,
                       'path': 'grade_school_math/data/'+upstream+'.jsonl', 'sha256': SOURCES[upstream+'.jsonl'][0]},
            'license': {'spdx': 'MIT', 'text': license_text, 'sha256': SOURCES['LICENSE'][0]},
            'selection': SELECTION, 'rows': rows[:COUNTS[split]],
            'pretraining_contamination_excluded': False,
        }
    return snapshots


def load_snapshot(path, split):
    if split not in ARTIFACTS:
        raise ValueError('Unknown public math split')
    path = Path(path)
    identifier, size = ARTIFACTS[split]
    if (path.is_symlink() or not path.is_file() or path.stat().st_size != size
            or not 0 < size <= MAX_SNAPSHOT_BYTES):
        raise ValueError('Public math snapshot is missing or exceeds its pinned size')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != identifier:
        raise ValueError('Public math snapshot differs from its locally pinned artifact')
    value = json.loads(raw)
    if (canonical(value) != raw or value['version'] != VERSION or value['split'] != split
            or value['private_inputs'] is not False or len(value['rows']) != COUNTS[split]
            or len({row['case'] for row in value['rows']}) != COUNTS[split]
            or any(final_number(row['answer']) != row['final_number'] for row in value['rows'])):
        raise ValueError('Invalid public math snapshot schema or numeric labels')
    return value


def curriculum_id():
    return digest({'version': VERSION, 'source_revision': SOURCE_REVISION,
                   'artifacts': ARTIFACTS, 'counts': COUNTS, 'selection': SELECTION})


def gate_definition():
    return {'version': 1, 'curriculum_id': curriculum_id(), 'evaluation_artifact_id': ARTIFACTS['evaluation'][0],
            'prompt_prefix': PROMPT_PREFIX,
            'minimum_exact_answer_gain': 4, 'maximum_exact_answer_regressions': 0,
            'minimum_relative_math_nll_improvement': .01, 'minimum_absolute_math_nll_improvement': 1e-5,
            'historical_gate_id': HISTORICAL_GATE_ID, 'historical_nll_relative_improvement': .001,
            'historical_nll_absolute_improvement': 1e-7, 'historical_fact_and_instruction_preservation': True,
            'decoding': 'greedy-no-calculator-no-tools', 'enable_thinking': False, 'maximum_new_tokens': 384,
            'maximum_sequence_tokens': 1024, 'maximum_answer_tokens': 512,
            'scope': 'Separate research gate; no authorization to publish an expert or restart native learning'}


def gate_id():
    return digest(gate_definition())


def score_answers(answers, evaluation):
    if evaluation.get('split') != 'evaluation' or digest(evaluation) != ARTIFACTS['evaluation'][0]:
        raise ValueError('Math scoring requires the exact pinned evaluation snapshot')
    targets = {row['case']: row['final_number'] for row in evaluation['rows']}
    if (not isinstance(answers, list) or len(answers) != len(targets)
            or any(not isinstance(row, dict) or set(row) != {'case', 'answer'}
                   or not isinstance(row['answer'], str) or len(row['answer'].encode()) > 8192 for row in answers)
            or {row['case'] for row in answers} != set(targets)):
        raise ValueError('Math evaluation has missing, repeated or unknown cases')
    return {row['case']: final_number(row['answer']) == targets[row['case']] for row in answers}


def verdict(before, after, evaluation):
    """Require new math gains as well as every historical quality condition."""
    from .public_quality import preservation_verdict as factual_verdict
    from .public_language_gate import preservation_verdict as instruction_verdict
    first, second = score_answers(before['answers'], evaluation), score_answers(after['answers'], evaluation)
    for result in [before, after, before['legacy'], after['legacy']]:
        nll = result['nll']
        if type(nll) not in (int, float) or not math.isfinite(nll) or nll <= 0:
            raise ValueError('Math and historical losses must be finite and positive')
    facts = factual_verdict(before['legacy']['quality'], after['legacy']['quality'])
    instructions = instruction_verdict(before['legacy']['instructions'], after['legacy']['instructions'])
    regressions = sorted(key for key in first if first[key] and not second[key])
    gain = sum(second.values())-sum(first.values())
    math_nll = after['nll'] < before['nll']-max(1e-5, before['nll']*.01)
    legacy_nll = after['legacy']['nll'] < before['legacy']['nll']-max(1e-7, before['legacy']['nll']*.001)
    return {'gate_id': gate_id(), 'total': len(first), 'baseline_passed': sum(first.values()),
            'candidate_passed': sum(second.values()), 'accuracy_gain': gain, 'regressions': regressions,
            'math_nll_gate_passed': math_nll, 'historical_nll_gate_passed': legacy_nll,
            'factual_verdict': facts, 'instruction_verdict': instructions,
            'eligible_for_research_promotion': gain >= 4 and not regressions and math_nll and legacy_nll
                and facts['eligible_for_promotion'] and instructions['eligible_for_promotion'],
            'native_publication_authorized': False, 'scope': gate_definition()['scope']}


def compact_evaluation(result):
    """Keep exact scoring semantics; bind full generated text separately by hash."""
    fields={'nll','tokens','answers','correct','total','output_tokens','at_output_limit'}
    if (not isinstance(result,dict) or set(result)!=fields or type(result['total']) is not int or result['total']!=COUNTS['evaluation']
            or type(result['correct']) is not int or not 0<=result['correct']<=result['total']
            or type(result['at_output_limit']) is not int
            or type(result['nll']) not in (int,float) or not math.isfinite(result['nll']) or result['nll']<=0
            or type(result['tokens']) is not int or not 1<=result['tokens']<=COUNTS['evaluation']*512
            or not isinstance(result['answers'],list) or len(result['answers'])!=COUNTS['evaluation']
            or not isinstance(result['output_tokens'],list) or len(result['output_tokens'])!=COUNTS['evaluation']
            or any(type(n) is not int or not 1<=n<=384 for n in result['output_tokens'])
            or result['at_output_limit']!=sum(n==384 for n in result['output_tokens'])):
        raise ValueError('Invalid bounded full math evaluation')
    projected=[];hashes=[]
    for row in result['answers']:
        if (not isinstance(row,dict) or set(row)!={'case','answer'} or not isinstance(row['case'],str)
                or not isinstance(row['answer'],str) or len(row['answer'].encode())>8192):
            raise ValueError('Invalid full math answer')
        number=final_number(row['answer'])
        projected.append({'case':row['case'],'answer':'' if number is None else '#### '+number})
        hashes.append(hashlib.sha256(row['answer'].encode()).hexdigest())
    return {'answer_projection':'canonical-numeric-final-line-v1','nll':result['nll'],'tokens':result['tokens'],
            'answers':projected,'raw_answer_sha256':hashes,'output_tokens':result['output_tokens'],
            'at_output_limit':result['at_output_limit']}


def verify_math_transition(report, evaluation, trace):
    """Raw generations must reconstruct the compact signed scoring record exactly."""
    if (not isinstance(trace,dict) or set(trace)!={'version','evaluation_artifact_id','before','after'}
            or type(trace['version']) is not int or trace['version']!=1 or trace['evaluation_artifact_id']!=ARTIFACTS['evaluation'][0]
            or len(canonical(trace))>MAX_TRACE_BYTES or digest(trace)!=report['math_trace_id']
            or len(canonical(trace))!=report['math_trace_bytes']):
        raise ValueError('Math trace does not bind its complete raw public generations')
    for side in ['before','after']:
        raw=trace[side];compact=report['math_'+side]
        if compact_evaluation(raw)!=compact or sum(score_answers(raw['answers'],evaluation).values())!=raw['correct']:
            raise ValueError('Math scoring projection differs from its original generated text')
    before={**report['math_before'],'legacy':{'nll':report['baseline_nll'],'quality':report['quality_before'],
                                             'instructions':report['instruction_before']}}
    after={**report['math_after'],'legacy':{'nll':report['candidate_nll'],'quality':report['quality_after'],
                                           'instructions':report['instruction_after']}}
    return verdict(before,after,evaluation)
