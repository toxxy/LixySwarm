"""Read-only frozen-worker qualification, never a model-promotion gate."""
import hashlib
import json
from pathlib import Path

BASELINE_ID='b5d9358df1d49b4d14c3991c781aef9319cf10e3c100b8233e00744b3309eea8'


def reference(path):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=BASELINE_ID:raise ValueError('Frozen-worker reference differs from its bound published evidence')
    report=json.loads(raw)
    evaluations=[report['trials'][i]['report']['candidate_result']['output']['evaluation'] for i in range(1,5)]
    fields=['math_before','quality_before','instruction_before','baseline_nll']
    if not all(all(e[k]==evaluations[0][k] for k in fields) for e in evaluations):
        raise ValueError('Cached worker baseline is not repeatable across its four original trials')
    return evaluations[0]


def qualify(legacy,math_projection,baseline):
    checks={'identical_factual_answers':legacy['quality']==baseline['quality_before'],
        'identical_instruction_answers':legacy['instructions']==baseline['instruction_before'],
        'identical_historical_nll':legacy['nll']==baseline['baseline_nll'],
        'identical_full_math_projection':math_projection==baseline['math_before']}
    return {'baseline_report_id':BASELINE_ID,'checks':checks,'unchanged_public_outputs':all(checks.values()),
        'factual_cases':40,'instruction_cases':40,'math_cases':128,'model_promotion_authorized':False,
        'scope':'Exact public generation and loss preservation of unchanged weights; no training improvement claim'}
