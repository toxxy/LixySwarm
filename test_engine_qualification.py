from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest

from src.learning.engine_qualification import reference,qualify


def test_bound_public_reference_and_exact_outputs_never_promote_a_model():
    baseline=reference(Path(__file__).parent/'experiments/reports/language_math_native_scaling_20261015.json')
    legacy={'nll':baseline['baseline_nll'],'quality':deepcopy(baseline['quality_before']),
        'instructions':deepcopy(baseline['instruction_before'])}
    math=deepcopy(baseline['math_before'])
    assert qualify(legacy,math,baseline)['unchanged_public_outputs']
    assert not qualify(legacy,math,baseline)['model_promotion_authorized']
    math['raw_answer_sha256'][0]='0'*64
    assert not qualify(legacy,math,baseline)['unchanged_public_outputs']
    legacy['nll']+=1e-12
    assert not qualify(legacy,baseline['math_before'],baseline)['checks']['identical_historical_nll']
    legacy['quality'][0]['answer']='changed public answer'
    assert not qualify(legacy,baseline['math_before'],baseline)['checks']['identical_factual_answers']


def test_reference_cannot_be_replaced_by_a_peer_report(tmp_path):
    path=tmp_path/'untrusted.json';path.write_text('{}')
    with pytest.raises(ValueError,match='bound published'):reference(path)


def test_artifact_report_io_is_standalone_and_preserves_previous_complete_record(tmp_path):
    from src.artifact_io import atomic_json
    path=tmp_path/'report.json';atomic_json(path,{'success':True});before=path.read_bytes()
    with pytest.raises(ValueError):atomic_json(path,{'loss':float('nan')})
    assert path.read_bytes()==before and not list(tmp_path.glob('.report.json.*'))
    subprocess.run([sys.executable,'-c',"import sys;from src.artifact_io import atomic_json;assert 'torch' not in sys.modules;assert not any(k.startswith('src.network') for k in sys.modules)"],check=True)
