from copy import deepcopy

import pytest

from src.learning.public_quality import quality_cases,correct_answer,preservation_verdict
from src.learning.public_conversation import public_corpus


def test_paraphrases_are_disjoint_from_training_and_known_madrid_failure_is_rejected():
    rows=quality_cases();train={row['prompt'] for row in public_corpus()['train']}
    assert len(rows)==40 and len({r['prompt'] for r in rows})==40
    assert not train.intersection(r['prompt'] for r in rows)
    case=next(r for r in rows if r['case']=='france_es_brief')
    assert correct_answer(case,'La capital de Francia es París.')
    assert not correct_answer(case,'Madrid')
    assert not correct_answer(case,'Puede ser París o Madrid.')


def test_no_previous_fact_may_be_lost_and_incomplete_evaluation_cannot_promote():
    baseline=[{'case':row['case'],'answer':row['accepted'][0]} for row in quality_cases()]
    candidate=deepcopy(baseline);candidate[0]['answer']='Madrid'
    verdict=preservation_verdict(baseline,candidate)
    assert verdict['baseline_passed']==40 and verdict['candidate_passed']==39
    assert verdict['regressions']==[baseline[0]['case']] and not verdict['eligible_for_promotion']
    with pytest.raises(ValueError,match='Incomplete'):preservation_verdict(baseline,candidate[:-1])
