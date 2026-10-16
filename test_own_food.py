"""Food selection is deterministic, public and separate from own pretraining."""
import base64
from collections import defaultdict
import json

import pytest

from src.learning import own_food
from src.learning.public_math import _source_rows


def source_fixture(monkeypatch):
    rows = [{'question': f'Public problem {i}?', 'answer': 'A public explanation. '*30+'\n#### 1'} for i in range(4300)]
    train = ('\n'.join(json.dumps(r) for r in rows)+'\n').encode()
    test = (json.dumps({'question': 'Separate historical test?', 'answer': 'One.\n#### 1'})+'\n').encode()
    sources = {'LICENSE': b'MIT public fixture', 'train.jsonl': train, 'test.jsonl': test}
    monkeypatch.setattr(own_food, '_source_file', lambda _directory, name: sources[name])
    return sources


def test_novel_programs_exclude_foundation_and_evaluation_and_cover_full_answers(tmp_path, monkeypatch):
    sources = source_fixture(monkeypatch); built = own_food.build(tmp_path)
    assert built['novel_training_documents'] == 76 and len(built['programs']) == 1
    sorted_rows, _ = _source_rows(sources['train.jsonl'], 'train')
    foundation = {r['case'] for r in sorted_rows[:4096]}
    descriptor = built['programs'][0]
    training = json.loads(built['objects'][descriptor['training_dataset_id']])
    evaluation = json.loads(built['objects'][descriptor['evaluation_dataset_id']])
    training_cases = {r['source_case'] for r in training['rows']}; evaluation_cases = {r['case'] for r in evaluation['rows']}
    assert not training_cases & (foundation | evaluation_cases) and not foundation & evaluation_cases
    reconstructed = defaultdict(bytes)
    for row in training['rows']:
        raw = base64.b64decode(row['bytes_b64'])
        reconstructed[row['source_case']] += raw if row['offset'] == 0 else raw[1:]
    expected = {r['case']: ('Question: '+r['question']+'\nAnswer: '+r['answer']+'\n\n').encode() for r in sorted_rows[4224:]}
    assert dict(reconstructed) == expected
    assert all(len(raw)>129 and raw.endswith(b'#### 1\n\n') for raw in reconstructed.values())
    path = tmp_path/'training.json'; path.write_bytes(built['objects'][descriptor['training_dataset_id']])
    docs = own_food.load(path, descriptor, 'train')
    assert len(docs) == descriptor['training_units'] and all(2 <= len(v) <= 129 for v in docs)
    assert own_food.build(tmp_path)['programs'] == built['programs']


def test_tampered_public_food_and_split_overlap_are_rejected(tmp_path, monkeypatch):
    sources = source_fixture(monkeypatch); built = own_food.build(tmp_path); descriptor = built['programs'][0]
    path = tmp_path/'training.json'; raw = built['objects'][descriptor['training_dataset_id']]
    path.write_bytes(raw[:-1]+b' ')
    with pytest.raises(ValueError, match='content'): own_food.load(path, descriptor, 'train')
    sources['test.jsonl'] = (json.dumps({'question': 'Public problem 0?', 'answer': 'One.\n#### 1'})+'\n').encode()
    with pytest.raises(ValueError, match='overlap'): own_food.build(tmp_path)
