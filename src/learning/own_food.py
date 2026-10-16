"""Reproducible public food outside the own foundation's original training set."""
import base64
import hashlib
import json
from pathlib import Path

from .public_math import (ARTIFACTS, COUNTS, SOURCE_REPOSITORY, SOURCE_REVISION, SOURCES,
    _source_file, _source_rows, canonical, digest)

FORMAT = 'lixyswarm-own-public-food-v1'


def recipe():
    return {'version': 1, 'format': FORMAT, 'source_repository': SOURCE_REPOSITORY,
        'source_revision': SOURCE_REVISION, 'source_sha256': SOURCES['train.jsonl'][0],
        'license_sha256': SOURCES['LICENSE'][0], 'license_spdx': 'MIT',
        'excluded_foundation_training_id': ARTIFACTS['train'][0],
        'excluded_sorted_training_rows': COUNTS['train'], 'new_evaluation_documents': 128,
        'training_documents_per_program': 128, 'segment_prediction_bytes': 128,
        'selection': 'existing-canonical-case-order-after-foundation-prefix;new-evaluation-first;then-disjoint-training-programs',
        'teachers_used': False, 'private_inputs': False}


def build(source_directory):
    """Pure text preparation: no weights, metrics, identity or private sources."""
    license_text = _source_file(source_directory, 'LICENSE').decode()
    rows, train_questions = _source_rows(_source_file(source_directory, 'train.jsonl'), 'train')
    _, test_questions = _source_rows(_source_file(source_directory, 'test.jsonl'), 'test')
    if train_questions & test_questions: raise ValueError('Public training and historical evaluation overlap')
    new = rows[COUNTS['train']:]
    if len(new) <= 128: raise ValueError('No novel public training food remains')
    evaluation = new[:128]; training = new[128:]; source_id = digest(recipe())
    common = {'version': FORMAT, 'recipe_id': source_id, 'private_inputs': False,
        'license': {'spdx': 'MIT', 'text': license_text, 'sha256': SOURCES['LICENSE'][0]}}
    eval_value = {**common, 'split': 'evaluation', 'rows': evaluation}
    raw_eval = canonical(eval_value); eval_id = hashlib.sha256(raw_eval).hexdigest()
    objects = {eval_id: raw_eval}; programs = []
    predicted = sum(min(len(('Question: '+r['question']+'\nAnswer: '+r['answer']+'\n\n').encode()),512)-1 for r in evaluation)
    for index, start in enumerate(range(0, len(training), 128)):
        documents = training[start:start+128]; segments = []
        for row in documents:
            doc = ('Question: '+row['question']+'\nAnswer: '+row['answer']+'\n\n').encode()
            for offset in range(0, len(doc)-1, 128):
                segment = doc[offset:offset+129]
                segments.append({'source_case': row['case'], 'offset': offset,
                    'bytes_b64': base64.b64encode(segment).decode()})
        if not 1 <= len(segments) <= 4096: raise ValueError('Own food program exceeds its bounded curriculum')
        payload = {**common, 'split': 'train', 'program_index': index, 'rows': segments}
        raw = canonical(payload); identifier = hashlib.sha256(raw).hexdigest(); objects[identifier] = raw
        descriptor = {'version': 1, 'recipe_id': source_id, 'index': index,
            'training_dataset_id': identifier, 'training_bytes': len(raw), 'training_units': len(segments),
            'source_documents': len(documents), 'source_case_ids_sha256': digest([r['case'] for r in documents]),
            'evaluation_dataset_id': eval_id, 'evaluation_bytes': len(raw_eval),
            'evaluation_documents': 128, 'evaluation_prediction_bytes': predicted}
        descriptor['program_id'] = digest(descriptor); programs.append(descriptor)
    return {'recipe': recipe(), 'programs': programs, 'objects': objects,
        'excluded_foundation_documents': COUNTS['train'], 'novel_training_documents': len(training),
        'novel_evaluation_documents': len(evaluation)}


def load(path, descriptor, split):
    """Only app-known derived bytes; metadata cannot name arbitrary local text."""
    if split not in {'train', 'evaluation'}: raise ValueError('Unknown own food split')
    key = 'training' if split == 'train' else 'evaluation'
    path = Path(path); size = descriptor[key+'_bytes']; identifier = descriptor[key+'_dataset_id']
    if (path.is_symlink() or not path.is_file() or path.stat().st_size != size or not 0 < size <= 8*1024**2):
        raise ValueError('Own food exceeds its pinned public size')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != identifier: raise ValueError('Own food differs from its pinned public content')
    value = json.loads(raw)
    if (canonical(value) != raw or value['version'] != FORMAT or value['split'] != split
            or value['recipe_id'] != descriptor['recipe_id'] or value['private_inputs'] is not False):
        raise ValueError('Own food has incompatible provenance')
    if split == 'train':
        if value['program_index'] != descriptor['index'] or len(value['rows']) != descriptor['training_units']:
            raise ValueError('Own training food changed program or size')
        docs = []
        for row in value['rows']:
            segment = base64.b64decode(row['bytes_b64'], validate=True)
            if not 2 <= len(segment) <= 129 or type(row['offset']) is not int or row['offset'] < 0 or row['offset'] % 128:
                raise ValueError('Own training segment is invalid')
            docs.append(segment)
        return docs
    if len(value['rows']) != descriptor['evaluation_documents']:
        raise ValueError('Own evaluation food changed size')
    from .swarm_foundation_job import documents
    return documents(value)
