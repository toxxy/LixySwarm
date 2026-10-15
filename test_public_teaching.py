from copy import deepcopy
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from src.contribution import ResourceRequirements
from src.learning.public_teaching import (VERSION, canonical, checked_labels, digest,
    public_teaching_corpus, teaching_prompt, transfer_verdict, verify_lessons, verify_student_report)
from src.network.lsp import LSPIdentity
from src.network.work_protocol import ResultReceipt, WorkResult, WorkUnit


@pytest.fixture
def lesson():
    teacher, owner = LSPIdentity.generate(), LSPIdentity.generate()
    options = dict(teacher_model_id='a' * 64, teacher_release_id='b' * 64,
                   teacher_worker_id=teacher.node_id_hex, requester_id=owner.node_id_hex)
    value = dict(version=VERSION, dataset_id=digest(public_teaching_corpus()), batches=[], **options)
    for start in range(0, 64, 4):
        work = WorkUnit.create(origin_node_id=owner.node_id_hex, operation='language.inference.v1',
            kind='inference', requirements=ResourceRequirements(kind='inference', gpu_required=True, ram_gb=5),
            payload={'model_artifact_id': 'a' * 64, 'messages': [{'role': 'user', 'content': teaching_prompt(start)}]})
        answers = {str(i): r['answer'] for i, r in enumerate(public_teaching_corpus()['train'][start:start + 4])}
        result = WorkResult(job_id=work.job_id, status='ok', output={
            'model_id': 'a' * 64, 'answer': json.dumps(answers, ensure_ascii=False), 'output_tokens': 48})
        result = WorkResult(**{**result.to_dict(), 'receipt': ResultReceipt.create(result, teacher, owner.node_id_hex).to_dict()})
        value['batches'].append({'start': start, 'offer': work.to_dict(), 'result': result.to_dict()})
    return value, options, teacher


def test_public_splits_are_disjoint_and_bounded():
    corpus = public_teaching_corpus()
    assert len(corpus['train']) == 64 and len(corpus['evaluation']) == 32
    assert {r['prompt'] for r in corpus['train']}.isdisjoint(r['prompt'] for r in corpus['evaluation'])
    assert {r['answer'] for r in corpus['train'] if r['kind'] == 'arithmetic'}.isdisjoint(
        r['answer'] for r in corpus['evaluation'] if r['kind'] == 'arithmetic')
    for start in range(0, 64, 4):
        prompt = teaching_prompt(start)
        assert len(prompt.encode()) <= 4096
        assert all(r['prompt'] not in prompt for r in corpus['evaluation'])
    for start in [-4, 1, 64, True]:
        with pytest.raises(ValueError): teaching_prompt(start)


def test_valid_archived_lessons_and_old_receipts(lesson):
    value, options, teacher = lesson
    for row in value['batches']:
        offer = row['offer']
        offer['created_at'] -= 7200
        offer['deadline'] -= 7200
        offer['job_id'] = digest({k: v for k, v in offer.items() if k != 'job_id'})
        original = row['result']
        result = WorkResult(**{**original, 'job_id': offer['job_id'], 'finished_at': original['finished_at'] - 7200, 'receipt': {}})
        row['result'] = {**result.to_dict(), 'receipt': ResultReceipt.create(result, teacher, options['requester_id']).to_dict()}
    checked = verify_lessons(value, **options)
    assert len(checked['accepted']) == 64 and not checked['rejected']


@pytest.mark.parametrize('change', [
    lambda v: v.update(private_memory='not allowed'),
    lambda v: v.update(teacher_model_id='c' * 64),
    lambda v: v.update(teacher_release_id='c' * 64),
    lambda v: v.update(dataset_id='c' * 64),
    lambda v: v['batches'].pop(),
    lambda v: v['batches'].__setitem__(1, deepcopy(v['batches'][0])),
    lambda v: v['batches'][0]['offer']['payload']['messages'][0].update(content='private replacement'),
    lambda v: v['batches'][0]['offer'].update(operation='arbitrary.program'),
    lambda v: v['batches'][0]['offer'].update(deadline=v['batches'][0]['offer']['created_at'] - 1),
    lambda v: v['batches'][0]['result']['output'].update(answer='{"0":"changed"}'),
    lambda v: v['batches'][0]['result']['receipt'].update(requester_node_id='c' * 64),
])
def test_reject_forged_or_private_lessons(lesson, change):
    value, options, _ = lesson
    change(value)
    with pytest.raises(ValueError): verify_lessons(value, **options)


def test_signed_wrong_label_is_discarded_not_replaced(lesson):
    value, options, teacher = lesson
    result = WorkResult.from_dict(value['batches'][0]['result'])
    answers = json.loads(result.output['answer'])
    answers['0'] = 'wrong'
    changed = WorkResult(**{**result.to_dict(), 'output': {**result.output, 'answer': json.dumps(answers)}, 'receipt': {}})
    value['batches'][0]['result'] = {**changed.to_dict(), 'receipt': ResultReceipt.create(changed, teacher, options['requester_id']).to_dict()}
    checked = verify_lessons(value, **options)
    assert len(checked['accepted']) == 63
    assert checked['rejected'] == [{'index': 0, 'reason': 'teacher_label_failed_public_oracle'}]
    assert all(r['index'] != 0 for r in checked['accepted'])


@pytest.mark.parametrize('answer', ['{"0":"a","0":"b","1":"c","2":"d","3":"e"}', '[]', '```json\n{}\n```'])
def test_malformed_labels_never_supply_targets(answer):
    accepted, rejected = checked_labels(answer, 0)
    assert not accepted and len(rejected) == 4


def evaluation(correct, nll):
    rows = public_teaching_corpus()['evaluation']
    return {'nll': nll, 'tokens': 128,
            'results': [{'index': i, 'answer': row['answer'] if i in correct else 'wrong'} for i, row in enumerate(rows)]}


def test_transfer_requires_new_accuracy_loss_improvement_and_no_regression():
    before = evaluation(set(range(12)), 1)
    assert transfer_verdict(before, evaluation(set(range(16)), .98))['eligible']
    assert not transfer_verdict(before, evaluation(set(range(16)), .999))['eligible']
    assert not transfer_verdict(before, evaluation(set(range(15)), .9))['eligible']
    assert not transfer_verdict(before, evaluation(set(range(1, 18)), .9))['eligible']
    for malformed in [evaluation(set(), float('nan')), {**before, 'results': before['results'][1:]},
                      {**before, 'tokens': 129}]:
        with pytest.raises(ValueError): transfer_verdict(before, malformed)


def archived_physical_report():
    return json.loads((Path(__file__).parent / 'experiments/reports/language_cross_size_transfer_20261014.json').read_text())


def test_physical_report_recomputes_without_tensor_engine():
    report = archived_physical_report()
    verdict = verify_student_report(report['student'], report['lessons'])
    assert verdict['transfer_verdict']['candidate_passed'] == 25
    assert not verdict['eligible_for_research_transfer']


@pytest.mark.parametrize('change', [
    lambda r: r.update(eligible_for_research_transfer=True),
    lambda r: r['policy'].update(minimum_new_correct=3),
    lambda r: r.update(published=True),
    lambda r: r.update(lessons_id='0' * 64),
    lambda r: r.update(incorporated_tokens=250),
    lambda r: [row.update(answer='Wrong city') for row in r['candidate']['historical']['quality']],
])
def test_archived_transfer_cannot_be_relabelled_as_accepted(change):
    archived = archived_physical_report()
    change(archived['student'])
    with pytest.raises(ValueError): verify_student_report(archived['student'], archived['lessons'])


def test_child_request_checks_public_lessons_before_importing_tensors(lesson, tmp_path):
    from src.learning.language_distillation_job import validate_request
    value, options, _ = lesson
    path = tmp_path / 'lessons.json'
    path.write_bytes(canonical(value))
    request = {'action': 'distill-public-v1', 'base_directory': str(tmp_path / 'public-base'),
        'lessons_file': str(path), 'output_directory': str(tmp_path / 'candidate'), 'teacher': options,
        'deadline': time.time() + 60, 'cpu_threads': 4, 'gpu_fraction': .5}
    assert len(validate_request(request)[1]['accepted']) == 64
    for bad in [{**request, 'cpu_threads': 100}, {**request, 'gpu_fraction': 2},
                {**request, 'deadline': time.time() - 1}, {**request, 'personal_memory': str(tmp_path)},
                {**request, 'teacher': {**options, 'teacher_model_id': 'f' * 64}}]:
        with pytest.raises(ValueError): validate_request(bad)


@pytest.mark.parametrize('mode', ['ready', 'permission_withdrawn', 'native_busy', 'battery'])
def test_parent_finalizes_durable_evaluation_without_retraining(monkeypatch, tmp_path, capsys, mode):
    from experiments import distill_public_language as probe
    from src.contribution import ContributionPolicy
    from src.network.client_state import ClientSettings
    archived = archived_physical_report()
    original = archived['student']
    directory, node = tmp_path / 'trial', tmp_path / 'node'
    directory.mkdir()
    node.mkdir()
    controls = ClientSettings(policy=ContributionPolicy.for_mode('maximum'))
    controls.paused = mode == 'permission_withdrawn'
    controls.save(node)
    (node / 'diagnostics.json').write_text(json.dumps({'node_id': original['physical_student'],
        'language_training': {'active': int(mode == 'native_busy')}, 'language': {'active': 0}}))
    lessons = tmp_path / 'lessons.json'
    lessons.write_bytes(canonical(archived['lessons']))
    head = {'release': {'model_artifact_id': original['teacher']['teacher_model_id'],
                         'release_id': original['teacher']['teacher_release_id']}}
    monkeypatch.setattr(probe, 'approved_teacher', lambda _: (head, []))
    monkeypatch.setattr(probe.ResourceGovernor, '_system_busy', lambda _: mode == 'battery')
    monkeypatch.setattr(probe, 'materialize_family', lambda *a, **kw: tmp_path / 'public-base')
    import src.network.public_knowledge as knowledge
    monkeypatch.setattr(knowledge, 'managed_public_bytes', lambda _: 0)
    engine_path = tmp_path / 'venv' / 'python'
    calls = []
    def replay(_self, command, request, execution, **kwargs):
        calls.append(command)
        assert command[0] == str(engine_path.absolute())
        assert kwargs['allowed']() and kwargs['ram_gb'] == 6
        return deepcopy(original)
    monkeypatch.setattr(probe.ComputeSupervisor, 'run', replay)
    args = SimpleNamespace(node_directory=node, directory=directory, lessons=lessons, engine_python=engine_path)
    if mode != 'ready':
        with pytest.raises(ValueError): probe.train(args, threading.Event())
        assert not calls
        return
    probe.train(args, threading.Event())
    assert len(calls) == 1
    assert json.loads(capsys.readouterr().out) == {'success': True, 'before': 22, 'after': 25,
                                                 'eligible_for_research_transfer': False}
    assert json.loads((directory / 'status.json').read_text())['phase'] == 'student_complete'
    assert json.loads((directory / 'report.json').read_text())['controls_and_native_progress_unchanged']
