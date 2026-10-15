"""Public, deterministic lessons for transfer between different-sized experts.

The oracle checks a teacher's labels; it never supplies a replacement label.
Only training prompts leave the student. Evaluation uses different templates
and arithmetic values; it is never sent to the teacher or used for updates.
"""
import hashlib
import json
import math
import re

from src.contribution import ResourceRequirements
from src.network.work_protocol import ResultReceipt, WorkResult

VERSION = 'lixyswarm-public-teaching-v1'
BATCH_SIZE = 4
MAX_LESSON_BYTES = 512 * 1024
TRANSFER_POLICY = {'version': 1, 'steps': 16, 'accumulation': 4, 'learning_rate': .00005,
                   'retention_weight': 1, 'clip_norm': 1, 'minimum_labels': 48,
                   'minimum_new_correct': 4, 'minimum_relative_nll_improvement': .01,
                   'preserve_all_correct_heldout_answers': True,
                   'existing_factual_and_instruction_gates_required': True}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def public_teaching_corpus():
    train, evaluation = [], []
    words = ['bronce', 'cedro', 'cobalto', 'coral', 'duna', 'espiga', 'faro', 'helecho',
             'junco', 'lino', 'musgo', 'nácar', 'olmo', 'pino', 'roca', 'sauce',
             'trébol', 'vidrio', 'zafiro', 'ámbar', 'abeto', 'cerezo', 'dalía', 'ébano']
    for split, indices in [(train, range(16)), (evaluation, range(16, 24))]:
        heldout = split is evaluation
        for i in indices:
            word, other, n = words[i], words[(i + 7) % len(words)], i + 47
            data = json.dumps({'task': {'label': word, 'ignored': other}, 'count': n}, ensure_ascii=False)
            prompts = [
                ('extraction', f'Del objeto {data}, escribe solo task.label; sin comillas.' if not heldout else
                 f'Lee {data}. ¿Qué texto guarda label dentro de task? Devuelve únicamente ese texto.', word),
                ('arithmetic', f'Calcula {n} menos 7. Contesta únicamente el número.' if not heldout else
                 f'Una caja contiene {n} piezas y se retiran 7. ¿Cuántas quedan? Escribe solo la cantidad.', str(n - 7)),
                ('routing', f'Datos ficticios: {word} almacena copias; {other} calcula gradientes. Escribe solo quién calcula gradientes.' if not heldout else
                 f'En este caso inventado, el equipo que calcula es {other} y el que guarda réplicas es {word}. Devuelve únicamente el nombre del equipo que calcula.', other),
                ('format', f'Escribe {word} y {other} separados por una coma, sin espacios ni otro texto.' if not heldout else
                 f'Forma una respuesta con estas palabras en este orden: {word}; {other}. Usa una coma entre ellas y elimina todos los espacios.', word + ',' + other),
            ]
            for kind, prompt, answer in prompts:
                split.append({'kind': kind, 'prompt': prompt, 'answer': answer})
    if len({r['prompt'] for r in train + evaluation}) != len(train + evaluation):
        raise ValueError('Public teaching inputs overlap')
    return {'version': VERSION, 'scope': 'public-synthetic-only', 'train': train, 'evaluation': evaluation}


def teaching_prompt(start, count=BATCH_SIZE):
    corpus = public_teaching_corpus()
    if (type(start) is not int or type(count) is not int or count != BATCH_SIZE
            or start % BATCH_SIZE or not 0 <= start <= len(corpus['train']) - count):
        raise ValueError('Only fixed public training batches may be taught')
    questions = {str(i): row['prompt'] for i, row in enumerate(corpus['train'][start:start + count])}
    return ('Resuelve las cuatro instrucciones públicas siguientes. Devuelve exclusivamente un objeto JSON '
            'con las claves "0", "1", "2", "3" y una respuesta de texto por clave. '
            'No incluyas explicaciones ni bloques de código. Instrucciones: ' + canonical(questions).decode())


def checked_labels(answer, start):
    """Incorrect or malformed teacher output is discarded, never oracle-filled."""
    rows = public_teaching_corpus()['train'][start:start + BATCH_SIZE]
    try:
        def pairs(items):
            if len({key for key, _ in items}) != len(items):
                raise ValueError('Duplicate teacher JSON keys')
            return dict(items)
        value = json.loads(answer, object_pairs_hook=pairs)
    except (ValueError, TypeError):
        value = None
    if not isinstance(value, dict) or set(value) != {'0', '1', '2', '3'}:
        return [], [{'index': start + i, 'reason': 'invalid_teacher_json'} for i in range(BATCH_SIZE)]
    accepted, rejected = [], []
    for i, row in enumerate(rows):
        label = value[str(i)]
        if not isinstance(label, str) or label != row['answer']:
            rejected.append({'index': start + i, 'reason': 'teacher_label_failed_public_oracle'})
        else:
            accepted.append({'index': start + i, 'kind': row['kind'], 'prompt': row['prompt'], 'answer': label})
    return accepted, rejected


def transfer_verdict(before, after):
    """Fixed unseen-template screening; passing is not a general language claim."""
    expected = public_teaching_corpus()['evaluation']
    def checked(value):
        if (not isinstance(value, dict) or set(value) != {'nll', 'tokens', 'results'}
                or type(value['nll']) not in (int, float) or not math.isfinite(value['nll']) or value['nll'] <= 0
                or type(value['tokens']) is not int or value['tokens'] < 32
                or not isinstance(value['results'], list) or len(value['results']) != len(expected)):
            raise ValueError('Incomplete transfer evaluation')
        correct = set()
        for i, row in enumerate(value['results']):
            if (not isinstance(row, dict) or set(row) != {'index', 'answer'} or type(row['index']) is not int
                    or row['index'] != i or not isinstance(row['answer'], str) or len(row['answer'].encode()) > 4096):
                raise ValueError('Transfer evaluation omits or repeats a held-out input')
            if row['answer'] == expected[i]['answer']:
                correct.add(i)
        return correct
    baseline, candidate = checked(before), checked(after)
    if before['tokens'] != after['tokens']:
        raise ValueError('Paired transfer evaluations must have the same targets')
    regressions = sorted(baseline - candidate)
    nll_improved = after['nll'] < before['nll'] * (1 - TRANSFER_POLICY['minimum_relative_nll_improvement'])
    return {'total': len(expected), 'baseline_passed': len(baseline), 'candidate_passed': len(candidate),
            'regressions': regressions, 'nll_improved': nll_improved,
            'eligible': nll_improved and not regressions
            and len(candidate) >= len(baseline) + TRANSFER_POLICY['minimum_new_correct']}


def transfer_policy():
    from .public_quality import digest_cases as factual_digest
    from .public_language_gate import digest_cases as instruction_digest
    return {**TRANSFER_POLICY, 'dataset_id': digest(public_teaching_corpus()),
            'factual_gate_id': factual_digest(), 'instruction_gate_id': instruction_digest(),
            'retention': 'exact-small-parent-public-anchor-probabilities', 'seed': 20261014}


def verify_student_report(report, lessons):
    """Recompute a local research verdict; this does not authorize publication."""
    from .public_bases import base_hash
    from .public_quality import preservation_verdict as factual_verdict
    from .public_language_gate import preservation_verdict as instruction_verdict
    from src.release.language_catalog import INITIAL_BUNDLES
    options = {k: lessons[k] for k in ['teacher_model_id', 'teacher_release_id', 'teacher_worker_id', 'requester_id']}
    checked = verify_lessons(lessons, **options)
    before, after = report['baseline'], report['candidate']
    transfer = transfer_verdict(before['transfer'], after['transfer'])
    factual = factual_verdict(before['historical']['quality'], after['historical']['quality'])
    instruction = instruction_verdict(before['historical']['instructions'], after['historical']['instructions'])
    eligible = transfer['eligible'] and factual['eligible_for_promotion'] and instruction['eligible_for_promotion']
    policy = transfer_policy()
    expected = {'success': True, 'base_name': 'qwen3-0.6b', 'base_hash': base_hash('qwen3-0.6b'),
        'student_parent_adapter_id': INITIAL_BUNDLES['qwen3-0.6b']['release']['model_artifact_id'],
        'student_parent_was_chat_approved': False, 'teacher': options, 'lessons_id': digest(lessons),
        'policy': policy, 'policy_id': digest(policy), 'accepted_teacher_labels': len(checked['accepted']),
        'training_unique_examples': len(checked['accepted']), 'training_presentations': 64,
        'rejected_teacher_labels': checked['rejected'], 'transfer_verdict': transfer,
        'factual_verdict': factual, 'instruction_verdict': instruction,
        'eligible_for_research_transfer': eligible, 'candidate_weights_retained': bool(eligible),
        'published': False, 'private_inputs': False, 'native_coordinator_cycles_added': 0, 'incorporated_tokens': 0}
    if any(report.get(k) != v for k, v in expected.items()):
        raise ValueError('Student research report has an unbound verdict or policy')
    if (not isinstance(report.get('candidate_adapter_id'), str)
            or not re.fullmatch('[0-9a-f]{64}', report['candidate_adapter_id'])
            or type(report.get('candidate_adapter_bytes')) is not int
            or not 1 <= report['candidate_adapter_bytes'] <= 32 * 1024**2
            or not isinstance(report.get('losses'), list) or len(report['losses']) != policy['steps']):
        raise ValueError('Incomplete bounded student candidate')
    return {'transfer_verdict': transfer, 'factual_verdict': factual,
            'instruction_verdict': instruction, 'eligible_for_research_transfer': eligible}


def verify_lessons(value, *, teacher_model_id, teacher_release_id, teacher_worker_id, requester_id):
    """Verify archived offers and signed answers, including expired past work.

    The caller supplies its already verified accepted teacher version. A signed
    answer alone cannot bind a prompt: the archived offer must hash to the same
    job ID. This verifier never treats a transport receipt as model approval.
    """
    fields = {'version', 'dataset_id', 'teacher_model_id', 'teacher_release_id',
              'teacher_worker_id', 'requester_id', 'batches'}
    if not isinstance(value, dict) or set(value) != fields or len(canonical(value)) > MAX_LESSON_BYTES:
        raise ValueError('Invalid bounded public lessons')
    for name, expected in [('version', VERSION), ('dataset_id', digest(public_teaching_corpus())),
                           ('teacher_model_id', teacher_model_id), ('teacher_release_id', teacher_release_id),
                           ('teacher_worker_id', teacher_worker_id), ('requester_id', requester_id)]:
        if value[name] != expected:
            raise ValueError('Lessons have another dataset, teacher or owner')
    for identifier in [teacher_model_id, teacher_release_id, teacher_worker_id, requester_id]:
        if not isinstance(identifier, str) or not re.fullmatch('[0-9a-f]{64}', identifier):
            raise ValueError('Invalid lesson identity')
    batches = value['batches']
    if not isinstance(batches, list) or len(batches) != 16:
        raise ValueError('Public lessons require the complete fixed training schedule')
    jobs, accepted, rejected = set(), [], []
    work_fields = {'job_id', 'origin_node_id', 'operation', 'kind', 'requirements', 'payload',
                   'created_at', 'deadline', 'nonce'}
    for expected_start, batch in zip(range(0, 64, BATCH_SIZE), batches):
        if not isinstance(batch, dict) or set(batch) != {'start', 'offer', 'result'} or batch['start'] != expected_start:
            raise ValueError('Lessons repeat, omit or reorder public training examples')
        offer = batch['offer']
        if not isinstance(offer, dict) or set(offer) != work_fields:
            raise ValueError('Missing archived teaching offer')
        payload = {'model_artifact_id': teacher_model_id,
                   'messages': [{'role': 'user', 'content': teaching_prompt(expected_start)}]}
        if (offer['origin_node_id'] != requester_id or offer['operation'] != 'language.inference.v1'
                or offer['kind'] != 'inference' or offer['payload'] != payload
                or offer['requirements'] != ResourceRequirements(kind='inference', gpu_required=True, ram_gb=5).__dict__
                or not isinstance(offer['nonce'], str) or not re.fullmatch('[0-9a-f]{32}', offer['nonce'])
                or digest({k: v for k, v in offer.items() if k != 'job_id'}) != offer['job_id']):
            raise ValueError('Teaching offer does not bind the exact public prompt')
        result = WorkResult.from_dict(batch['result'])
        output = result.output
        if (result.job_id != offer['job_id'] or result.job_id in jobs or result.status != 'ok' or result.error
                or set(output) != {'model_id', 'answer', 'output_tokens'} or output['model_id'] != teacher_model_id
                or not isinstance(output['answer'], str) or not 1 <= len(output['answer'].encode()) <= 8192
                or type(output['output_tokens']) is not int or not 1 <= output['output_tokens'] <= 128
                or any(type(t) not in (int, float) or not math.isfinite(t)
                       for t in [offer['created_at'], offer['deadline'], result.finished_at])
                or not offer['created_at'] <= result.finished_at <= offer['deadline']
                or not 1 <= offer['deadline'] - offer['created_at'] <= 3600
                or not ResultReceipt.from_dict(result.receipt).verify(result,
                    expected_worker_id=teacher_worker_id, expected_requester_id=requester_id)):
            raise ValueError('Unbound, expired-at-execution, repeated or unsigned teaching result')
        jobs.add(result.job_id)
        good, bad = checked_labels(output['answer'], expected_start)
        accepted.extend(good)
        rejected.extend(bad)
    return {'dataset_id': value['dataset_id'], 'accepted': accepted, 'rejected': rejected,
            'teacher_model_id': teacher_model_id, 'teacher_release_id': teacher_release_id,
            'teacher_worker_id': teacher_worker_id, 'requester_id': requester_id}
