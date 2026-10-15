"""Fixed synthetic conversation-memory ablation, not training data."""
import hashlib
import json
import re

from src.network.work_protocol import ResultReceipt,WorkResult


def memory_cases():
    rows = [
        ('es_code', ['En este ejemplo ficticio, la clave del nodo Amapola es jaspe-642.'],
         '¿Cuál es la clave del nodo Amapola? Responde solo la clave.', 'jaspe-642'),
        ('es_label', ['En este ejemplo ficticio, la etiqueta del equipo Azucena es aguamarina-731.'],
         '¿Cuál es la etiqueta del equipo Azucena? Responde solo la etiqueta.', 'aguamarina-731'),
        ('es_location', ['En este ejemplo ficticio, la copia del proyecto Bruma está en estante-624.'],
         '¿Dónde está la copia del proyecto Bruma? Responde solo su ubicación.', 'estante-624'),
        ('es_correction', ['En este ejemplo ficticio, la clave del equipo Cedro es niebla-413.',
                           'Corrección: la clave del equipo Cedro es granito-958.'],
         '¿Cuál es la clave actual del equipo Cedro? Responde solo la clave.', 'granito-958'),
        ('en_code', ['In this fictional example, the code for the device River is cinder-593.'],
         'What is the code for the device River? Reply with only the code.', 'cinder-593'),
        ('en_location', ['In this fictional example, the backup for project Maple is at shelf-826.'],
         'Where is the backup for project Maple? Reply with only the location.', 'shelf-826'),
        ('en_label', ['In this fictional example, the label for project Willow is violet-218.'],
         'What is the label for project Willow? Reply with only the label.', 'violet-218'),
        ('en_correction', ['In this fictional example, the code for device Orchid is cobalt-365.',
                           'Correction: the code for device Orchid is copper-947.'],
         'What is the current code for device Orchid? Reply with only the code.', 'copper-947'),
    ]
    return [{'case': name, 'statements': statements, 'question': question, 'expected': expected}
            for name, statements, question, expected in rows]


def statement_messages(case):
    return [message for statement in case['statements'] for message in
            [{'role': 'user', 'content': statement}, {'role': 'assistant', 'content': 'Recorded for this fictional example.'}]]


def distractor_messages():
    return [message for i in range(101) for message in
        [{'role': 'user', 'content': f'Registro de práctica {i}: hay dos libros sobre la mesa.'},
         {'role': 'assistant', 'content': 'Registro recibido.'}]]


def case_digest():
    value = {'version': 1, 'cases': memory_cases(), 'distractors': distractor_messages(),
             'max_context_bytes': 4096, 'max_messages': 8, 'minimum_recalled_correct': 7, 'minimum_gain': 6,
             'preserve_correct_cases':True}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def verdict(results):
    cases = memory_cases()
    if not isinstance(results, list) or len(results) != len(cases):
        raise ValueError('Incomplete public memory comparison')
    before = after = 0
    regressions=[]
    jobs=set();requester=None;expert=None
    for case, row in zip(cases, results):
        if row.get('case') != case['case']:
            raise ValueError('Repeated or missing memory case')
        correct = []
        for mode in ['recent', 'memory']:
            answer = row[mode]['answer']
            if not isinstance(answer, str): raise ValueError('Invalid memory answer')
            correct.append(bool(re.fullmatch(re.escape(case['expected']) + r'[.!]?', answer.strip(), flags=re.I)))
            if not 1 <= row[mode]['context_bytes'] <= 4096 or not 1 <= row[mode]['context_messages'] <= 8:
                raise ValueError('Memory ablation exceeded the common context budget')
            result=WorkResult.from_dict(row[mode]['signed_result'])
            receipt=ResultReceipt.from_dict(result.receipt)
            if requester is None:requester=receipt.requester_node_id
            if expert is None:expert=row[mode]['model_id']
            if (result.status!='ok' or result.job_id in jobs or row[mode]['receipt']!=result.receipt
                    or result.output.get('answer')!=answer or result.output.get('model_id')!=row[mode]['model_id']
                    or row[mode]['model_id']!=expert or not row[mode]['context_sent'] or row[mode]['route']!='network'
                    or not receipt.verify(result,expected_worker_id=row[mode]['provider_id'],expected_requester_id=requester)):
                raise ValueError('Memory ablation contains an altered, repeated or unsigned answer')
            jobs.add(result.job_id)
        before += correct[0]
        after += correct[1]
        if correct[0] and not correct[1]:regressions.append(case['case'])
        if (row['recent']['model_id'] != row['memory']['model_id']
                or row['recent']['provider_id'] != row['memory']['provider_id']
                or row['recent']['receipt']['job_id'] == row['memory']['receipt']['job_id']
                or not row['fresh_process']['success'] or not row['fresh_process']['restored_in_fresh_process']
                or row['recent']['recalled_episodes']!=0 or row['memory']['recalled_episodes'] < 1):
            raise ValueError('Memory comparison changed the expert, reused work or omitted persistence')
    return {'case_digest': case_digest(), 'total': len(cases), 'recent_correct': before,
            'memory_correct': after, 'gain': after - before,'regressions':regressions,
            'eligible': after >= 7 and after - before >= 6 and not regressions,
            'scope': 'Synthetic episodic recall and correction with one unchanged expert; not a general memory or neural-routing claim'}
