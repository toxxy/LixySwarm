"""Held-out public factual paraphrases for preservation before model promotion.

Literal synthetic questions only. No conversation, memory or files are read.
These bounded checks detect factual regression, not general language quality.
"""
import hashlib
import json
import re
import unicodedata


FACTS=[
    ('france','Francia','France','paris',('madrid','berlin','rome','roma','tokyo')),
    ('germany','Alemania','Germany','berlin',('paris','madrid','rome','roma','tokyo')),
    ('italy','Italia','Italy','rome',('paris','madrid','berlin','tokyo')),
    ('spain','España','Spain','madrid',('paris','berlin','rome','roma','tokyo')),
    ('japan','Japón','Japan','tokyo',('paris','madrid','berlin','rome','roma')),
]


def quality_cases():
    rows=[]
    for key,es,en,expected,rejected in FACTS:
        variants=[
            ('es_brief',f'¿Cuál es la capital de {es}? Responde brevemente en español.'),
            ('es_city',f'Dime el nombre de la ciudad que es capital de {es}.'),
            ('es_only',f'Escribe únicamente la capital de {es}, sin explicación.'),
            ('es_question',f'Necesito un dato geográfico: ¿qué ciudad es capital de {es}?'),
            ('en_brief',f'What is the capital of {en}? Answer briefly in English.'),
            ('en_city',f'Name the city that is the capital of {en}.'),
            ('en_only',f'Write only the capital of {en}, without explanation.'),
            ('en_question',f'A geography question: which city is the capital of {en}?'),
        ]
        for variant,prompt in variants:
            accepted=(expected,'roma') if key=='italy' else (expected,'tokio') if key=='japan' else (expected,)
            rows.append({'case':f'{key}_{variant}','prompt':prompt,'accepted':list(accepted),'rejected':list(rejected)})
    return rows


def digest_cases():
    return hashlib.sha256(json.dumps(quality_cases(),sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def correct_answer(case,answer):
    if not isinstance(answer,str) or not answer:return False
    folded=''.join(c for c in unicodedata.normalize('NFKD',answer.casefold()) if not unicodedata.combining(c))
    words=set(re.findall(r'\b[a-z]+\b',folded))
    return bool(words.intersection(case['accepted'])) and not words.intersection(case['rejected'])


def preservation_verdict(baseline,candidate):
    expected={row['case'] for row in quality_cases()}
    def index(rows):
        if len(rows)!=len(expected) or {r['case'] for r in rows}!=expected:raise ValueError('Incomplete or duplicate quality cases')
        cases={row['case']:row for row in quality_cases()}
        return {r['case']:correct_answer(cases[r['case']],r['answer']) for r in rows}
    before=index(baseline);after=index(candidate)
    regressions=sorted(key for key in expected if before[key] and not after[key])
    return {'case_digest':digest_cases(),'total':len(expected),'baseline_passed':sum(before.values()),
            'candidate_passed':sum(after.values()),'regressions':regressions,
            'eligible_for_promotion':not regressions and all(after.values()),
            'scope':'Factual paraphrase preservation only; not a general conversation score'}
