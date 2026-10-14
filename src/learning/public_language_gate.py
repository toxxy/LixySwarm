"""Fixed public instruction/context screening; keyword checks have limited scope."""
import hashlib
import json
import re

SEED_CASES = [('es_addition', 'Responde únicamente con el resultado: ¿cuánto es 2 + 2?', '^4[.!]?$'),
 ('en_multiplication', 'Reply with just the number: what is 7 times 8?', '^56[.!]?$'),
 ('es_translation',
  'Traduce al inglés únicamente esta frase: El gato está en la casa.',
  '(?i)^(the )?cat is (in|inside) the house[.!]?$'),
 ('en_translation', 'Translate into Spanish, without explanation: Good morning.', '(?i)^buenos días[.!]?$'),
 ('es_instruction', 'Escribe exactamente la palabra conectada y nada más.', '^conectada[.!]?$'),
 ('en_instruction', 'Write exactly the word ready and nothing else.', '^ready[.!]?$'),
 ('es_fact', '¿Cuál es la capital de Francia? Responde en una oración breve en español.', '(?i)par[íi]s'),
 ('en_fact', 'What is the chemical formula of water? Reply with only the formula.', '^H[₂2]O[.!]?$'),
 ('es_order',
  'Ordena estos números de menor a mayor. Responde solo con la lista: 9, 2, 5.',
  '^2\\s*,\\s*5\\s*,\\s*9[.]?$'),
 ('en_extract',
  'Given this JSON: {"node": "ant", "status": "online"}. Return only the value of status.',
  '^online[.!]?$'),
 ('es_public_context',
  'Usa solo estos datos: El VPS de LixySwarm sirve de relay y descubrimiento. No entrena modelos. ¿El VPS '
  'entrena? Responde sí o no.',
  '(?i)^no[.!]?$'),
 ('en_public_context',
  'Use these facts: an ant is a physical computer; experts are models hosted on it. Can one ant host two '
  'experts? Answer yes or no.',
  '(?i)^yes[.!]?$')]

DIALOGUE_CASES = [('es_germany', '¿Cuál es la capital de Alemania? Contesta brevemente.', '(?i)berl[íi]n'),
 ('en_italy', 'What is the capital of Italy? Keep it brief.', '(?i)rome'),
 ('es_spain', '¿Cuál es la capital de España?', '(?i)madrid'),
 ('en_japan', 'What is the capital of Japan?', '(?i)tokyo'),
 ('es_week', '¿Cuántos días tiene una semana? Responde con un número.', '^7[.!]?$'),
 ('en_triangle', 'How many sides does a triangle have? Reply with a number.', '^3[.!]?$'),
 ('es_subtraction', 'Da solo el resultado de 15 menos 6.', '^9[.!]?$'),
 ('en_division', 'Reply only with the result of 81 divided by 9.', '^9[.!]?$'),
 ('es_translate', 'Traduce al inglés sin explicación: Gracias por ayudarme.', '(?i)thank.*help'),
 ('en_translate', 'Translate into Spanish with no explanation: I like learning.', '(?i)me gusta aprender'),
 ('es_definition',
  'Explica en español y en una oración qué es una red descentralizada.',
  '(?i)(nodo|equipo|participante|central)'),
 ('en_definition',
  'Explain a decentralized network in one English sentence.',
  '(?i)(node|computer|central|participant)'),
 ('es_replicas',
  'Una red tiene copias del mismo modelo en tres equipos. Si uno se apaga, ¿se pierde necesariamente el '
  'modelo? Responde y explica brevemente.',
  '(?i)(no|otras copias|otros equipos)'),
 ('en_replicas',
  'Three computers have copies of a model. If one shuts down, is the model necessarily lost? Give a short '
  'explanation.',
  '(?i)(no|other cop|remaining)'),
 ('es_privacy',
  'Datos: la memoria personal se guarda solo en el equipo del usuario. ¿Debe enviarse automáticamente a '
  'otros nodos? Responde sí o no.',
  '(?i)^no[.!]?$'),
 ('en_privacy',
  "Facts: personal memory stays on the user's own computer. Should it automatically be sent to other nodes? "
  'Answer yes or no.',
  '(?i)^no[.!]?$'),
 ('es_role',
  'Datos: Aurora solo transporta mensajes; Boreal calcula gradientes. ¿Cuál calcula gradientes? Escribe solo '
  'su nombre.',
  '^Boreal[.!]?$'),
 ('en_role',
  'Facts: Aurora only forwards messages; Boreal computes gradients. Which computes gradients? Write only its '
  'name.',
  '^Boreal[.!]?$'),
 ('es_json', 'Extrae solo el valor de active del JSON {"active": false, "count": 12}.', '(?i)^false[.!]?$'),
 ('en_json', 'Extract only the value of count from JSON {"active": false, "count": 12}.', '^12[.!]?$'),
 ('es_homogeneous',
  'Dos modelos tienen matrices de tamaños diferentes. ¿Es correcto sumar sus gradientes directamente? '
  'Responde sí o no.',
  '(?i)^no[.!]?$'),
 ('en_homogeneous',
  'Two models have differently shaped parameter matrices. Is it correct to directly add their gradients? '
  'Answer yes or no.',
  '(?i)^no[.!]?$'),
 ('es_compare',
  'Datos: A procesa 20 fragmentos por minuto y B procesa 8. ¿Cuál procesa más por minuto? Responde A o B.',
  '^A[.!]?$'),
 ('en_compare',
  'Facts: A processes 20 shards per minute and B processes 8. Which processes more per minute? Reply A or B.',
  '^A[.!]?$')]

MULTITURN = [('es_context',
  [{'role': 'user', 'content': 'En este ejemplo ficticio, la clave de la tarea es lavanda.'},
   {'role': 'assistant', 'content': 'La clave de esta tarea es lavanda.'},
   {'role': 'user', 'content': '¿Cuál es la clave? Responde solo la palabra.'}],
  '^lavanda[.!]?$'),
 ('en_context',
  [{'role': 'user', 'content': 'In this fictional example, the task label is amber.'},
   {'role': 'assistant', 'content': 'The task label is amber.'},
   {'role': 'user', 'content': 'What is the label? Reply with only the word.'}],
  '^amber[.!]?$'),
 ('es_correction',
  [{'role': 'user', 'content': 'En este ejemplo, el equipo de interés se llama Azul.'},
   {'role': 'assistant', 'content': 'El equipo se llama Azul.'},
   {'role': 'user', 'content': 'Corrijo el ejemplo: se llama Coral. ¿Cómo se llama ahora? Solo el nombre.'}],
  '^Coral[.!]?$'),
 ('en_correction',
  [{'role': 'user', 'content': 'The example node is called Blue.'},
   {'role': 'assistant', 'content': 'The node is Blue.'},
   {'role': 'user', 'content': 'Correction: it is called Coral. What is its name now? Just the name.'}],
  '^Coral[.!]?$')]

def cases():
    return [(name,[{'role':'user','content':prompt}],pattern) for name,prompt,pattern in SEED_CASES+DIALOGUE_CASES]+MULTITURN


def digest_cases():
    return hashlib.sha256(json.dumps(cases(),ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def preservation_verdict(before,after):
    expected={row[0]:row[2] for row in cases()}
    def score(rows):
        if not isinstance(rows,list) or len(rows)!=len(expected) or {r['case'] for r in rows}!=set(expected):
            raise ValueError('Incomplete instruction/context gate')
        return {row['case']:re.search(expected[row['case']],row['answer']) is not None for row in rows}
    baseline=score(before);candidate=score(after)
    regressions=sorted(k for k in expected if baseline[k] and not candidate[k])
    mandatory={'es_instruction','en_instruction','es_public_context','en_public_context'}
    return {'case_digest':digest_cases(),'total':len(expected),'baseline_passed':sum(baseline.values()),
        'candidate_passed':sum(candidate.values()),'regressions':regressions,
        'eligible_for_promotion':not regressions and sum(candidate.values())>=37 and all(candidate[k] for k in mandatory),
        'scope':'Elementary public instruction/context screening; not broad reasoning certification'}
