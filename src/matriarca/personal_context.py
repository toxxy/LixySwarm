"""Text retrieval for Matriarca's local, encrypted conversation memory.

This component does not import tensor models, read files, export memories or
produce global signals. Its lexical space is explicitly distinct from the
legacy neural infrasound space. The GUI owns encryption and remote consent.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import re
import unicodedata

SPACE_ID = 'lixyswarm-personal-text-es-en-v1'
MAX_EPISODES = 128
MAX_EPISODE_BYTES = 1024
STOP = set('a al algo como con cual cuando de del el ella en es esa ese esta este esto fue ha hay la las lo los me mi mis no o para por que se si sin solo su sus un una uno unos y ya '
           'about an and are as at be been by can do does for from had has have how i in is it its me my of on or our please so some that the their them there these they this to was were what when which who with you your'.split())


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def episode_id(messages):
    return hashlib.sha256(canonical({'space_id': SPACE_ID, 'messages': messages})).hexdigest()


def terms(text):
    folded = ''.join(c for c in unicodedata.normalize('NFKD', text.casefold()) if not unicodedata.combining(c))
    return set(word for word in re.findall(r'\w+', folded) if len(word) >= 2 and word not in STOP)


class PersonalContextMemory:
    def __init__(self, value=None):
        self.value = deepcopy(value) if value is not None else {
            'version': 1, 'scope': 'personal', 'space_id': SPACE_ID, 'next_position': 0, 'episodes': []}
        self.validate()

    def validate(self):
        value = self.value
        if (not isinstance(value, dict) or set(value) != {'version', 'scope', 'space_id', 'next_position', 'episodes'}
                or type(value['version']) is not int or value['version'] != 1 or value['scope'] != 'personal'
                or value['space_id'] != SPACE_ID or type(value['next_position']) is not int
                or not 0 <= value['next_position'] <= 10**12 or not isinstance(value['episodes'], list)
                or len(value['episodes']) > MAX_EPISODES):
            raise ValueError('La memoria personal tiene un formato o espacio incompatible')
        ids, positions = set(), []
        for row in value['episodes']:
            if (not isinstance(row, dict) or set(row) != {'episode_id', 'position', 'messages'}
                    or type(row['position']) is not int or not 0 <= row['position'] < value['next_position']
                    or not isinstance(row['messages'], list) or len(row['messages']) != 2):
                raise ValueError('Fragmento de memoria personal inválido')
            for message, role in zip(row['messages'], ['user', 'assistant']):
                if (not isinstance(message, dict) or set(message) != {'role', 'content'} or message['role'] != role
                        or not isinstance(message['content'], str) or not message['content']):
                    raise ValueError('La memoria solo conserva intercambios locales completos')
            if (sum(len(m['content'].encode()) for m in row['messages']) > MAX_EPISODE_BYTES
                    or row['episode_id'] != episode_id(row['messages']) or row['episode_id'] in ids):
                raise ValueError('Fragmento personal alterado, repetido o demasiado grande')
            ids.add(row['episode_id'])
            positions.append(row['position'])
        if positions != sorted(set(positions)):
            raise ValueError('El orden de la memoria personal no es válido')

    def observe(self, messages):
        """Consolidate actual complete turns; never invent facts or answers."""
        self.validate()
        known = {row['episode_id'] for row in self.value['episodes']}
        source=messages[-200:]
        for user, assistant in zip(source, source[1:]):
            if user['role'] != 'user' or assistant['role'] != 'assistant':
                continue
            # A previous unanswered question or an assistant's guess must not
            # become the remembered source for the same future question.
            if '?' in user['content'] or '¿' in user['content']:
                continue
            pair = [user, assistant]
            if (not all(isinstance(m['content'], str) and m['content'] for m in pair)
                    or sum(len(m['content'].encode()) for m in pair) > MAX_EPISODE_BYTES):
                continue
            identifier = episode_id(pair)
            if identifier in known:
                continue
            self.value['episodes'].append({'episode_id': identifier, 'position': self.value['next_position'], 'messages': deepcopy(pair)})
            self.value['next_position'] += 1
            known.add(identifier)
        self.value['episodes'] = self.value['episodes'][-MAX_EPISODES:]
        self.validate()

    def retrieve(self, question, *, exclude=(), limit=2):
        self.validate()
        if not isinstance(question, str) or not 1 <= len(question.encode()) <= 4096 or type(limit) is not int or not 0 <= limit <= 2:
            raise ValueError('Consulta personal fuera de sus límites')
        query = terms(question)
        episodes = self.value['episodes']
        # User statements select a memory. Assistant speculation cannot make
        # an otherwise unrelated personal episode match the current question.
        documents = [terms(row['messages'][0]['content']) for row in episodes]
        frequency = Counter(term for document in documents for term in document)
        excluded = set(exclude)
        ranked = []
        for row, document in zip(episodes, documents):
            overlap = query & document
            if not overlap or row['episode_id'] in excluded:
                continue
            score = sum(math.log1p((len(episodes) + 1) / (frequency[term] + 1)) for term in overlap)
            # For the same matching terms, a later correction precedes an old
            # statement. Returned episodes are rendered in chronological order.
            ranked.append((score, row['position'], row))
        selected = sorted(ranked, key=lambda item: item[:2], reverse=True)[:limit]
        return deepcopy([item[2] for item in sorted(selected, key=lambda item: item[1])])

    def snapshot(self):
        self.validate()
        return deepcopy(self.value)
