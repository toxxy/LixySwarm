from copy import deepcopy
import json
import subprocess
import sys

import pytest

from desktop.chat_history import ChatHistory, bounded_context, recalled_context
from src.matriarca.personal_context import PersonalContextMemory, SPACE_ID


class KeyStore:
    def __init__(self): self.values = {}
    def get_password(self, service, account): return self.values.get((service, account))
    def set_password(self, service, account, value): self.values[service, account] = value


def pair(text, answer='Entendido.'):
    return [{'role': 'user', 'content': text}, {'role': 'assistant', 'content': answer}]


def test_memory_recovers_an_old_statement_beyond_visible_history_after_restart(tmp_path):
    backend = KeyStore()
    history = ChatHistory(tmp_path, backend=backend)
    marker = 'synthetic-private-code-zafiro-682'
    fact = pair('La clave del equipo Amapola es ' + marker)
    history.store.save({'unrelated_personal_bank': {'keep': True}})
    history.save(fact, False, memory_enabled=True)
    recent = [message for i in range(101) for message in pair(f'Registro de práctica {i}: hay dos libros sobre la mesa.')]
    history.save(fact + recent, False)
    restored = ChatHistory(tmp_path, backend=backend)
    assert len(restored.messages) == 200 and marker not in json.dumps(restored.messages)
    question = '¿Cuál es la clave del equipo Amapola?'
    baseline = bounded_context(restored.messages, question)
    recalled, ids = restored.context(restored.messages, question)
    assert ids and marker not in json.dumps(baseline) and marker in json.dumps(recalled)
    assert recalled[-1] == baseline[-1] == {'role': 'user', 'content': question}
    assert len(recalled) <= 8 and sum(len(row['content'].encode()) for row in recalled) <= 4096
    assert marker.encode() not in (tmp_path / 'personal.enc').read_bytes()
    assert restored.store.load()['unrelated_personal_bank'] == {'keep': True}
    assert history.memory.snapshot() == restored.memory.snapshot()


def test_disabled_memory_does_not_consolidate_and_forgetting_preserves_history(tmp_path):
    history = ChatHistory(tmp_path, backend=KeyStore())
    messages = pair('El proyecto ficticio Cobre usa la etiqueta lirio-731.')
    history.save(messages, True)
    assert not history.memory_enabled and not history.memory.snapshot()['episodes']
    history.save(messages, True, memory_enabled=True)
    assert history.memory.snapshot()['episodes']
    history.save(messages, True, memory_enabled=False)
    assert history.memory.snapshot()['episodes']
    assert history.context([], '¿Cuál es la etiqueta de Cobre?')[1] == []
    history.forget_memory()
    assert not history.memory.snapshot()['episodes'] and not history.memory_enabled
    assert history.store.load()['chat']['messages'] == messages


def test_memory_deduplicates_complete_turns_and_does_not_learn_guesses():
    memory = PersonalContextMemory()
    messages = pair('El proyecto Brisa usa roble-341.')
    memory.observe(messages)
    memory.observe(messages)
    assert len(memory.snapshot()['episodes']) == 1
    memory.observe(pair('¿Cuál es la clave de Brisa?', 'La clave inventada es equivocada.'))
    memory.observe([{'role': 'user', 'content': 'Una frase pendiente sin respuesta.'}])
    assert len(memory.snapshot()['episodes']) == 1
    memory.observe(pair('Nada relacionado aquí.', 'La clave de Brisa es imaginada.'))
    assert len(memory.retrieve('¿Cuál es la clave de Brisa?')) == 1


def test_recent_correction_is_last_and_memory_budget_never_expands():
    memory = PersonalContextMemory()
    old = pair('La clave del equipo Amapola es anterior-328.')
    new = pair('Corrección: la clave del equipo Amapola es nueva-972.')
    memory.observe(old + new)
    recalled, identifiers = recalled_context([], '¿Cuál es la clave del equipo Amapola?', memory)
    assert len(identifiers) == 2
    assert recalled[:4] == old + new
    recent = pair('Esta es una conversación reciente distinta.')
    long_question = 'clave del equipo Amapola ' + 'x' * (4096 - len('clave del equipo Amapola '))
    assert recalled_context(recent, long_question, memory) == ([{'role': 'user', 'content': long_question}], [])
    assert recalled_context(old, '¿Cuál es la clave del equipo Amapola?', memory)[0].count(old[0]) == 1


@pytest.mark.parametrize('change', [
    lambda v: v.update(scope='global'),
    lambda v: v.update(space_id='untrained-neural-space'),
    lambda v: v.update(version=2),
    lambda v: v['episodes'][0].update(episode_id='0' * 64),
    lambda v: v['episodes'].append(deepcopy(v['episodes'][0])),
    lambda v: v['episodes'][0]['messages'][0].update(role='system'),
    lambda v: v['episodes'][0]['messages'][0].update(content='tampered'),
])
def test_incompatible_or_tampered_personal_space_is_rejected(change):
    memory = PersonalContextMemory()
    memory.observe(pair('El equipo ficticio Amapola tiene una clave local.'))
    value = memory.snapshot()
    change(value)
    with pytest.raises(ValueError): PersonalContextMemory(value)


def test_record_limit_and_unrelated_queries():
    memory = PersonalContextMemory()
    for i in range(150): memory.observe(pair(f'El registro ficticio {i} contiene un dato local.'))
    assert len(memory.snapshot()['episodes']) == 128
    assert not memory.retrieve('Fotosíntesis y astronomía')
    assert memory.snapshot()['space_id'] == SPACE_ID
    snapshot = memory.snapshot()
    snapshot['episodes'].clear()
    assert len(memory.snapshot()['episodes']) == 128


def test_personal_retrieval_does_not_import_tensor_models():
    result = subprocess.run([sys.executable, '-c',
        'from desktop.chat_history import ChatHistory; import sys; '
        'assert "torch" not in sys.modules and "transformers" not in sys.modules; print("ok")'],
        capture_output=True, text=True, timeout=10, check=True)
    assert result.stdout.strip() == 'ok'


def test_broker_never_sends_personal_recall_without_context_consent(monkeypatch, tmp_path):
    from test_chat_service import broker_fixture
    broker, sent = broker_fixture(monkeypatch)
    history = ChatHistory(tmp_path, backend=KeyStore())
    marker = 'synthetic-private-memory-43819'
    history.save(pair('La clave del equipo Amapola es ' + marker), False, memory_enabled=True)
    messages, ids = history.context([], '¿Cuál es la clave del equipo Amapola?')
    assert ids
    original = history.memory.snapshot()
    try:
        broker.ask({'request_id': 'a' * 32, 'messages': messages, 'remote_context': False})
        assert marker not in json.dumps(sent[-1])
        broker.ask({'request_id': 'b' * 32, 'messages': messages, 'remote_context': True})
        assert marker in json.dumps(sent[-1])
        assert marker not in json.dumps(broker.snapshot())
        assert history.memory.snapshot() == original
        assert all(marker.encode() not in path.read_bytes() for path in tmp_path.iterdir() if path.is_file())
    finally: broker.close()


def test_fresh_process_fixture_guard_emits_no_memory_text(monkeypatch, tmp_path):
    from desktop import memory_fixture_check, chat_history
    from src.learning.public_memory_gate import memory_cases,statement_messages,distractor_messages
    history=ChatHistory(tmp_path,backend=KeyStore())
    case=memory_cases()[0]
    history.save(statement_messages(case),True,memory_enabled=True)
    history.save(distractor_messages()[-200:],True)
    monkeypatch.setattr(chat_history,'ChatHistory',lambda _:history)
    checked=memory_fixture_check.check(tmp_path,case['case'])
    assert checked['success'] and checked['restored_in_fresh_process']
    assert case['expected'] not in json.dumps(checked)
    history.value['other_personal_bank']={'private':'keep local'}
    with pytest.raises(ValueError):memory_fixture_check.check(tmp_path,case['case'])


def memory_comparison_fixture():
    import hashlib
    from dataclasses import replace
    from src.learning.public_memory_gate import memory_cases
    from src.network.lsp import LSPIdentity
    from src.network.work_protocol import WorkResult,ResultReceipt
    owner,worker=LSPIdentity.generate(),LSPIdentity.generate()
    results=[]
    for case in memory_cases():
        row={'case':case['case'],'fresh_process':{'success':True,'restored_in_fresh_process':True}}
        for mode in ['recent','memory']:
            answer='Unknown.' if mode=='recent' else case['expected']
            result=WorkResult(job_id=hashlib.sha256((case['case']+mode).encode()).hexdigest(),status='ok',
                output={'model_id':'a'*64,'answer':answer,'output_tokens':8})
            result=replace(result,receipt=ResultReceipt.create(result,worker,owner.node_id_hex).to_dict())
            row[mode]={'answer':answer,'model_id':'a'*64,'provider_id':worker.node_id_hex,
                'receipt':result.receipt,'signed_result':result.to_dict(),'context_bytes':512,'context_messages':8,
                'context_sent':True,'route':'network','recalled_episodes':int(mode=='memory')}
        results.append(row)
    return results


def test_memory_gate_requires_fixed_cases_budget_fresh_work_and_signatures():
    from src.learning.public_memory_gate import verdict
    comparisons=memory_comparison_fixture()
    assert verdict(comparisons)['eligible']
    assert verdict(comparisons)['memory_correct']==8
    for change in [lambda r:r.pop(),lambda r:r[0]['memory'].update(context_bytes=4097),
        lambda r:r[0]['memory']['signed_result']['output'].update(answer='forged'),
        lambda r:r[0]['memory'].update(provider_id='0'*64),
        lambda r:r[0]['fresh_process'].update(restored_in_fresh_process=False),
        lambda r:r[0].update(memory=deepcopy(r[0]['recent']))]:
        changed=deepcopy(comparisons);change(changed)
        with pytest.raises(ValueError):verdict(changed)
