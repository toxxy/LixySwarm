import base64
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import time
import uuid

import pytest

from src.network.lsp import LSPIdentity
from src.network.public_memory import PublicMemoryService,PublicMemorySettings,select_cards,augment_messages
from src.network.public_knowledge import PublicKnowledgeService,KnowledgeSettings
from src.network.client_state import ClientSettings
from src.network.artifact_store import ArtifactError
from src.release import public_memory as memory


def language_source():
    return json.loads((Path(__file__).parent/'experiments/reports/public_memory_language_source_20261014.json').read_text())


@pytest.fixture
def setup(monkeypatch,tmp_path):
    publisher=LSPIdentity.generate()
    monkeypatch.setattr(memory,'SIGNER',publisher.node_id_hex)
    registry=memory.PublicMemoryRegistry(tmp_path/'source')
    def add(epoch):
        document=memory.build_document(epoch,language_source())
        manifest=memory.sign_document(document,registry.chain(),publisher)
        stage=registry.store.incoming/(uuid.uuid4().hex+'.json');stage.write_bytes(memory.canonical(document))
        registry.store.commit_generated(stage,kind='dataset',media_type='application/json')
        registry.install(registry.chain()+[manifest])
        return manifest,document
    return registry,publisher,add


def test_public_sources_signed_versions_corrections_and_restart(setup):
    registry,publisher,add=setup
    first,before=add(0);second,after=add(1)
    assert len(before['cards'])==len(after['cards'])==10
    assert first['content_id']!=second['content_id']
    assert registry.document()==after
    restored=memory.PublicMemoryRegistry(registry.root.parent.parent)
    assert restored.chain()==[first,second] and restored.document()==after
    assert restored.install([first])==second
    fork=memory.sign_document(before,[],publisher)
    with pytest.raises(ValueError,match='fork'):restored.install([fork])
    assert restored.chain()==[first,second]


@pytest.mark.parametrize('change',[
    lambda d:d.update(scope='personal'),lambda d:d.update(private_memory='secret'),
    lambda d:d['cards'][0].update(text='private-canary-never-publish'),
    lambda d:d['sources'].update(personal_history=[]),
    lambda d:d['sources'].update(synthetic_epoch=True),
    lambda d:d['sources']['accepted_language']['evaluations'].clear(),
])
def test_personal_or_unverified_sources_cannot_be_signed(setup,change):
    registry,publisher,_=setup
    doc=memory.build_document(0,language_source());change(doc)
    with pytest.raises(ValueError):memory.sign_document(doc,[],publisher)
    assert not registry.chain()


def test_wrong_signer_domain_manifest_and_corrupt_content_leave_head_unchanged(setup):
    registry,publisher,add=setup
    first,document=add(0)
    other=LSPIdentity.generate()
    with pytest.raises(ValueError):memory.sign_document(document,[],other)
    wrong=deepcopy(first);wrong['scope']='personal'
    with pytest.raises(ValueError):memory.verify_chain([wrong])
    wrong=deepcopy(first);wrong['signature']=base64.b64encode(publisher.sign(b'wrong domain')).decode()
    with pytest.raises(ValueError):memory.verify_chain([wrong])
    registry.document()
    registry.store.object_path(first['content_id']).write_bytes(b'changed after caching')
    with pytest.raises(ValueError):registry.document()
    assert registry.chain()==[first]


def test_explicit_public_topics_and_budget_never_use_personal_history(setup):
    registry,_,add=setup
    add(0)
    _,doc=add(1)
    assert select_cards(doc,'What is my personal code for Orchid?')==[]
    cards=select_cards(doc,'According to the public registry, what is the code for device Orchid?')
    assert cards and cards[0]['record_id']=='en_correction'
    assert 'copper-947' in cards[0]['text'] and 'cobalt-365' not in cards[0]['text']
    query='According to the public catalog, what is the accepted qwen3-1.7b adapter hash?'
    model=select_cards(doc,query)
    assert [row['record_id'] for row in model]==['language-adapter']
    messages=[{'role':'user','content':'synthetic-private-history'}, {'role':'assistant','content':'private reply'}, {'role':'user','content':query}]
    augmented,ids=augment_messages(messages,model)
    assert augmented[:-1]==messages[:-1] and augmented[-1]['content'].endswith(query)
    assert ids==['language-adapter'] and len(augmented)==len(messages)
    assert sum(len(row['content'].encode()) for row in augmented)<=4096
    full=[{'role':'user','content':'x'*4096}]
    assert augment_messages(full,model)==(full,[])


def test_partial_or_invalid_public_content_cannot_replace_a_valid_head(setup):
    registry,publisher,add=setup
    first,_=add(0)
    document=memory.build_document(1,language_source());manifest=memory.sign_document(document,[first],publisher)
    with pytest.raises((OSError,ValueError)):registry.install([first,manifest])
    assert registry.chain()==[first]
    # Even a trusted signer cannot relabel a private string as an accepted
    # source: the receiver independently derives every card from public inputs.
    document['cards'][0]['text']='private-memory-canary'
    body={k:v for k,v in manifest.items() if k not in {'signature','release_id'}}
    body.update(content_id=memory.digest(document),content_bytes=len(memory.canonical(document)))
    forged={**body,'release_id':memory.digest(body),'signature':base64.b64encode(publisher.sign(memory.DOMAIN+memory.canonical(body))).decode()}
    stage=registry.store.incoming/'rejected.json';stage.write_bytes(memory.canonical(document))
    registry.store.commit_generated(stage,kind='dataset',media_type='application/json')
    with pytest.raises(ValueError):registry.install([first,forged])
    assert registry.chain()==[first]


def test_public_memory_controls_are_closed_and_do_not_load_personal_storage(tmp_path):
    assert PublicMemorySettings.load(tmp_path).enabled
    PublicMemorySettings(enabled=False).save(tmp_path)
    assert not PublicMemorySettings.load(tmp_path).enabled
    path=tmp_path/'public-memory-controls.json';path.write_text('{"version":1,"enabled":true,"personal_path":"secret"}')
    with pytest.raises(ValueError):PublicMemorySettings.load(tmp_path)


def test_physical_broker_consent_separates_public_evidence_from_private_history(monkeypatch,tmp_path,setup):
    from test_chat_service import broker_fixture
    registry,_,add=setup;add(0);_,document=add(1)
    broker,sent=broker_fixture(monkeypatch)
    class Public:
        enabled=True
        def retrieve(self,query):
            assert 'private-history-canary' not in query
            return (select_cards(document,query),registry.chain()[-1]) if self.enabled else ([],None)
        def snapshot(self):return {'available':self.enabled,'records':10}
    public=Public();broker.public_memory=public
    question='According to the public catalog, what is the accepted qwen3-1.7b adapter hash?'
    request={'request_id':'1'*32,'remote_context':False,'messages':[{'role':'user','content':'private-history-canary'},
        {'role':'assistant','content':'private answer'},{'role':'user','content':question}]}
    try:
        answer=broker.ask(request)
        assert answer['public_memory']['used'] and not answer['context_sent']
        assert 'private-history-canary' not in json.dumps(sent[-1]) and len(sent[-1]['messages'])==1
        assert sent[-1]['messages'][-1]['content'].endswith(question)
        public.enabled=False;request['request_id']='2'*32
        assert not broker.ask(request)['public_memory']['used']
        assert sent[-1]['messages']==[{'role':'user','content':question}]
        assert registry.document()==document
    finally:broker.close()


def test_signed_memory_survives_producer_loss_and_serves_from_replica_over_lsp(setup,tmp_path):
    from test_public_knowledge import node,RESOURCES
    from src.contribution import ResourceRequirements
    from src.network.public_memory import CATALOG
    registry,_,add=setup;add(0);head,document=add(1)
    source=registry.root.parent.parent;replica_dir=tmp_path/'replica';reader_dir=tmp_path/'reader'
    provider,replica,reader=[node(path) for path in [source,replica_dir,reader_dir]]
    stacks=[]
    settings=ClientSettings()
    for instance,path in [(provider,source),(replica,replica_dir),(reader,reader_dir)]:
        knowledge=PublicKnowledgeService(instance,path,RESOURCES)
        # This test exercises memory only; do not schedule the unrelated
        # reference-model background acquisition when peers connect.
        knowledge.complete=True
        knowledge.update(settings,KnowledgeSettings(enabled=instance is provider,consented_at=time.time()))
        service=PublicMemoryService(knowledge,path);service.update(settings)
        stacks.append((knowledge,service))
    try:
        provider.start();replica.start();reader.start()
        assert replica.connect_peer('127.0.0.1',provider.port)
        accepted=stacks[1][1].acquire(provider.identity.node_id_hex,threading.Event())
        assert accepted==head and stacks[1][1].registry.document()==document
        assert not stacks[1][0].allowed  # Reception did not enable public sharing.
        provider.stop();stacks[0][1].close();stacks[0][0].close()
        sharing=KnowledgeSettings(enabled=True,consented_at=time.time())
        stacks[1][0].update(settings,sharing)
        assert reader.connect_peer('127.0.0.1',replica.port)
        assert stacks[2][1].acquire(replica.identity.node_id_hex,threading.Event())==head
        assert stacks[2][1].registry.document()==document
        restored=memory.PublicMemoryRegistry(reader_dir)
        assert restored.document()==document
        sharing.enabled=False;stacks[1][0].update(settings,sharing)
        result=stacks[2][0].work.submit(CATALOG,{'space_id':memory.SPACE_ID},ResourceRequirements(kind='artifact'),
            peer_id=replica.identity.node_id_hex,timeout_s=3)
        assert result.status!='ok'
        assert restored.document()==document
    finally:
        for knowledge,service in stacks:service.close();knowledge.close()
        for instance in [reader,replica,provider]:instance.stop()


def test_unaccepted_content_is_never_exposed_by_memory_namespace(setup,tmp_path):
    from test_public_knowledge import node,RESOURCES
    registry,_,add=setup;add(0)
    source=registry.root.parent.parent;instance=node(source)
    knowledge=PublicKnowledgeService(instance,source,RESOURCES)
    knowledge.update(ClientSettings(),KnowledgeSettings(enabled=True,consented_at=time.time()))
    service=PublicMemoryService(knowledge,source)
    stage=registry.store.incoming/'unaccepted.json';stage.write_text('{"private":"not accepted"}')
    artifact=registry.store.commit_generated(stage,kind='dataset',media_type='application/json')
    execution=SimpleNamespace(raise_if_cancelled=lambda:None,deadline=time.time()+5)
    try:
        with pytest.raises(ArtifactError,match='accepted'):
            service._guard(service.transfer._describe,content=True)({'artifact_id':artifact.artifact_id},execution)
        service.controls.enabled=False
        assert not service._cached(SimpleNamespace(operation='public-memory.catalog.v1'),{'chain':registry.chain()})
    finally:service.close();knowledge.close();instance.stop()
