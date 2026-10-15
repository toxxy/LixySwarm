import hashlib
import io
import json
import platform
import sys
import threading
import zipfile
from types import SimpleNamespace

import pytest

from src.contribution import ContributionPolicy
from src.network.client_state import ClientSettings, NodeStateLock
from src.network.engine_catalog import EngineDownload, EnginePart
from src.network.engine_download import prepare_engine
from src.network.language_compute import LanguageSettings
from src.network import language_preparation as preparation
from test_language_install import fixture as model_fixture


def context(monkeypatch, tmp_path, hook=lambda: None, corrupt=False):
    store, objects = model_fixture(monkeypatch, tmp_path)
    from src.network import language_install
    release, descriptor = language_install.verify_bundle({})
    descriptor['base_name'] = preparation.CHAT_FAMILY
    bundle = {'release': {'release_id': release.release_id}}
    deps = [(identifier, size) for _, identifier, size in objects]
    monkeypatch.setattr(preparation, 'INITIAL_BUNDLES', {preparation.CHAT_FAMILY: bundle})
    monkeypatch.setattr(preparation, 'CHAT_ADAPTER', release.model_artifact_id)
    monkeypatch.setattr(preparation, 'dependencies', lambda *_: deps)
    monkeypatch.setattr(preparation,'LanguageVersionRegistry',lambda *_:SimpleNamespace(head=lambda:bundle))
    name = 'LixySwarmLanguageCUDA'+('.exe' if sys.platform == 'win32' else '')
    binary = b'public engine fixture; never executed'
    manifest = json.dumps({'version': 1, 'name': 'LixySwarmLanguageCUDA', 'platform': sys.platform,
        'machine': platform.machine(), 'torch': 'fixture', 'operations': ['language.inference.v1'],
        'entrypoint': name, 'files': [{'path': name, 'bytes': len(binary), 'sha256': hashlib.sha256(binary).hexdigest()}]}).encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr(name, binary)
        archive.writestr('engine-package.json', manifest)
    data = buffer.getvalue()
    entry = EngineDownload('https://example.invalid/fixture.zip', hashlib.sha256(data).hexdigest(), len(data),
        hashlib.sha256(manifest).hexdigest(), len(manifest)+len(binary), profile='language-cuda')
    monkeypatch.setattr(preparation, 'language_engine', lambda _: entry)
    opens = []
    def opener(*_, **__):
        opens.append(True)
        hook()
        response = io.BytesIO(b'x'*len(data) if corrupt else data)
        response.status = 200
        response.headers = {}
        return response
    monkeypatch.setattr(preparation, 'prepare_engine', lambda entry, directory, **kw:
        prepare_engine(entry, directory, opener=opener, **kw))
    policy = ContributionPolicy.for_mode('maximum')
    policy.gpu_percent = 100
    policy.max_gpu_jobs = 1
    ClientSettings(policy=policy).save(tmp_path)
    families = SimpleNamespace(root=tmp_path/'public/language', store=store)
    service = preparation.LanguagePreparation(SimpleNamespace(), SimpleNamespace(allowed=True), families, tmp_path,
        {'has_gpu': True, 'gpu_vram_gb': 8})
    # Package fixtures are never executable; the public GPU smoke test belongs
    # to the physical-machine run. Keep orchestration tests explicitly scoped.
    monkeypatch.setattr(service,'_probe',lambda *_:None)
    monkeypatch.setattr(service,'_prepare_version',lambda settings,entry,allowed:
        (release,language_install.materialize_family(tmp_path,preparation.CHAT_FAMILY,allowed=allowed)))
    return service, entry, descriptor, opens


def finish(service):
    service.thread.join(timeout=5)
    assert not service.thread.is_alive()


def test_verified_model_and_native_engine_activate_without_changing_user_policy(monkeypatch, tmp_path):
    service, entry, _, opens = context(monkeypatch, tmp_path)
    controls = (tmp_path/'controls.json').read_bytes()
    service.start()
    finish(service)
    assert service.snapshot()['state'] == 'complete', service.snapshot()
    engine = LanguageSettings.load(tmp_path)
    assert engine.enabled and engine.adapter_id == preparation.CHAT_ADAPTER
    assert engine.command[0].startswith(str(tmp_path/'engines'/entry.manifest_sha256))
    assert (tmp_path/'controls.json').read_bytes() == controls
    assert json.loads(service.intent.read_text())['requested'] is False
    assert opens


def test_engine_finishes_and_discards_download_cache_before_model_materialization(monkeypatch,tmp_path):
    service,entry,_,_=context(monkeypatch,tmp_path);original=service._prepare_version;observed=[]
    def prepare_model(*args):
        assert (tmp_path/'engines'/entry.manifest_sha256).is_dir()
        assert not list((tmp_path/'engine-downloads').iterdir())
        observed.append('engine-installed-before-model')
        return original(*args)
    monkeypatch.setattr(service,'_prepare_version',prepare_model)
    service.start();finish(service)
    assert service.snapshot()['state']=='complete' and observed==['engine-installed-before-model']


def test_stage_peak_and_proven_model_links_fit_eight_gib_without_assuming_copy_support(monkeypatch,tmp_path):
    from src.network.artifact_store import ArtifactStore
    gib=1024**3;store=ArtifactStore(tmp_path/'public/language/artifacts')
    bundle={'release':{'release_id':'c'*64}}
    monkeypatch.setattr(preparation,'INITIAL_BUNDLES',{preparation.CHAT_FAMILY:bundle})
    monkeypatch.setattr(preparation,'dependencies',lambda *_:[('d'*64,4*gib)])
    entry=EngineDownload('https://example.invalid/engine','a'*64,2*gib,'b'*64,6*gib,profile='language-cuda',regular_file_bytes=3*gib)
    # Peak with shared model files: max(2+3 during extraction, 3+4 finally).
    linked=preparation.remaining_budget(tmp_path,store,entry,hardlinks=True)
    copied=preparation.remaining_budget(tmp_path,store,entry,hardlinks=False)
    assert linked==7*gib+preparation.BUDGET_MARGIN_BYTES and linked<8*gib
    assert copied==11*gib+preparation.BUDGET_MARGIN_BYTES and copied>8*gib


@pytest.mark.parametrize('withdraw', ['pause', 'selection'])
def test_permission_or_operator_selection_changes_prevent_activation(monkeypatch, tmp_path, withdraw):
    def change():
        if withdraw == 'pause':
            controls = ClientSettings.load(tmp_path)
            controls.paused = True
            controls.save(tmp_path)
        else:
            LanguageSettings(adapter_id='a'*64).save(tmp_path)
    service, _, _, _ = context(monkeypatch, tmp_path, hook=change)
    service.start()
    finish(service)
    assert service.snapshot()['state'] == 'waiting'
    assert not LanguageSettings.load(tmp_path).enabled
    if withdraw == 'selection':
        assert LanguageSettings.load(tmp_path).adapter_id == 'a'*64
        assert json.loads(service.intent.read_text())['requested'] is False


def test_corrupt_package_cannot_enable_expert(monkeypatch, tmp_path):
    service, _, _, _ = context(monkeypatch, tmp_path, corrupt=True)
    service.start()
    finish(service)
    assert service.snapshot()['state'] == 'waiting'
    assert not LanguageSettings.load(tmp_path).enabled


def test_preparation_respects_low_laptop_resources_and_serializes_with_cpu_download(monkeypatch, tmp_path):
    service, _, _, opens = context(monkeypatch, tmp_path)
    controls = ClientSettings.load(tmp_path)
    controls.policy.max_ram_gb = 2
    controls.save(tmp_path)
    with pytest.raises(ValueError, match='5 GB'):
        service.start()
    assert not service.intent.exists() and not opens
    controls.policy.max_ram_gb = 5
    controls.save(tmp_path)
    with NodeStateLock(tmp_path/'preparation-lock'):
        service.start()
        finish(service)
        assert not opens
    service.next_attempt = 0
    service.update()
    finish(service)
    assert service.snapshot()['state'] == 'complete'


def test_partial_archives_reduce_remaining_reservation_for_resume(monkeypatch, tmp_path):
    service, entry, _, _ = context(monkeypatch, tmp_path)
    from dataclasses import replace
    entry = replace(entry, parts=(EnginePart('https://example.invalid/one', 'a'*64, entry.bytes),))
    before = preparation.remaining_budget(tmp_path, service.families.store, entry)
    root = tmp_path/'engine-downloads'
    root.mkdir()
    (root/('a'*64+'.part')).write_bytes(b'x'*17)
    assert preparation.remaining_budget(tmp_path, service.families.store, entry) == before-17


def test_stop_preserves_download_intent_as_disabled_without_activation(monkeypatch, tmp_path):
    begun=threading.Event();released=threading.Event()
    def hook():begun.set();assert released.wait(3)
    service,_,_,_=context(monkeypatch,tmp_path,hook=hook)
    service.start();assert begun.wait(3)
    service.stop();released.set();finish(service)
    assert service.snapshot()['state']=='stopped'
    assert not LanguageSettings.load(tmp_path).enabled
    assert not json.loads(service.intent.read_text())['requested']


def test_probe_uses_resource_lease_and_rejects_wrong_answers(monkeypatch,tmp_path):
    from src.contribution import ResourceGovernor
    from src.network.lsp import LSPIdentity
    service,_,_,_=context(monkeypatch,tmp_path)
    service.node.identity=LSPIdentity.generate()
    governor=ResourceGovernor(ClientSettings.load(tmp_path).policy,
        hardware=SimpleNamespace(cpu_cores=8,ram_gb=8,has_gpu=True,gpu_vram_gb=8,disk_gb=10),storage_path=tmp_path)
    governor._system_busy=lambda:False
    service.knowledge.governor=governor
    answer={'model_id':preparation.CHAT_ADAPTER,'answer':'Paris','output_tokens':1}
    def compute(_self,command,request,execution,**kwargs):
        assert governor.status()['active_gpu_jobs']==1
        assert governor.status()['leased_ram_gb']==5
        assert kwargs['allowed']() and kwargs['ram_gb']==5
        assert request['messages'][0]['content']==preparation.quality_cases()[0]['prompt']
        assert request['adapter_id']==answer['model_id']
        return answer
    monkeypatch.setattr(preparation.ComputeSupervisor,'run',compute)
    preparation.LanguagePreparation._probe(service,'fixture',tmp_path,ClientSettings.load(tmp_path),lambda:True)
    assert governor.status()['active_gpu_jobs']==0
    answer['model_id']='c'*64
    preparation.LanguagePreparation._probe(service,'fixture',tmp_path,ClientSettings.load(tmp_path),lambda:True,'c'*64)
    answer['model_id']=preparation.CHAT_ADAPTER
    answer['answer']='Madrid'
    with pytest.raises(ValueError,match='no pasó'):
        preparation.LanguagePreparation._probe(service,'fixture',tmp_path,ClientSettings.load(tmp_path),lambda:True)
    assert governor.status()['active_gpu_jobs']==0


def test_preparation_selects_the_verified_current_adapter_instead_of_genesis(monkeypatch,tmp_path):
    service,entry,_,_=context(monkeypatch,tmp_path)
    current=SimpleNamespace(release_id='b'*64,model_artifact_id='c'*64)
    monkeypatch.setattr(preparation,'LanguageVersionRegistry',lambda *_:SimpleNamespace(head=lambda:{'release':{'release_id':current.release_id}}))
    monkeypatch.setattr(service,'_prepare_version',lambda settings,entry,allowed:(current,tmp_path/'public-fixture'))
    checked=[]
    monkeypatch.setattr(service,'_probe',lambda command,base,settings,allowed,adapter_id:checked.append(adapter_id))
    service.start();finish(service)
    assert service.snapshot()['state']=='complete',service.snapshot()
    assert checked==[current.model_artifact_id]
    assert LanguageSettings.load(tmp_path).adapter_id==current.model_artifact_id


def test_newer_head_while_preparing_cannot_activate_an_obsolete_adapter(monkeypatch,tmp_path):
    service,entry,_,_=context(monkeypatch,tmp_path)
    head={'release':{'release_id':'b'*64}}
    monkeypatch.setattr(preparation,'LanguageVersionRegistry',lambda *_:SimpleNamespace(head=lambda:head))
    monkeypatch.setattr(service,'_prepare_version',lambda settings,entry,allowed:
        (SimpleNamespace(release_id='a'*64,model_artifact_id='c'*64),tmp_path/'public-fixture'))
    service.start();finish(service)
    assert service.snapshot()['state']=='waiting'
    assert not LanguageSettings.load(tmp_path).enabled


@pytest.mark.parametrize('failure',['evaluation','family','signature'])
def test_current_version_materialization_rechecks_real_signed_history_and_evaluation(monkeypatch,tmp_path,failure):
    from test_language_versions import fixture
    from src.release.language_versions import LanguageVersionRegistry
    root=tmp_path/'public/language'
    store,first,update,_publisher=fixture(monkeypatch,root);second,report=update(first)
    registry=LanguageVersionRegistry(root,'qwen3-0.6b');registry.install([first,second],store)
    monkeypatch.setattr(preparation,'CHAT_FAMILY','qwen3-0.6b')
    monkeypatch.setattr(preparation,'INITIAL_PROVIDERS',set())
    service=preparation.LanguagePreparation(SimpleNamespace(peers=lambda:[]),SimpleNamespace(allowed=True),
        SimpleNamespace(root=root,store=store),tmp_path,{'has_gpu':True,'gpu_vram_gb':8})
    entry=EngineDownload('https://example.invalid/fixture','a'*64,1,'b'*64,1,profile='language-cuda')
    controls=ClientSettings(policy=ContributionPolicy.for_mode('maximum'))
    selected,base=service._prepare_version(controls,entry,lambda:True)
    assert selected.release_id==second['release']['release_id']
    assert (base/'adapter.safetensors').read_bytes()==store.object_path(second['release']['model_artifact_id']).read_bytes()
    assert not LanguageSettings.load(tmp_path).enabled
    if failure=='evaluation':
        evaluation=second['descriptor']['evaluation_id'];store.object_path(evaluation).write_bytes(b'corrupt public fixture')
    else:
        path=registry.bundles/(second['release']['release_id']+'.json')
        damaged=json.loads(path.read_text())
        if failure=='family':damaged['descriptor']['base_name']='qwen3-1.7b'
        else:damaged['release']['signatures'][0]['signature']='A'*88
        path.write_text(json.dumps(damaged))
    with pytest.raises(ValueError):service._prepare_version(controls,entry,lambda:True)
