import hashlib
import time
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.network.lsp import LSPIdentity
from src.network.artifact_store import ArtifactService,ArtifactStore
from src.network.artifact_store import ArtifactError,ArtifactManifest
from src.network.lsp_v3 import LSPNodeV3
from src.network.work_protocol import WorkCoordinator
from src.contribution import ContributionPolicy,ResourceGovernor
from src.network.public_knowledge import managed_public_bytes
from src.release import language_family as family
from src.release.manifest import ReleaseManifest,ReleaseError


def bundle_fixture(monkeypatch):
    identity=LSPIdentity.generate();monkeypatch.setattr(family,'SIGNER',identity.node_id_hex)
    name='qwen3-0.6b';sizes={**family.COMMON_SIZES,**family.SIZES[name]}
    descriptor={'version':1,'base_name':name,'base_hash':family.base_hash(name),
        'files':[{'name':key,'artifact_id':value,'bytes':sizes[key]} for key,value in family.BASES[name].items()],
        'adapter_id':family.ADAPTERS[name],'adapter_bytes':4599712,'evaluation_id':'a'*64}
    release=ReleaseManifest.create(model_artifact_id=descriptor['adapter_id'],model_format='safetensors-v1',sequence=0,
        config_artifact_id=hashlib.sha256(family.canonical(descriptor)).hexdigest(),tokenizer_artifact_id=family.BASES[name]['tokenizer.json']).sign(identity)
    bundle={'release':release.to_dict(),'descriptor':descriptor}
    monkeypatch.setattr(family,'INITIAL_BUNDLES',{name:bundle})
    return bundle


def test_signed_family_binds_base_adapter_evaluation_and_tokenizer(monkeypatch):
    bundle=bundle_fixture(monkeypatch)
    release,descriptor=family.verify_bundle(bundle)
    assert release.model_artifact_id==descriptor['adapter_id']
    assert len(family.dependencies(bundle))==9
    bad=deepcopy(bundle);bad['descriptor']['evaluation_id']='b'*64
    with pytest.raises(ReleaseError):family.verify_bundle(bad)
    bad=deepcopy(bundle);bad['release']['signatures']=[]
    with pytest.raises(ReleaseError):family.verify_bundle(bad)


@pytest.mark.parametrize('field,value',[('name','../personal.enc'),('name','private.pt'),('artifact_id','a'*64),('bytes',1)])
def test_dependency_paths_weights_and_sizes_are_closed_upstream_pins(monkeypatch,field,value):
    bundle=bundle_fixture(monkeypatch);bundle['descriptor']['files'][0][field]=value
    with pytest.raises(ReleaseError):family.verify_bundle(bundle)


def test_fetch_client_does_not_overwrite_physical_node_artifact_guards(tmp_path):
    class Work:
        def register_handler(self,*_,**__):raise AssertionError('Fetch-only must not replace a guard')
    ArtifactService(Work(),ArtifactStore(tmp_path),namespace='language.artifact',register_handlers=False)


def test_signed_size_mismatch_is_rejected_before_requesting_chunks(tmp_path):
    manifest=ArtifactManifest('a'*64,1024*1024,'model','application/octet-stream',time.time())
    calls=[]
    class Work:
        def submit(self,operation,*_,**__):
            calls.append(operation)
            return SimpleNamespace(status='ok',output={'manifest':manifest.to_dict()})
    service=ArtifactService(Work(),ArtifactStore(tmp_path),namespace='language.artifact',register_handlers=False)
    with pytest.raises(ArtifactError,match='signed dependency'):
        service.fetch('a'*64,peer_id='b'*64,expected_size=1)
    assert calls==['language.artifact.describe.v1']
    assert not list(service.store.incoming.iterdir())


def test_language_namespace_transfers_verified_content_without_claiming_a_physical_pilot(tmp_path):
    hardware=SimpleNamespace(cpu_cores=1,ram_gb=1,has_gpu=False,gpu_vram_gb=0,disk_gb=10)
    nodes=[];works=[];stores=[]
    try:
        for name in ['source','receiver']:
            root=tmp_path/name;root.mkdir()
            node=LSPNodeV3(LSPIdentity.generate(),host='127.0.0.1',port=0,address_book_path=root/'peers.json',target_outbound=0,allow_private=True)
            node.start();nodes.append(node)
            governor=ResourceGovernor(ContributionPolicy.for_mode('relay'),hardware=hardware,storage_path=root)
            works.append(WorkCoordinator(node,governor,max_offers_per_minute=60));stores.append(ArtifactStore(root/'artifacts'))
        content=b'public namespace fixture'*10000;source=tmp_path/'fixture';source.write_bytes(content)
        manifest=stores[0].import_file(source,kind='model')
        ArtifactService(works[0],stores[0],namespace='language.artifact')
        client=ArtifactService(works[1],stores[1],namespace='language.artifact',register_handlers=False)
        assert nodes[1].connect_peer('127.0.0.1',nodes[0].port)
        path=client.fetch(manifest.artifact_id,peer_id=nodes[0].identity.node_id_hex,timeout_s=5,expected_size=len(content))
        assert path.read_bytes()==content
    finally:
        for work in works:work.close()
        for node in nodes:node.stop()


def test_disk_accounting_includes_all_families_engines_and_partial_downloads(tmp_path):
    for name,size in [('public/artifacts/file',3),('public/language/artifacts/incoming/file.part',7),('engines/motor',11),('language-models/base/weights',13),('personal.enc',17)]:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'x'*size)
    assert managed_public_bytes(tmp_path)==34
