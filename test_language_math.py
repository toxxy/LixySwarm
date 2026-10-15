"""Synthetic policy/receipt/trace transitions; not physical model-quality evidence."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct
import threading
from types import SimpleNamespace

import pytest

from src.learning import language_policy as policy, public_math as math
from src.network.artifact_store import ArtifactStore,digest_file
from src.network.chat_versions import ChatVersions
from src.network.work_protocol import ResultReceipt,WorkResult
from src.network.compute_process import ComputeProcessError
from src.release import language_versions as versions
from test_language_versions import fixture as language_fixture
from test_public_math import snapshots


def bind_math_fixture_policy(monkeypatch):
    for key,value in [('dataset_id',math.curriculum_id()),('training_artifact_id',math.ARTIFACTS['train'][0]),
                      ('evaluation_artifact_id',math.ARTIFACTS['evaluation'][0]),('evaluation_gate_id',math.gate_id())]:
        monkeypatch.setitem(policy.MATH_POLICY,key,value)
    monkeypatch.setattr(policy,'MATH_POLICY_ID',hashlib.sha256(json.dumps(policy.MATH_POLICY,sort_keys=True,separators=(',',':')).encode()).hexdigest())


def math_fixture(monkeypatch,tmp_path,snapshots):
    bind_math_fixture_policy(monkeypatch)
    store,first,update,publisher=language_fixture(monkeypatch,tmp_path,name='qwen3-1.7b')
    parent,old=update(first);chain=[first,parent]
    def put(content,kind):
        path=store.incoming/'math-fixture';path.write_bytes(content)
        return store.commit_generated(path,kind=kind,media_type='application/json' if kind=='evaluation' else 'application/octet-stream')
    for path,_value in snapshots.values():store.import_file(path,kind='dataset',media_type='application/json')
    content=store.object_path(parent['release']['model_artifact_id']).read_bytes()
    model=put(content[:-4]+struct.pack('<f',.2),'model')
    report=deepcopy(old);report.update(version=4,gate_id=versions.gate_id(4),dataset_id=policy.dataset_id(4),
        parent_release_id=parent['release']['release_id'],adapter_id=model.artifact_id)
    results=[]
    for raw,worker in zip(old['contributions'],update.workers):
        result=WorkResult.from_dict(raw);out={**result.output,'adapter_id':parent['release']['model_artifact_id'],
            'dataset_id':policy.dataset_id(4),'training_policy_id':policy.policy_id(4),'quantization_relative_l2_error':.0002,'replay_tokens':192}
        result=replace(result,output=out)
        results.append(replace(result,receipt=ResultReceipt.create(result,worker,publisher.node_id_hex).to_dict()).to_dict())
    report['contributions']=results
    before={'nll':1.,'tokens':24,'answers':[{'case':row['case'],'answer':'Reasoning.\n#### 0'} for row in snapshots['evaluation'][1]['rows']],
            'correct':0,'total':8,'output_tokens':[4]*8,'at_output_limit':0}
    after=deepcopy(before);after.update(nll=.8,correct=4)
    for answer,row in zip(after['answers'][:4],snapshots['evaluation'][1]['rows']):answer['answer']=row['answer']
    trace={'version':1,'evaluation_artifact_id':math.ARTIFACTS['evaluation'][0],'before':before,'after':after}
    value=put(math.canonical(trace),'evaluation')
    report.update(math_before=math.compact_evaluation(before),math_after=math.compact_evaluation(after),
                  math_trace_id=value.artifact_id,math_trace_bytes=value.size)
    evaluation=put(math.canonical(report),'evaluation')
    bundle=versions.build_update(chain,model,evaluation,report,publisher,store=store)
    put(math.canonical(bundle['descriptor']),'other')
    return store,chain,bundle,report,model,publisher,put


def test_math_release_requires_complete_trace_and_retains_all_historical_evaluation(monkeypatch,tmp_path,snapshots):
    store,chain,bundle,report,_model,_publisher,_put=math_fixture(monkeypatch,tmp_path,snapshots)
    assert bundle['descriptor']['version']==5 and bundle['release']['minimum_client_version']=='0.4.0a18'
    with pytest.raises(ValueError,match='trace|test data'):versions.verify_evaluation(report,chain[-1],bundle)
    assert versions.verify_evaluation(report,chain[-1],bundle,store=store)['physical_workers']==sorted(versions.WORKERS)
    registry=versions.LanguageVersionRegistry(tmp_path,'qwen3-1.7b');registry.install(chain+[bundle],store)
    assert versions.LanguageVersionRegistry(tmp_path,'qwen3-1.7b').head()==bundle
    dependency_ids={identifier for identifier,_ in versions.dependencies(bundle)}
    assert {math.ARTIFACTS['train'][0],math.ARTIFACTS['evaluation'][0],report['math_trace_id']}<=dependency_ids
    trace=store.object_path(report['math_trace_id']);raw=trace.read_bytes();trace.write_bytes(b'corrupt public trace')
    with pytest.raises(ValueError):registry.install(chain+[bundle],store)
    trace.write_bytes(raw)
    assert registry.head()==bundle


@pytest.mark.parametrize('problem',['score_projection','raw_answer_digest','old_loss','math_loss','precision','replay_tokens','policy'])
def test_even_signed_math_metadata_cannot_bypass_trace_data_quality_or_precision(monkeypatch,tmp_path,snapshots,problem):
    store,chain,_bundle,report,model,publisher,put=math_fixture(monkeypatch,tmp_path,snapshots)
    bad=deepcopy(report)
    if problem=='score_projection':bad['math_after']['answers'][4]['answer']='#### 5'
    elif problem=='raw_answer_digest':bad['math_after']['raw_answer_sha256'][0]='0'*64
    elif problem=='old_loss':bad['candidate_nll']=bad['baseline_nll']*1.01
    elif problem=='math_loss':bad['math_after']['nll']=bad['math_before']['nll']
    elif problem=='precision':bad['contributions'][0]['output']['quantization_relative_l2_error']=.002
    elif problem=='replay_tokens':bad['contributions'][0]['output']['replay_tokens']=True
    else:bad['contributions'][0]['output']['training_policy_id']=policy.HALF_POLICY_ID
    evaluation=put(math.canonical(bad),'evaluation')
    with pytest.raises(ValueError):versions.build_update(chain,model,evaluation,bad,publisher,store=store)


def test_light_chat_requires_math_trace_and_test_data_but_no_weights(monkeypatch,tmp_path,snapshots):
    store,chain,bundle,report,_model,_publisher,_put=math_fixture(monkeypatch,tmp_path/'producer',snapshots)
    reports={chain[1]['descriptor']['evaluation_id']:json.loads(store.object_path(chain[1]['descriptor']['evaluation_id']).read_text()),
             bundle['descriptor']['evaluation_id']:report}
    chat=ChatVersions(tmp_path/'chat')
    with pytest.raises(ValueError):chat.install(chain+[bundle],reports)
    for identifier,_ in versions.evidence_dependencies(bundle):
        manifest=store.manifest(identifier);chat.store.import_file(store.object_path(identifier),kind=manifest.kind,media_type=manifest.media_type)
    chat.install(chain+[bundle],reports)
    assert ChatVersions(tmp_path/'chat').head()==bundle
    assert not chat.store.has(bundle['release']['model_artifact_id'])
    path=chat.store.object_path(report['math_trace_id']);path.write_bytes(b'corrupted trace')
    with pytest.raises(ValueError):chat.head()


def test_existing_closed_memory_policy_keeps_a_self_contained_verified_source(monkeypatch,tmp_path,snapshots):
    from src.release.public_memory import language_source,build_document,verify_document
    root=tmp_path/'public/language'
    store,chain,bundle,_report,_model,_publisher,_put=math_fixture(monkeypatch,root,snapshots)
    versions.LanguageVersionRegistry(root,'qwen3-1.7b').install(chain+[bundle],store)
    source=language_source(tmp_path)
    assert source['chain']==chain
    assert source['chain'][-1]['release']['release_id']!=bundle['release']['release_id']
    document=build_document(1,source)
    assert verify_document(document,1)==document


def test_new_policy_is_opt_in_and_old_files_do_not_enable_it(tmp_path):
    from src.network.language_training import LanguageTrainingSettings
    old=LanguageTrainingSettings();value=old.__dict__.copy();value.pop('math_curriculum_enabled')
    (tmp_path/'language-training.json').write_text(json.dumps(value))
    assert not LanguageTrainingSettings.load(tmp_path).math_curriculum_enabled
    assert policy.dataset_id(4)!=policy.dataset_id(3) and policy.policy_id(4)!=policy.HALF_POLICY_ID
    assert policy.dataset_length(4)==4096 and policy.answer_limit(4)==512


def test_new_data_program_keeps_stopped_history_and_uses_its_own_cursor(monkeypatch,tmp_path,snapshots):
    from src.network.language_learning import LanguageLearningCoordinator,LanguageLearningSettings
    from src.network.telemetry import atomic_json
    store,chain,_bundle,_report,_model,_publisher,_put=math_fixture(monkeypatch,tmp_path/'public',snapshots)
    versions.LanguageVersionRegistry(tmp_path/'public','qwen3-1.7b').install(chain,store)
    node=tmp_path/'node';previous=node/'language-learning/ledger.json'
    old={'version':1,'base_name':'qwen3-1.7b','dataset_id':policy.dataset_id(1),'round':5,'cursor':96,'epoch':0,
         'pending':None,'status':'no_improvement','accepted_sequence':1,'rejections':3,'generation':0,
         'history':[{'round':n,'accepted':False} for n in range(5)],'error':None}
    atomic_json(previous,old);original=previous.read_bytes()
    knowledge=SimpleNamespace(node=SimpleNamespace(identity=SimpleNamespace(node_id_hex=next(iter(versions.WORKERS)))))
    training=SimpleNamespace(families=SimpleNamespace(root=tmp_path/'public',store=store))
    coordinator=LanguageLearningCoordinator(knowledge,training,node)
    coordinator.settings=LanguageLearningSettings(training_policy_version=4,base_name='qwen3-1.7b')
    assert coordinator._select_program();coordinator._initialize()
    assert coordinator.ledger['round']==0 and coordinator.ledger['cursor']==0 and coordinator.ledger['generation']==0
    assert coordinator.snapshot()['historical_rounds']==5 and previous.read_bytes()==original
    admission=json.loads((coordinator.root/'program.json').read_text())
    assert admission['previous_ledger_sha256']==hashlib.sha256(original).hexdigest()
    restored=LanguageLearningCoordinator(knowledge,training,node);restored.settings=coordinator.settings
    assert restored._select_program() and restored.ledger==coordinator.ledger and previous.read_bytes()==original


def test_program_change_cannot_discard_old_pending_work(monkeypatch,tmp_path):
    from test_language_learning import setup
    from src.network.language_learning import LanguageLearningSettings
    coordinator,_controls,_head,_calls=setup(monkeypatch,tmp_path)
    coordinator._initialize();coordinator._save(pending={'id':'f'*32,'attempts':1})
    original=coordinator.path.read_bytes()
    coordinator.settings=LanguageLearningSettings(training_policy_version=4,base_name='qwen3-1.7b')
    assert not coordinator._select_program() and coordinator.program_error
    assert coordinator.path.read_bytes()==original and coordinator.ledger['pending']['id']=='f'*32


def test_math_worker_acquires_training_data_before_starting_the_tensor_child(monkeypatch,tmp_path,snapshots):
    from test_public_math import transfer_fixture
    from test_language_training import versioned_service
    from src.network import language_install
    from src.learning.public_bases import base_hash
    from src.contribution import ResourceRequirements
    bind_math_fixture_policy(monkeypatch)
    _source,target,work,provider,owner,proofs=transfer_fixture(tmp_path,snapshots)
    training,head,payload,execution=versioned_service(monkeypatch,tmp_path/'node')
    training.engine.base_name='qwen3-1.7b';training.engine.batching_enabled=True;training.engine.half_precision_enabled=True
    training.engine.math_curriculum_enabled=True;training.engine.requesters=[provider]
    training.families.store=target;training.knowledge.work=work;training.knowledge.budget=None
    training.knowledge.node=SimpleNamespace(identity=SimpleNamespace(node_id_hex=owner),connected_node_ids=[provider],
        peers=lambda:[{'node_id':provider,'transport':'relay','encrypted':True,'host':'8.8.8.8'}])
    payload.update(base_name='qwen3-1.7b',dataset_id=policy.dataset_id(4),training_policy_id=policy.policy_id(4))
    execution.work.origin_node_id=provider;execution.cancel_event=threading.Event()
    execution.raise_if_cancelled=lambda:None;execution.remaining_s=lambda:30
    execution.resource_requirements=lambda:ResourceRequirements(kind='training',gpu_required=True,ram_gb=6)
    monkeypatch.setattr(language_install,'materialize_version',lambda *_a,**_kw:tmp_path/'public-model')
    def run(_command,request,_execution,**kwargs):
        assert kwargs['allowed']() and len(proofs)==2
        assert request['action']=='gradient-math'
        assert math.load_snapshot(request['training_file'],'train')==snapshots['train'][1]
        path=Path(request['output_file']);path.write_bytes(b'synthetic typed-engine output fixture')
        return {**{k:payload[k] for k in ['adapter_id','dataset_id','start','count','training_policy_id']},
                'base_hash':base_hash('qwen3-1.7b'),'tokens':16,'replay_tokens':192,'loss':.3,'retention_loss':.1,
                'quantization_relative_l2_error':.0002,'gradient_id':digest_file(path)[0],'gradient_bytes':path.stat().st_size}
    monkeypatch.setattr(training.supervisor,'run',run)
    result=training.math_gradient(payload,execution)
    assert result['dataset_id']==policy.dataset_id(4) and training.completed==1
    assert training.dataset_preparation['route']['public_internet'] and not training.dataset_preparation['initially_local']
    training.engine.math_curriculum_enabled=False
    with pytest.raises(ComputeProcessError):training.ensure_math_data(execution,lambda:True)


@pytest.mark.parametrize('problem',['gain','math_loss','regression'])
def test_consistent_raw_trace_still_must_pass_the_stronger_quality_gate(monkeypatch,tmp_path,snapshots,problem):
    store,chain,_bundle,report,model,publisher,put=math_fixture(monkeypatch,tmp_path,snapshots)
    bad=deepcopy(report);trace=json.loads(store.object_path(report['math_trace_id']).read_text())
    if problem=='gain':
        trace['after']['answers'][3]['answer']='#### 0';trace['after']['correct']=3
    elif problem=='math_loss':trace['after']['nll']=.995
    else:
        trace['before']['answers'][-1]['answer']=snapshots['evaluation'][1]['rows'][-1]['answer'];trace['before']['correct']=1
        trace['after']['answers'][4]['answer']=snapshots['evaluation'][1]['rows'][4]['answer'];trace['after']['correct']=5
    artifact=put(math.canonical(trace),'evaluation')
    bad.update(math_before=math.compact_evaluation(trace['before']),math_after=math.compact_evaluation(trace['after']),
               math_trace_id=artifact.artifact_id,math_trace_bytes=artifact.size)
    evaluation=put(math.canonical(bad),'evaluation')
    with pytest.raises(ValueError,match='stronger'):versions.build_update(chain,model,evaluation,bad,publisher,store=store)


def test_math_views_metadata_and_repair_over_real_local_lsp(monkeypatch,tmp_path,snapshots):
    from src.contribution import ContributionPolicy,ResourceGovernor
    from src.network.lsp import LSPIdentity
    from src.network.lsp_v3 import LSPNodeV3
    from src.network.work_protocol import WorkCoordinator
    from src.network.language_families import LanguageFamilies
    from src.network.language_version_service import fetch_history
    from src.network.language_compute import LanguageComputeService
    from src.contribution import ResourceRequirements
    root=tmp_path/'provider'
    store,chain,bundle,report,_model,_publisher,_put=math_fixture(monkeypatch,root/'public/language',snapshots)
    versions.LanguageVersionRegistry(root/'public/language','qwen3-1.7b').install(chain+[bundle],store)
    nodes=[];works=[]
    try:
        for name in ['provider','receiver']:
            directory=tmp_path/name
            node=LSPNodeV3(LSPIdentity.generate(),host='127.0.0.1',port=0,address_book_path=directory/'peers.json',target_outbound=0,allow_private=True)
            nodes.append(node)
            governor=ResourceGovernor(ContributionPolicy.for_mode('relay'),hardware=SimpleNamespace(cpu_cores=1,ram_gb=1,disk_gb=1,has_gpu=False,gpu_vram_gb=0),storage_path=directory)
            works.append(WorkCoordinator(node,governor,max_offers_per_minute=600))
        LanguageFamilies(SimpleNamespace(work=works[0],allowed=True,budget=SimpleNamespace(wait=lambda *_a,**_kw:None),cancel=threading.Event()),root)
        for node in nodes:node.start()
        assert nodes[1].connect_peer('127.0.0.1',nodes[0].port)
        provider=nodes[0].identity.node_id_hex
        assert fetch_history(works[1],'qwen3-1.7b',provider)==chain+[bundle]
        for version in [1,2,3,4]:
            result=works[1].submit('language.version-head.v'+str(version),{'base_name':'qwen3-1.7b'},
                ResourceRequirements(kind='artifact'),peer_id=provider)
            assert result.status=='ok' and result.output==(bundle if version==4 else chain[-1])
        old=SimpleNamespace(node=SimpleNamespace(identity=nodes[1].identity,
            peers=lambda:[{'node_id':provider,'capabilities':{'language_versions_v3':True}}]),submit=works[1].submit)
        assert fetch_history(old,'qwen3-1.7b',provider)==chain
        chat=ChatVersions(tmp_path/'chat')
        chat.synchronize(works[1],provider,cancel_event=threading.Event(),preflight=lambda *_:None)
        assert chat.head()==bundle and not chat.store.has(bundle['release']['model_artifact_id'])
        assert not chat.store.has(math.ARTIFACTS['train'][0])
        chat.store.object_path(report['math_trace_id']).write_bytes(b'corrupted public generation trace')
        with pytest.raises(ValueError):chat.head()
        chat.synchronize(works[1],provider,cancel_event=threading.Event(),preflight=lambda *_:None)
        assert chat.head()==bundle
        local=LanguageComputeService(SimpleNamespace(work=SimpleNamespace(register_handler=lambda *_a,**_kw:None)),root,{})
        local.ready=True;local.engine.base_name='qwen3-1.7b';local.engine.adapter_id=bundle['release']['model_artifact_id']
        assert bundle['release']['model_artifact_id'] in local.approved_models()
        assert bundle['release']['model_artifact_id'] not in local.approved_models(maximum_descriptor=4)
        assert chain[-1]['release']['model_artifact_id'] in local.approved_models(maximum_descriptor=4)
    finally:
        for work in works:work.close()
        for node in nodes:node.stop()
