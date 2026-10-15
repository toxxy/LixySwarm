import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
from contextlib import contextmanager

import pytest
import torch

from src.learning.language_dense_selector import DenseExpertSelector,train_dense_selectors,binding,save_dense,load_dense
from src.learning.language_selector import tensor_digest
from src.learning import language_prefix_router as protocol,language_prefix_router_job as job
from src.learning.public_math import canonical


def test_small_prefix_pings_learn_verified_outcomes_and_freeze_public_memory():
    p={**protocol.definition(),'phase_steps':8};features=torch.randn(24,32);targets=torch.tensor([[1.,0.] if i%2 else [0.,1.] for i in range(24)])
    a,fit,_=train_dense_selectors(features,targets,p)
    b,_,_=train_dense_selectors(features,targets.flip(1),p)
    assert tensor_digest(a['dolphin'].state_dict())!=tensor_digest(b['dolphin'].state_dict())
    assert fit['dolphin_parameter_count']<100000
    assert all(tensor_digest(a[n].dolphin.state_dict())==fit['dolphin_state_id'] for n in p['memory_modes'])
    old=tensor_digest(a['public'].state_dict());choices,prob=a['public'].choose(features[:4])
    assert prob.shape==(4,2) and choices.shape==(4,) and tensor_digest(a['public'].state_dict())==old
    with pytest.raises(ValueError):train_dense_selectors(features.requires_grad_(),targets,p)


def test_dense_reload_binds_query_representation_experts_and_learned_dolphin(tmp_path):
    model=DenseExpertSelector();metadata=binding(model,'public');path=tmp_path/'router.safetensors'
    save_dense(model,path,metadata);restored=DenseExpertSelector();load_dense(restored,path,metadata)
    assert tensor_digest(restored.state_dict())==tensor_digest(model.state_dict())
    for changed in [{**metadata,'memory_scope':'private'},{**metadata,'feature_space_id':'a'*64},
                    {**metadata,'expert_ids':metadata['expert_ids'][::-1]},{**metadata,'dolphin_state_id':'b'*64}]:
        before=tensor_digest(restored.state_dict())
        with pytest.raises(ValueError):load_dense(restored,path,changed)
        assert tensor_digest(restored.state_dict())==before


def test_prefix_extraction_has_no_answer_or_labels_interface():
    class PublicQuery(dict):
        def __getitem__(self,key):
            if key in {'answer','final_number'}:pytest.fail('Reference answer entered the prefix encoder')
            return super().__getitem__(key)
    class Tokenizer:
        def apply_chat_template(self,messages,**kw):
            assert messages[0]['role']=='user';return messages[0]['content']
        def encode(self,text,**kw):assert 'secret reference' not in text;return [1,2,3]
    prefixes=job.encode_prefixes(Tokenizer(),[PublicQuery(question='public question',answer='secret reference')],vocab_size=32)
    class Backbone:
        def __call__(self,**kw):
            assert set(kw)=={'input_ids','attention_mask','use_cache','return_dict'}
            assert kw['input_ids'].tolist()==[[1,2,3]]
            return SimpleNamespace(last_hidden_state=torch.arange(12,dtype=torch.float32).reshape(1,3,4))
    result=job.extract_prefix_features(SimpleNamespace(model=Backbone()),prefixes,dimensions=4,device='cpu',check=lambda:None,progress=lambda *_:None)
    assert result.tolist()==[[8.,9.,10.,11.]] and not result.requires_grad


def test_complete_prefix_fit_commits_all_states_and_new_predictions_before_any_generation(monkeypatch,tmp_path):
    import transformers
    from src.learning import public_math,public_bases,weights_io,language_adapter,language_streamed_float,language_specialist
    from src.learning import language_dense_selector as dense
    p={**protocol.definition(),'phase_steps':2};monkeypatch.setattr(protocol,'definition',lambda:p)
    monkeypatch.setattr(job,'definition',lambda:p);monkeypatch.setattr(dense,'definition',lambda:p)
    base=tmp_path/'base';base.mkdir();pins={}
    for name,raw in [('config.json',b'config'),('tokenizer.json',b'tokenizer')]:
        (base/name).write_bytes(raw);pins[name]=hashlib.sha256(raw).hexdigest()
    monkeypatch.setitem(public_bases.BASES,'qwen3-1.7b',pins)
    # Keep the prospectively declared representation identity internally bound.
    p['feature_space_id']=protocol.feature_id();p['feature_definition']=protocol.feature_definition()
    snapshot={'split':'train','rows':[{'case':f'{i:064x}','question':f'public query {i}','answer':'#### 1','final_number':'1'} for i in range(4096)]}
    monkeypatch.setattr(public_math,'load_snapshot',lambda path,split:snapshot if split=='train' else pytest.fail('original EVAL opened'))
    outcome_paths=[];outcome_ids=[]
    for i in range(2):
        path=tmp_path/f'outcome{i}';raw=canonical({'synthetic_fixture':i});path.write_bytes(raw)
        outcome_paths.append(str(path));outcome_ids.append(hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(job,'OUTCOME_IDS',tuple(outcome_ids))
    monkeypatch.setattr(job,'dataset',lambda *_:[{'target':[1.,0.]} for _ in range(128)])
    parent=tmp_path/'parent';parent.write_bytes(b'parent');specialist=tmp_path/'specialist';specialist.write_bytes(b'specialist')
    original_digest=weights_io.digest_file
    monkeypatch.setattr(weights_io,'digest_file',lambda path:(job.PARENT_MODEL,6) if Path(path)==parent else
        (job.SPECIALIST_ID,10) if Path(path)==specialist else original_digest(path))
    class Tokenizer:
        def apply_chat_template(self,messages,**kw):return messages[0]['content']
        def encode(self,text,**kw):return [3,int(text.rsplit(' ',1)[1])+4]
        def decode(self,tokens,**kw):return '#### 1' if int(tokens[0])==100 else '#### 2'
    output=tmp_path/'output';generated=[]
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.slot=torch.nn.Parameter(torch.ones(1));self.config=SimpleNamespace(vocab_size=512,use_cache=False)
        def model(self,**kw):
            inputs=kw['input_ids'];vector=torch.sin(torch.arange(2048,dtype=torch.float32)*float(inputs[0,-1]))
            return SimpleNamespace(last_hidden_state=vector.reshape(1,1,2048).expand(1,inputs.shape[1],2048))
        def generate(self,**kw):
            committed=json.loads((output/'prediction-commitment.json').read_text());protocol.verify_predictions(committed)
            for name,item in committed['learned_artifacts'].items():assert original_digest(output/name)==(item['sha256'],item['bytes'])
            generated.append(kw['input_ids'].tolist());token=100 if float(self.slot)==1. else 101
            return torch.cat([kw['input_ids'],torch.tensor([[token]])],dim=1)
    model=Model()
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**kw:Tokenizer())
    monkeypatch.setattr(transformers.AutoModelForCausalLM,'from_pretrained',lambda *a,**kw:model)
    monkeypatch.setattr(language_adapter,'install_adapter',lambda m:{'p':m.slot})
    def load(m,path,*_):
        with torch.no_grad():m.slot.fill_(1 if Path(path)==parent else 2)
    monkeypatch.setattr(language_adapter,'load_adapter',load)
    monkeypatch.setattr(language_streamed_float,'install_streamed_float32',lambda m:{'scheme':'synthetic fixture'})
    class Bank:
        def __init__(self,m,parameters,states):self.model=m;self.states=states
        @contextmanager
        def activate(self,choice):
            old=self.model.slot.detach().clone()
            try:
                with torch.no_grad():self.model.slot.copy_(self.states[choice]['p'])
                yield self.model
            finally:
                with torch.no_grad():self.model.slot.copy_(old)
    monkeypatch.setattr(language_specialist,'FrozenAdapterBank',Bank)
    request={'action':p['action'],'protocol_id':protocol.protocol_id(),'compute_dtype':'float32','streamed_base':True,
        'model_id':job.PARENT_MODEL,'parent_release_id':job.PARENT_RELEASE,'base_directory':str(base),'adapter_file':str(parent),
        'training_file':str(tmp_path/'train'),'specialist_file':str(specialist),'outcome_files':outcome_paths,
        'output_directory':str(output),'device':'cpu','cpu_threads':2,'gpu_fraction':1.,'deadline':time.time()+120}
    response=job.run(request);report=json.loads((output/'diagnostic.json').read_text())
    assert response['success'] and len(generated)==256
    assert all(132<=ids[0][-1]<=259 for ids in generated)
    assert report['verification']['parent_correct']==128 and report['verification']['specialist_correct']==0
    assert report['original_evaluation_opened'] is False and report['native_promotion_authorized'] is False
    assert float(model.slot)==1. and len(report['artifacts'])==7
