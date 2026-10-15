import hashlib
import json
import time

import pytest
import torch

from src.learning.language_selector import train_selectors,tensor_digest
from src.learning.language_selector_protocol import definition
from src.learning import language_outcome_selector as component,language_outcome_selector_job as job
from src.learning.public_math import canonical
from test_language_outcome_selector import fixture


def test_measured_outcomes_change_learning_without_a_nll_teacher():
    p={**definition(),'phase_steps':8};queries=[[1+(i%2),3,4] for i in range(24)]
    targets=torch.tensor([[1.,0.] if i%2 else [0.,1.] for i in range(24)])
    a,fit,_=train_selectors(queries,None,32,p,teacher_targets=targets)
    b,_,_=train_selectors(queries,None,32,p,teacher_targets=targets.flip(1))
    assert tensor_digest(a['public'].state_dict())!=tensor_digest(b['public'].state_dict())
    assert not torch.allclose(a['dolphin'].choose(queries)[1],b['dolphin'].choose(queries)[1])
    for invalid in [targets*2,targets.long(),targets[:1],torch.full_like(targets,float('nan'))]:
        with pytest.raises(ValueError,match='outcome probabilities'):train_selectors(queries,None,32,p,teacher_targets=invalid)


def test_complete_crossfit_saves_real_states_never_loads_a_language_model_or_evaluation(monkeypatch,tmp_path):
    import transformers
    from src.learning import public_math,public_bases
    snapshot,outcomes=fixture(monkeypatch);p={**component.definition(),'phase_steps':2,'batch_size':4}
    monkeypatch.setattr(component,'definition',lambda:p);monkeypatch.setattr(job,'definition',lambda:p)
    monkeypatch.setattr(public_math,'load_snapshot',lambda path,split:snapshot if split=='train' else pytest.fail('opened evaluation'))
    base=tmp_path/'base';base.mkdir();pins={}
    for name,raw in [('config.json',canonical({'vocab_size':32})),('tokenizer.json',b'public fixture tokenizer'),('tokenizer_config.json',b'public fixture config')]:
        (base/name).write_bytes(raw);pins[name]=hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(public_bases,'BASES',{'qwen3-1.7b':pins})
    class Tokenizer:
        def apply_chat_template(self,messages,**kw):
            assert messages[0]['role']=='user' and '#### 1' not in messages[0]['content'];return messages[0]['content']
        def encode(self,*a,**kw):return [1,2,3]
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**kw:Tokenizer())
    monkeypatch.setattr(transformers.AutoModelForCausalLM,'from_pretrained',lambda *a,**kw:pytest.fail('loaded frozen language experts'))
    paths=[]
    for i,value in enumerate(outcomes):
        path=tmp_path/(str(i)+'.json');path.write_bytes(canonical(value));paths.append(str(path))
    monkeypatch.setattr(job,'OUTCOME_IDS',component.OUTCOME_IDS)
    request={'action':p['action'],'protocol_id':component.protocol_id(),'training_file':str(tmp_path/'train'),
        'outcome_files':paths,'base_directory':str(base),'output_directory':str(tmp_path/'output'),
        'device':'cpu','cpu_threads':2,'gpu_fraction':1.,'deadline':time.time()+120}
    result=job.run(request);report=json.loads((tmp_path/'output/diagnostic.json').read_text())
    assert result['success'] and report['verification']['out_of_fit_predictions_verified']
    assert len(report['artifacts'])==40 and len(report['folds'])==8
    assert torch.load(tmp_path/'output/0/optimizer-state.pt',weights_only=True)['dolphin']['state']
    assert report['original_evaluation_opened'] is False and not report['native_promotion_authorized']
    with pytest.raises(ValueError):job.validate_request({**request,'evaluation_file':str(tmp_path/'eval')})
