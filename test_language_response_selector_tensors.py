from copy import deepcopy

import pytest
import torch

from src.learning.language_response_selector import (fit_projection,project,train,ResponseSelector,binding,save,load)
from src.learning.language_response_router import definition


@pytest.fixture(scope='module')
def fitted():
    old=torch.get_num_threads();torch.set_num_threads(1);torch.manual_seed(123)
    hidden=torch.randn(65,2048);lengths=torch.rand(65,2)
    projection=fit_projection(hidden,lengths);features=project(hidden,lengths,projection)
    targets=torch.stack([(torch.arange(65)%2).float(),((torch.arange(65)//4)%2).float(),
        torch.full((65,),.05),torch.full((65,),.5)],-1)
    variants,fit,optimizers=train(features,targets)
    yield hidden,lengths,projection,features,targets,variants,fit,optimizers
    torch.set_num_threads(old)


def test_training_projection_is_finite_normalized_and_sign_canonical(fitted):
    hidden,lengths,p,features,*_=fitted
    assert features.shape==(65,18) and not features.requires_grad
    assert torch.allclose(features.mean(0),torch.zeros(18),atol=1e-5)
    assert torch.allclose(features.std(0,unbiased=False),torch.ones(18),atol=1e-5)
    assert torch.allclose(p['basis']@p['basis'].T,torch.eye(16),atol=1e-6)
    assert torch.all(p['basis'].gather(1,p['basis'].abs().argmax(1,keepdim=True))>0)
    bad=hidden.clone();bad[0,0]=float('nan')
    with pytest.raises(ValueError):fit_projection(bad,lengths)
    with pytest.raises(ValueError):project(hidden.requires_grad_(),lengths,p)
    hidden.requires_grad_(False)


def test_svd_projection_can_be_saved_and_reloaded_without_changing_its_values(tmp_path,fitted):
    from safetensors.torch import save_file,load_file
    hidden,lengths,projection,features,*_=fitted
    assert all(v.is_contiguous() for v in projection.values())
    path=tmp_path/'projection.safetensors';save_file(projection,str(path));restored=load_file(str(path))
    assert all(torch.equal(v,restored[k]) for k,v in projection.items())
    assert torch.equal(features,project(hidden,lengths,restored))


def test_equal_memory_training_changes_only_public_outcome_support(fitted):
    _,_,_,features,targets,nets,fit,optimizers=fitted
    assert fit['projection_fit_used_test'] is False and fit['memory_scope']=='public-training-only'
    assert fit['support_indices']==list(range(0,65,4))
    assert set(fit['support_indices']).isdisjoint(fit['query_indices'])
    assert nets['zero'].public_memory.count_nonzero()==0
    assert torch.equal(nets['public'].public_memory[:,:8],nets['shuffled'].public_memory[:,:8])
    assert torch.equal(nets['public'].public_memory[:,8:12],targets[fit['support_indices']])
    assert not torch.equal(nets['public'].public_memory[:,8:12],nets['shuffled'].public_memory[:,8:12])
    for mode in definition()['memory_modes']:
        assert fit['phase2'][mode]['steps']==definition()['phase_steps']
        for key,value in nets['dolphin'].dolphin.state_dict().items():assert torch.equal(value,nets[mode].dolphin.state_dict()[key])
        assert torch.equal(nets['dolphin'].utility_head.weight,nets[mode].utility_head.weight)
    for optimizer in [optimizers['dolphin'],*optimizers['memory'].values()]:
        assert len(optimizer['state'])==sum(len(g['params']) for g in optimizer['param_groups'])
        for state in optimizer['state'].values():
            assert float(state['step'])==256
            assert torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all()


def test_learned_memory_reloads_and_rejects_another_feature_projection_or_role(tmp_path,fitted):
    _,_,_,features,_,nets,*_=fitted
    for name,net in nets.items():
        metadata=binding(net,name,'a'*64);path=tmp_path/(name+'.safetensors');save(net,path,metadata)
        fresh=ResponseSelector(len(net.public_memory));load(fresh,path,metadata);fresh.eval()
        assert all(torch.equal(value,fresh.state_dict()[key]) for key,value in net.state_dict().items())
        with torch.no_grad():assert torch.allclose(net(features,use_memory=name!='dolphin'),
            fresh(features,use_memory=name!='dolphin'),atol=1e-6,rtol=1e-5)
        for field,value in [('projection_id','b'*64),('variant','zero' if name!='zero' else 'public'),('feature_space_id','0'*64)]:
            wrong=deepcopy(metadata);wrong[field]=value
            with pytest.raises(ValueError):load(fresh,path,wrong)


def test_independent_cpu_reader_recovers_complete_states_and_rejects_missing_steps(tmp_path,fitted):
    import hashlib
    from safetensors.torch import save_file
    from src.learning.language_response_audit import verify_saved_state
    from src.learning.language_response_router import protocol_id,feature_id
    from src.learning.public_math import canonical
    hidden,lengths,projection,features,targets,nets,fit,optimizers=fitted
    common={'protocol_id':protocol_id(),'feature_space_id':feature_id(),'scope':'public-training-only'}
    save_file({'hidden':hidden,'lengths':lengths,'targets':targets},str(tmp_path/'train-features.safetensors'),
        metadata={'format':'lixyswarm-public-response-features-v1','binding':canonical(common).decode()})
    save_file(projection,str(tmp_path/'projection.safetensors'),
        metadata={'format':'lixyswarm-public-response-projection-v1','binding':canonical(common).decode()})
    projection_id=hashlib.sha256((tmp_path/'projection.safetensors').read_bytes()).hexdigest()
    bindings={};predictions={}
    for name,net in nets.items():
        metadata=binding(net,name,projection_id);save(net,tmp_path/(name+'.safetensors'),metadata);bindings[name]=metadata
        with torch.no_grad():predictions[name]=net(features,use_memory=name!='dolphin').tolist()
    torch.save(optimizers,tmp_path/'optimizer-state.pt')
    artifacts={p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in tmp_path.iterdir()}
    rows=[{'index':i,'case':'case-'+str(i),'target':targets[i].tolist()} for i in range(65)]
    report={'protocol_id':protocol_id(),'feature_space_id':feature_id(),'fresh_evaluation_performed':False,
        'language_experts_updated':False,'model_promoted':False,'training_indices':list(range(65)),
        'case_ids':[v['case'] for v in rows],'fit':fit,'artifacts':artifacts,'bindings':bindings,'train_predictions':predictions}
    recovered=verify_saved_state(tmp_path,report,rows)
    assert recovered['success'] and recovered['new_optimizer_updates']==0
    assert recovered['language_model_loaded'] is False and recovered['new_generation_performed'] is False
    assert recovered['optimizer_states']['dolphin']['parameters_with_moments']==40
    assert recovered['optimizer_states']['public']['parameters_with_moments']==18
    changed=deepcopy(report);changed['fit']['phase2']['public']['steps']=1
    with pytest.raises(ValueError):verify_saved_state(tmp_path,changed,rows)
