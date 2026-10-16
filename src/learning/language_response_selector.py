"""Small learned pings and public outcome memory over a TRAIN-fitted projection."""
from copy import deepcopy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from safetensors.torch import save_file,load_file

from src.matriarca.matriarca import MatriarcaConfig,MatriarcaModel
from .language_dense_selector import DenseEcholocation
from .language_selector import tensor_digest
from .language_response_router import definition,protocol_id,feature_id


def fit_projection(hidden,lengths):
    components=definition()['projection_components']
    if (hidden.ndim!=2 or len(hidden)<=components or hidden.shape[1]<components or hidden.requires_grad
            or hidden.dtype!=torch.float32 or lengths.shape!=(len(hidden),2) or lengths.requires_grad
            or lengths.dtype!=torch.float32 or not torch.isfinite(hidden).all() or not torch.isfinite(lengths).all()
            or torch.any(lengths<0) or torch.any(lengths>1)):
        raise ValueError('Projection accepts finite detached TRAIN responses and bounded token lengths only')
    x=hidden.cpu().double();mean=x.mean(0);_,_,basis=torch.linalg.svd(x-mean,full_matrices=False)
    basis=basis[:components]
    signs=basis.gather(1,basis.abs().argmax(1,keepdim=True)).sign()
    basis=(basis*torch.where(signs==0,torch.ones_like(signs),signs)).float().contiguous();mean=mean.float()
    raw=torch.cat([(hidden.cpu()-mean)@basis.T,lengths.cpu()],-1)
    return {'mean':mean,'basis':basis,'center':raw.mean(0),'scale':raw.std(0,unbiased=False).clamp_min(1e-4)}


def project(hidden,lengths,projection):
    if (hidden.ndim!=2 or lengths.shape!=(len(hidden),2) or hidden.dtype!=torch.float32 or lengths.dtype!=torch.float32
            or hidden.requires_grad or lengths.requires_grad or not torch.isfinite(hidden).all()
            or not torch.isfinite(lengths).all() or torch.any(lengths<0) or torch.any(lengths>1)):
        raise ValueError('Response projection inputs must be detached, finite and bounded')
    values={k:v.to(hidden.device) for k,v in projection.items()}
    result=(torch.cat([(hidden-values['mean'])@values['basis'].T,lengths],-1)-values['center'])/values['scale']
    if not torch.isfinite(result).all():raise ValueError('Nonfinite response representation')
    return result.detach()


class ResponseSelector(nn.Module):
    def __init__(self,support_count):
        super().__init__();self.dolphin=DenseEcholocation(18);self.utility_head=nn.Linear(8,4)
        self.memory=MatriarcaModel(MatriarcaConfig(embd_dim=16,infrasound_dim=8,n_heads=4,n_layers=1,dropout=0))
        self.memory_vote=nn.Linear(8,4);self.register_buffer('public_memory',torch.zeros(support_count,16))

    def forward(self,features,*,use_memory=True):
        echoes=self.dolphin(features);logits=self.utility_head(echoes)
        if use_memory:logits=logits+.5*self.memory_vote(self.memory(F.pad(echoes,(0,8)),self.public_memory)).tanh()
        return logits.sigmoid()


def train(features,targets,*,check=lambda:None,progress=lambda *_:None):
    p=definition()
    if (features.ndim!=2 or features.shape[1]!=18 or len(features)<20 or features.requires_grad
            or features.dtype!=torch.float32 or not torch.isfinite(features).all()
            or targets.shape!=(len(features),4) or targets.dtype!=torch.float32 or targets.requires_grad
            or not torch.isfinite(targets).all() or torch.any(targets<0) or torch.any(targets>1)):
        raise ValueError('Utility learning requires finite frozen TRAIN features and observed outcome/cost targets')
    torch.manual_seed(p['seed']);support=list(range(0,len(features),p['support_stride']))
    queries=[i for i in range(len(features)) if i not in support]
    model=ResponseSelector(len(support)).to(features.device)
    def fit(net,indices,parameters,memory,seed,phase):
        generator=torch.Generator().manual_seed(seed);parameters=list(parameters)
        optimizer=torch.optim.AdamW(parameters,lr=p['learning_rate'],weight_decay=p['weight_decay'])
        first=last=None
        for step in range(p['phase_steps']):
            check();batch=[indices[i] for i in torch.randperm(len(indices),generator=generator)[:p['batch_size']].tolist()]
            optimizer.zero_grad(set_to_none=True);pred=net(features[batch],use_memory=memory)
            loss=(pred[:,:2]-targets[batch,:2]).square().mean()+.25*(pred[:,2:]-targets[batch,2:]).square().mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite public utility learning loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step()
            last=float(loss.detach());first=last if first is None else first
            if step%32==0 or step+1==p['phase_steps']:progress(phase,step+1,p['phase_steps'])
        return {'steps':p['phase_steps'],'first_batch_loss':first,'last_batch_loss':last},optimizer.state_dict()
    phase1,opt1=fit(model,list(range(len(features))),[*model.dolphin.parameters(),*model.utility_head.parameters()],
        False,p['seed']+1,'dolphin')
    model.eval()
    with torch.no_grad():bank=F.pad(torch.cat([model.dolphin(features[support]),targets[support]],-1),(0,4))
    dolphin_id=tensor_digest(model.dolphin.state_dict());variants={'dolphin':model};phase2={};opts={}
    permutation=torch.randperm(len(support),generator=torch.Generator().manual_seed(p['seed']+2)).to(features.device)
    for mode in p['memory_modes']:
        net=deepcopy(model)
        for parameter in net.parameters():parameter.requires_grad_(False)
        for parameter in [*net.memory.parameters(),*net.memory_vote.parameters()]:parameter.requires_grad_(True)
        # This encoder stores new arbitrary memories in the general module;
        # this experiment only reads an already fixed public support bank.
        for parameter in net.memory.memory_encoder.parameters():parameter.requires_grad_(False)
        with torch.no_grad():
            net.public_memory.copy_(bank)
            if mode=='zero':net.public_memory.zero_()
            elif mode=='shuffled':net.public_memory[:,8:12]=bank[permutation,8:12]
        phase2[mode],opts[mode]=fit(net,queries,[v for v in net.parameters() if v.requires_grad],
            True,p['seed']+3,'memory-'+mode)
        net.eval();variants[mode]=net
        if tensor_digest(net.dolphin.state_dict())!=dolphin_id:raise ValueError('Memory changed its frozen learned Dolphin space')
    return (variants,{'phase1':phase1,'phase2':phase2,'support_indices':support,'query_indices':queries,
        'dolphin_state_id':dolphin_id,'projection_fit_used_test':False,'memory_scope':'public-training-only'},
        {'dolphin':opt1,'memory':opts})


def binding(model,variant,projection_id):
    if variant not in {'dolphin','zero','shuffled','public'} or len(projection_id)!=64:
        raise ValueError('Invalid response-memory variant or learned projection binding')
    return {'protocol_id':protocol_id(),'feature_space_id':feature_id(),'projection_id':projection_id,
        'expert_ids':definition()['expert_ids'],'variant':variant,'scope':'public-training-only',
        'support_count':len(model.public_memory),'dolphin_state_id':tensor_digest(model.dolphin.state_dict())}


def save(model,path,metadata):
    if metadata!=binding(model,metadata['variant'],metadata['projection_id']):raise ValueError('Response selector binding mismatch')
    values={k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}
    if any(not torch.isfinite(v).all() for v in values.values()):raise ValueError('Nonfinite learned response state')
    save_file(values,str(path),metadata={'format':'lixyswarm-response-selector-v1','binding':canonical_metadata(metadata)})


def canonical_metadata(value):return json.dumps(value,sort_keys=True,separators=(',',':'))


def load(model,path,metadata):
    expected=binding(model,metadata['variant'],metadata['projection_id']);expected['dolphin_state_id']=metadata['dolphin_state_id']
    if metadata!=expected or Path(path).stat().st_size>1024**2:raise ValueError('Response selector scope or size changed')
    with safe_open(path,framework='pt',device='cpu') as reader:
        if reader.metadata()!={'format':'lixyswarm-response-selector-v1','binding':canonical_metadata(metadata)}:
            raise ValueError('Response selector feature space, projection, or expert identity changed')
    values=load_file(str(path),device='cpu');state=model.state_dict()
    if set(values)!=set(state) or any(values[k].shape!=v.shape or values[k].dtype!=v.dtype or not torch.isfinite(values[k]).all() for k,v in state.items()):
        raise ValueError('Response selector tensor schema mismatch')
    if tensor_digest({k[8:]:v for k,v in values.items() if k.startswith('dolphin.')})!=metadata['dolphin_state_id']:
        raise ValueError('Response memory is attached to another learned Dolphin')
    model.load_state_dict(values,strict=True)
