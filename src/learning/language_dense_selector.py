"""Small learned pings and Matriarca over a pinned frozen query representation."""
from copy import deepcopy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from safetensors.torch import save_file,load_file

from src.matriarca.matriarca import MatriarcaConfig,MatriarcaModel
from .language_selector import tensor_digest
from .language_prefix_router import feature_id,protocol_id,definition


class DenseEcholocation(nn.Module):
    def __init__(self,dimensions):
        super().__init__();self.dimensions=dimensions
        self.pings=nn.ModuleList([nn.Sequential(nn.Linear(dimensions,16),nn.GELU(),nn.Linear(16,8),nn.LayerNorm(8)) for _ in range(5)])
        self.triangulation=nn.MultiheadAttention(8,4,dropout=0,batch_first=True)
        self.fusion=nn.Sequential(nn.Linear(8,8),nn.LayerNorm(8),nn.Tanh())

    def forward(self,features):
        if (features.ndim!=2 or features.shape[1]!=self.dimensions or features.dtype!=torch.float32
                or features.requires_grad or not torch.isfinite(features).all()):
            raise ValueError('Pings require finite frozen FP32 query features')
        values=F.layer_norm(features,(self.dimensions,))
        echoes=torch.stack([ping(values) for ping in self.pings],dim=1)
        triangulated,_=self.triangulation(echoes[:,:1],echoes,echoes,need_weights=False)
        return self.fusion(triangulated[:,0])


class DenseExpertSelector(nn.Module):
    def __init__(self,dimensions=2048,*,support_count=32):
        super().__init__()
        if type(dimensions) is not int or not 1<=dimensions<=4096 or type(support_count) is not int or not 1<=support_count<=512:
            raise ValueError('Unsupported frozen-prefix selector dimensions')
        self.dolphin=DenseEcholocation(dimensions);self.expert_head=nn.Linear(8,2)
        self.memory=MatriarcaModel(MatriarcaConfig(embd_dim=16,infrasound_dim=8,n_heads=4,n_layers=1,dropout=0))
        self.memory_vote=nn.Sequential(nn.Linear(8,2),nn.Tanh())
        self.register_buffer('public_memory',torch.zeros(support_count,16))

    def forward(self,features,*,use_memory=True):
        values=self.dolphin(features);scores=self.expert_head(values)
        if use_memory:scores=.8*scores+.2*self.memory_vote(self.memory(F.pad(values,(0,8)),self.public_memory))
        return scores

    @torch.no_grad()
    def choose(self,features,*,use_memory=True):
        probabilities=self(features,use_memory=use_memory).softmax(-1)
        if not torch.isfinite(probabilities).all():raise ValueError('Nonfinite frozen-prefix route')
        return probabilities.argmax(-1),probabilities


def train_dense_selectors(features,targets,p,*,check=lambda:None,progress=lambda *_:None):
    if (features.ndim!=2 or len(features)<8 or features.requires_grad or features.dtype!=torch.float32
            or not torch.isfinite(features).all() or targets.shape!=(len(features),2) or targets.dtype!=torch.float32
            or not torch.isfinite(targets).all() or torch.any(targets<0) or torch.any(targets>1)
            or not torch.allclose(targets.sum(-1),torch.ones_like(targets[:,0]),atol=1e-6,rtol=0)):
        raise ValueError('Invalid frozen-prefix TRAIN features or verified outcome targets')
    torch.manual_seed(p['seed']);support=list(range(0,len(features),p['support_stride']))
    queries=[i for i in range(len(features)) if i not in support]
    model=DenseExpertSelector(features.shape[1],support_count=len(support)).to(features.device)
    initial=tensor_digest(model.dolphin.state_dict())
    def fit(net,indices,parameters,*,memory,seed,phase):
        generator=torch.Generator().manual_seed(seed)
        optimizer=torch.optim.AdamW(parameters,lr=p['learning_rate'],weight_decay=p['weight_decay'])
        first=last=None
        for step in range(p['phase_steps']):
            check();batch=[indices[i] for i in torch.randperm(len(indices),generator=generator)[:p['batch_size']].tolist()]
            optimizer.zero_grad(set_to_none=True)
            loss=-(targets[batch]*net(features[batch],use_memory=memory).log_softmax(-1)).sum(-1).mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite prefix-selector TRAIN loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step()
            last=float(loss.detach());first=last if first is None else first
            if step%16==0 or step+1==p['phase_steps']:progress(phase,step+1,p['phase_steps'])
        return {'steps':p['phase_steps'],'first_batch_loss':first,'last_batch_loss':last},optimizer.state_dict()
    phase1,optimizer1=fit(model,list(range(len(features))),list(model.dolphin.parameters())+list(model.expert_head.parameters()),
        memory=False,seed=p['seed']+1,phase='dolphin')
    model.eval()
    with torch.no_grad():bank=F.pad(torch.cat([model.dolphin(features[support]),targets[support]],-1),(0,6))
    encoded=tensor_digest(model.dolphin.state_dict());variants={};phase2={};optimizers={}
    permutation=torch.randperm(len(support),generator=torch.Generator().manual_seed(p['seed']+2)).to(features.device)
    for mode in p['memory_modes']:
        net=deepcopy(model)
        for parameter in net.parameters():parameter.requires_grad_(False)
        for parameter in [*net.memory.parameters(),*net.memory_vote.parameters()]:parameter.requires_grad_(True)
        with torch.no_grad():
            net.public_memory.copy_(bank)
            if mode=='zero':net.public_memory.zero_()
            elif mode=='shuffled':net.public_memory[:,8:10]=bank[permutation,8:10]
            elif mode!='public':raise ValueError('Unknown prefix memory comparison')
        trainable=[v for v in net.parameters() if v.requires_grad]
        phase2[mode],optimizers[mode]=fit(net,queries,trainable,memory=True,seed=p['seed']+3,phase='memory-'+mode)
        net.eval();variants[mode]=net
        if tensor_digest(net.dolphin.state_dict())!=encoded:raise ValueError('Prefix memory changed its frozen learned feature space')
    variants['dolphin']=model
    return variants,{'phase1':phase1,'phase2':phase2,'support_indices':support,'query_indices':queries,
        'dolphin_state_id':encoded,'initial_dolphin_state_id':initial,
        'dolphin_parameter_count':sum(v.numel() for v in model.dolphin.parameters()),
        'memory_scope':'public-training-only','prediction_rows_used_for_fit':False}, {'dolphin':optimizer1,'memory':optimizers}


def binding(model,variant):
    if model.dolphin.dimensions!=definition()['feature_definition']['dimensions']:
        raise ValueError('Dense selector dimensions do not match its pinned representation')
    return {'protocol_id':protocol_id(),'feature_space_id':feature_id(),'expert_ids':definition()['expert_ids'],
        'dolphin_state_id':tensor_digest(model.dolphin.state_dict()),'variant':variant,'memory_scope':'public-training-only',
        'dimensions':model.dolphin.dimensions,'support_count':len(model.public_memory)}


def save_dense(model,path,metadata):
    if metadata!=binding(model,metadata.get('variant')) or metadata['variant'] not in {'dolphin','zero','shuffled','public'}:
        raise ValueError('Invalid dense selector binding')
    tensors={k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}
    if any(not torch.isfinite(v).all() for v in tensors.values()) or sum(v.numel()*v.element_size() for v in tensors.values())>2*1024**2:
        raise ValueError('Dense selector exceeds its finite learned-state bound')
    save_file(tensors,str(path),metadata={'format':'lixyswarm-frozen-prefix-selector-v1','binding':json.dumps(metadata,sort_keys=True,separators=(',',':'))})


def load_dense(model,path,metadata):
    expected=binding(model,metadata.get('variant'))
    expected['dolphin_state_id']=metadata.get('dolphin_state_id')
    if metadata!=expected or metadata['variant'] not in {'dolphin','zero','shuffled','public'}:
        raise ValueError('Dense selector scope or feature binding mismatch')
    path=Path(path)
    if path.stat().st_size>2*1024**2:raise ValueError('Dense selector file exceeds its bound')
    with safe_open(path,framework='pt',device='cpu') as reader:
        if reader.metadata()!={'format':'lixyswarm-frozen-prefix-selector-v1','binding':json.dumps(metadata,sort_keys=True,separators=(',',':'))}:
            raise ValueError('Dense selector artifact binding mismatch')
    values=load_file(str(path),device='cpu');state=model.state_dict()
    if set(values)!=set(state) or any(values[k].shape!=v.shape or values[k].dtype!=v.dtype or not torch.isfinite(values[k]).all() for k,v in state.items()):
        raise ValueError('Dense selector tensor schema mismatch')
    if tensor_digest({k[len('dolphin.'):]:v for k,v in values.items() if k.startswith('dolphin.')})!=metadata['dolphin_state_id']:
        raise ValueError('Dense selector learned Dolphin does not match its memory')
    model.load_state_dict(values,strict=True)
