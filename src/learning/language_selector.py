"""Learned public-only Dolphin/Matriarca choice between frozen language experts.

Only tensor engines import this module. Query prefixes, never reference answers,
enter the selector. It records no query history or personal memory.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import save_file, load_file
from safetensors import safe_open

from src.agents.dolphin_agent import DolphinConfig, Echolocation
from src.matriarca.matriarca import MatriarcaConfig, MatriarcaModel


def tensor_digest(tensors):
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        value=tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name,list(value.shape),str(value.dtype)],separators=(',',':')).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def pad_queries(queries, device):
    if not queries or any(not row or len(row)>1024 for row in queries):
        raise ValueError('Selector requires bounded nonempty query prefixes')
    tokens=torch.zeros(len(queries),max(map(len,queries)),dtype=torch.long,device=device)
    mask=torch.zeros_like(tokens)
    for i,row in enumerate(queries):
        tokens[i,:len(row)]=torch.tensor(row,dtype=torch.long,device=device)
        mask[i,:len(row)]=1
    return tokens,mask


class LanguageExpertSelector(nn.Module):
    def __init__(self,vocab_size,*,n_embd=32,support_count=98):
        super().__init__()
        if type(vocab_size) is not int or not 1<=vocab_size<=160000 or n_embd!=32 or type(support_count) is not int or not 1<=support_count<=512:
            raise ValueError('Unsupported bounded language selector architecture')
        self.vocab_size=vocab_size;self.n_embd=n_embd
        self.dolphin=Echolocation(DolphinConfig(vocab_size=vocab_size,n_embd=n_embd,
            feromon_dim=8,echo_dim=8,dropout=0))
        self.expert_head=nn.Linear(8,2)
        self.memory=MatriarcaModel(MatriarcaConfig(embd_dim=16,infrasound_dim=8,
            n_heads=4,n_layers=1,dropout=0))
        self.memory_vote=nn.Sequential(nn.Linear(8,2),nn.Tanh())
        self.register_buffer('public_memory',torch.zeros(support_count,16))

    def features(self,tokens,mask):
        return self.dolphin(tokens,mask)[0]

    def forward(self,tokens,mask,*,use_memory=True):
        features=self.features(tokens,mask)
        scores=self.expert_head(features)
        if use_memory:
            query=F.pad(features,(0,8))
            vote=self.memory_vote(self.memory(query,self.public_memory))
            scores=.8*scores+.2*vote
        return scores

    @torch.no_grad()
    def choose(self,queries,*,use_memory=True):
        tokens,mask=pad_queries(queries,next(self.parameters()).device)
        scores=self(tokens,mask,use_memory=use_memory)
        if not torch.isfinite(scores).all():raise ValueError('Nonfinite selector scores')
        probabilities=scores.softmax(dim=-1)
        return probabilities.argmax(dim=-1),probabilities


def train_selectors(queries,teacher_nll,vocab_size,protocol,*,device='cpu',check=lambda:None,progress=lambda *_:None):
    """Both phases see TRAIN only; all memory variants use the same budget."""
    if (teacher_nll.shape!=(len(queries),2) or not torch.isfinite(teacher_nll).all()
            or torch.any(teacher_nll<0) or len(queries)<8):
        raise ValueError('Invalid public TRAIN expert losses')
    torch.manual_seed(protocol['seed'])
    support=list(range(0,len(queries),protocol['support_stride']))
    query_indices=[i for i in range(len(queries)) if i not in support]
    model=LanguageExpertSelector(vocab_size,n_embd=protocol['dolphin_n_embd'],support_count=len(support)).to(device)
    targets=(-teacher_nll.float()/protocol['teacher_temperature']).softmax(dim=-1).to(device)
    def fit(net,indices,parameters,*,memory,seed,phase):
        generator=torch.Generator().manual_seed(seed)
        optimizer=torch.optim.AdamW(parameters,lr=protocol['learning_rate'],weight_decay=protocol['weight_decay'])
        first=last=None
        for step in range(protocol['phase_steps']):
            check()
            batch=[indices[i] for i in torch.randperm(len(indices),generator=generator)[:protocol['batch_size']].tolist()]
            tokens,mask=pad_queries([queries[i] for i in batch],device)
            optimizer.zero_grad(set_to_none=True)
            scores=net(tokens,mask,use_memory=memory)
            loss=-(targets[batch]*scores.log_softmax(dim=-1)).sum(dim=-1).mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite public selector training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step()
            value=float(loss.detach());first=value if first is None else first;last=value
            if step%16==0 or step+1==protocol['phase_steps']:progress(phase,step+1,protocol['phase_steps'])
        return {'steps':protocol['phase_steps'],'first_batch_loss':first,'last_batch_loss':last},optimizer.state_dict()
    first_parameters=list(model.dolphin.parameters())+list(model.expert_head.parameters())
    # Confidence and memory-encoding branches not used by this objective are
    # intentionally not claimed as trained semantic labels.
    initial_encoder_id=tensor_digest(model.dolphin.state_dict())
    phase1,optimizer1=fit(model,list(range(len(queries))),first_parameters,memory=False,
        seed=protocol['seed']+1,phase='dolphin')
    model.eval()
    with torch.no_grad():
        tokens,mask=pad_queries([queries[i] for i in support],device)
        features=model.features(tokens,mask)
        bank=F.pad(torch.cat([features,targets[support]],dim=-1),(0,6))
    encoder_id=tensor_digest(model.dolphin.state_dict())
    base=deepcopy(model.state_dict());variants={};training={};optimizers={}
    permutation=torch.randperm(len(support),generator=torch.Generator().manual_seed(protocol['seed']+2)).to(device)
    for mode in protocol['memory_modes']:
        check();net=deepcopy(model);net.load_state_dict(base,strict=True)
        for p in net.parameters():p.requires_grad_(False)
        for p in list(net.memory.parameters())+list(net.memory_vote.parameters()):p.requires_grad_(True)
        with torch.no_grad():
            net.public_memory.copy_(bank)
            if mode=='zero':net.public_memory.zero_()
            elif mode=='shuffled':net.public_memory[:,8:10]=bank[permutation,8:10]
            elif mode!='public':raise ValueError('Unknown public memory ablation')
        parameters=[p for p in net.parameters() if p.requires_grad]
        training[mode],optimizers[mode]=fit(net,query_indices,parameters,memory=True,
            seed=protocol['seed']+3,phase='memory-'+mode)
        net.eval();variants[mode]=net
        if tensor_digest(net.dolphin.state_dict())!=encoder_id:raise ValueError('Memory training changed its frozen feature space')
    model.eval();variants['dolphin']=model
    return variants,{'phase1':phase1,'phase2':training,'support_indices':support,
        'query_indices':query_indices,'dolphin_state_id':encoder_id,'initial_dolphin_state_id':initial_encoder_id,
        'memory_feature_space':protocol.get('memory_feature_space','lixyswarm-public-language-expert-echo-v1'),
        'memory_scope':'public-training-only','held_out_used_for_training_or_selection':False},{
        'dolphin':optimizer1,'memory':optimizers}


class RoutedFrozenExperts:
    """The selected expert executes once; reference answers never enter routing."""
    def __init__(self,model,selector,*,use_memory=True):
        self.model=model;self.selector=selector;self.use_memory=use_memory

    def eval(self):
        self.model.eval();self.selector.eval();return self

    def _choice(self,inputs,labels=None):
        if inputs.ndim!=2 or inputs.shape[0]!=1:raise ValueError('Routed language execution requires one bounded query')
        prefix=inputs[0]
        if labels is not None:
            if labels.shape!=inputs.shape:raise ValueError('Routed labels must match the language sequence')
            positions=(labels[0]!=-100).nonzero().flatten()
            if not len(positions) or int(positions[0])==0 or torch.any(labels[0,int(positions[0]):]==-100):
                raise ValueError('A reference answer must follow a nonempty masked query prefix')
            prefix=prefix[:int(positions[0])]
        choices,_=self.selector.choose([prefix.tolist()],use_memory=self.use_memory)
        choice=int(choices[0])
        if choice not in {0,1}:raise ValueError('Selector chose an undeclared frozen expert')
        return choice

    def __call__(self,*,input_ids,labels=None,**kwargs):
        from .language_adapter import adapters_enabled
        choice=self._choice(input_ids,labels)
        with adapters_enabled(self.model,choice==0):
            return self.model(input_ids=input_ids,labels=labels,**kwargs)

    def generate(self,*,input_ids,**kwargs):
        from .language_adapter import adapters_enabled
        choice=self._choice(input_ids)
        with adapters_enabled(self.model,choice==0):
            return self.model.generate(input_ids=input_ids,**kwargs)


def validate_binding(metadata):
    fields={'protocol_id','base_hash','tokenizer_id','dolphin_state_id','feature_space','memory_scope','variant'}
    version2=isinstance(metadata,dict) and 'expert_ids' in metadata
    if version2:
        fields.add('expert_ids')
        ids=metadata['expert_ids']
        if (not isinstance(ids,list) or len(ids)!=2 or len(set(v for v in ids if isinstance(v,str)))!=2
                or any(not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for v in ids)):
            raise ValueError('Ordered frozen expert identities are invalid')
    if (not isinstance(metadata,dict) or set(metadata)!=fields or metadata['memory_scope']!='public-training-only'
            or metadata['feature_space']!=('lixyswarm-public-language-adapter-selection-v2' if version2 else 'lixyswarm-public-language-expert-echo-v1')
            or metadata['variant'] not in {'zero','shuffled','public','dolphin'}
            or any(not isinstance(metadata[key],str) or len(metadata[key])!=64 or any(c not in '0123456789abcdef' for c in metadata[key])
                for key in ['protocol_id','base_hash','tokenizer_id','dolphin_state_id'])):
        raise ValueError('Invalid selector feature space or public scope binding')


def save_selector(model,path,metadata):
    validate_binding(metadata)
    tensors={k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}
    if (any(not torch.isfinite(v).all() for v in tensors.values())
            or sum(v.numel()*v.element_size() for v in tensors.values())>31*1024**2):raise ValueError('Invalid bounded selector state')
    if tensor_digest(model.dolphin.state_dict())!=metadata['dolphin_state_id']:raise ValueError('Memory feature space differs from its frozen Dolphin')
    save_file(tensors,str(path),metadata={'format':'lixyswarm-language-selector-v2' if 'expert_ids' in metadata else 'lixyswarm-language-selector-v1','binding':json.dumps(metadata,sort_keys=True,separators=(',',':'))})


def load_selector(model,path,metadata):
    validate_binding(metadata)
    path=Path(path)
    if path.stat().st_size>32*1024**2:raise ValueError('Selector artifact exceeds its bound')
    expected={'format':'lixyswarm-language-selector-v2' if 'expert_ids' in metadata else 'lixyswarm-language-selector-v1','binding':json.dumps(metadata,sort_keys=True,separators=(',',':'))}
    with safe_open(path,framework='pt',device='cpu') as source:
        if source.metadata()!=expected or metadata.get('memory_scope')!='public-training-only':raise ValueError('Selector feature space or public scope mismatch')
    state=load_file(str(path),device='cpu');parameters=model.state_dict()
    if set(state)!=set(parameters) or any(state[k].shape!=v.shape or state[k].dtype!=v.dtype or not torch.isfinite(state[k]).all()
        for k,v in parameters.items()):raise ValueError('Selector tensor names, shapes, types or values mismatch')
    if tensor_digest({k[len('dolphin.'):]:v for k,v in state.items() if k.startswith('dolphin.')})!=metadata['dolphin_state_id']:
        raise ValueError('Selector artifact has a different frozen Dolphin feature space')
    model.load_state_dict(state,strict=True)
