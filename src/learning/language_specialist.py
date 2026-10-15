"""Public math specialization and transactionally selected immutable adapters."""
from contextlib import contextmanager
import math

import torch

from .language_adapter import adapters_enabled,adapter_parameters,AdapterLinear
from .language_selector import RoutedFrozenExperts


def fit_specialist(model,parameters,math_examples,replay_examples,protocol,*,device,check=lambda:None,progress=lambda *_:None):
    """A fixed single epoch. No evaluation, selection or remote identity here."""
    batch=protocol['specialist_batch_size'];replay_count=protocol['specialist_replay_examples']
    if len(math_examples)!=protocol['specialist_training_examples'] or len(math_examples)%batch or not replay_examples:
        raise ValueError('Specialist requires its complete fixed public TRAIN partition')
    if not parameters or any(not p.requires_grad for p in parameters.values()):
        raise ValueError('Specialist training requires the locally selected adapter parameters')
    owned=adapter_parameters(model)
    if set(parameters)!=set(owned) or any(p is not owned[name] for name,p in parameters.items()):
        raise ValueError('Specialist updates must cover exactly the owned adapter parameters')
    initial={name:p.detach().clone() for name,p in parameters.items()};previous=model.training;completed=False
    optimizer=torch.optim.AdamW(list(parameters.values()),lr=protocol['specialist_learning_rate'],
        betas=tuple(protocol['specialist_betas']),eps=protocol['specialist_epsilon'],weight_decay=0.)
    order=torch.randperm(len(math_examples),generator=torch.Generator().manual_seed(protocol['seed'])).tolist()
    trace=[];model.train()
    def supervise(rows,weight):
        count=sum(len(answer) for prefix,answer in rows);total=0.
        for prefix,answer in rows:
            check();inputs=torch.tensor([prefix+answer],device=device)
            labels=torch.tensor([[-100]*len(prefix)+answer],device=device)
            loss=model(input_ids=inputs,labels=labels,use_cache=False).loss
            value=float(loss.detach())
            if not math.isfinite(value) or value<0:raise ValueError('Nonfinite specialist TRAIN loss')
            (loss*len(answer)/count*weight).backward();total+=value*len(answer)/count
        return total,count
    try:
        for step,start in enumerate(range(0,len(order),batch)):
            check();optimizer.zero_grad(set_to_none=True)
            current=[math_examples[i] for i in order[start:start+batch]]
            replay_indices=[(step*replay_count+i)%len(replay_examples) for i in range(replay_count)]
            replay=[replay_examples[i] for i in replay_indices]
            math_loss,math_tokens=supervise(current,1.)
            replay_loss,replay_tokens=supervise(replay,protocol['specialist_replay_weight'])
            norm=float(torch.nn.utils.clip_grad_norm_(list(parameters.values()),1.,error_if_nonfinite=True))
            rate=protocol['specialist_learning_rate']*min(1.,(step+1)/protocol['specialist_warmup_steps'])
            for group in optimizer.param_groups:group['lr']=rate
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in parameters.values()):raise ValueError('Nonfinite specialist adapter update')
            trace.append({'step':step+1,'math_indices':order[start:start+batch],
                'replay_indices':replay_indices,'math_nll':math_loss,'replay_nll':replay_loss,
                'math_answer_tokens':math_tokens,'replay_answer_tokens':replay_tokens,'gradient_norm_before_clip':norm,'learning_rate':rate})
            progress('specialist-train',step+1,len(order)//batch)
        check();completed=True
        return {'steps':len(trace),'math_examples':len(order),'math_order':order,'steps_trace':trace,
            'public_training_only':True,'evaluation_used':False,'generalist_file_updated':False},optimizer.state_dict()
    finally:
        if not completed:
            with torch.no_grad():
                for name,p in parameters.items():p.copy_(initial[name])
        model.zero_grad(set_to_none=True);model.train(previous)


class FrozenAdapterBank:
    """Two complete adapter states, copied only after validating both in full."""
    def __init__(self,model,parameters,states):
        if len(states)!=2 or not parameters or any(p.requires_grad for p in parameters.values()):
            raise ValueError('The two adapter experts must be frozen before routing')
        owned={id(p) for module in model.modules() if isinstance(module,AdapterLinear) for p in [module.lora_a,module.lora_b]}
        if {id(p) for p in parameters.values()}!=owned:raise ValueError('Frozen experts must cover exactly the owned adapters')
        for state in states:
            if set(state)!=set(parameters) or any(t.shape!=parameters[name].shape or t.dtype!=parameters[name].dtype
                or not torch.isfinite(t).all() for name,t in state.items()):
                raise ValueError('Frozen expert adapter names, shapes, types or values mismatch')
        self.model=model;self.parameters=parameters
        self.states=[{name:t.detach().to(parameters[name].device).clone() for name,t in state.items()} for state in states]

    @contextmanager
    def activate(self,choice):
        if type(choice) is not int or choice not in {0,1}:raise ValueError('Unknown frozen adapter expert')
        originals={name:p.detach().clone() for name,p in self.parameters.items()}
        try:
            with torch.no_grad():
                for name,p in self.parameters.items():p.copy_(self.states[choice][name])
            with adapters_enabled(self.model,True):yield self.model
        finally:
            with torch.no_grad():
                for name,p in self.parameters.items():p.copy_(originals[name])


class RoutedAdapterExperts(RoutedFrozenExperts):
    def __init__(self,bank,selector,*,use_memory=True):
        super().__init__(bank.model,selector,use_memory=use_memory);self.bank=bank

    def __call__(self,*,input_ids,labels=None,**kwargs):
        with self.bank.activate(self._choice(input_ids,labels)) as model:
            return model(input_ids=input_ids,labels=labels,**kwargs)

    def generate(self,*,input_ids,**kwargs):
        with self.bank.activate(self._choice(input_ids)) as model:
            return model.generate(input_ids=input_ids,**kwargs)
