"""Math supervision plus a common public replay descent constraint; tensor-only."""
import math
import time


def project_with_replay(new, replay):
    """Remove a conflicting component, then add common replay before final clipping."""
    import torch
    if not new or set(new)!=set(replay):raise ValueError('Math and replay gradients have different parameters')
    if any(new[name].shape!=replay[name].shape or not torch.isfinite(new[name]).all()
           or not torch.isfinite(replay[name]).all() for name in new):
        raise ValueError('Invalid math or replay gradient tensors')
    squared=sum(float(t.double().square().sum()) for t in replay.values())
    dot=sum(float((new[name].double()*replay[name].double()).sum()) for name in new)
    if not math.isfinite(squared) or not math.isfinite(dot):raise ValueError('Nonfinite replay projection')
    coefficient=min(0.,dot)/squared if squared else 0.
    result={name:(new[name].double()+replay[name].double()*(1.-coefficient)).to(new[name].dtype) for name in new}
    if any(not torch.isfinite(t).all() for t in result.values()):raise ValueError('Nonfinite projected replay gradient')
    return result


def backward(model, tokenizer, rows, device, deadline):
    import torch
    from .language_adapter import adapter_parameters
    from .language_policy import MATH_POLICY
    from .public_conversation import public_corpus
    parameters=adapter_parameters(model)
    def check():
        if time.time()>=deadline:raise TimeoutError('Math gradient deadline exceeded')
    def encode(prompt, answer, maximum, answer_maximum):
        text=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        prefix=tokenizer.encode(text,add_special_tokens=False)
        target=tokenizer.encode(answer,add_special_tokens=False)+[tokenizer.eos_token_id]
        if len(prefix)+len(target)>maximum or len(target)>answer_maximum:
            raise ValueError('Public replay or math example exceeds its immutable bound')
        return torch.tensor([prefix+target],device=device),torch.tensor([[-100]*len(prefix)+target],device=device),len(target)
    def supervise(examples):
        tokens=sum(row[2] for row in examples);loss_sum=0.
        for inputs,labels,count in examples:
            check();loss=model(input_ids=inputs,labels=labels,use_cache=False).loss
            (loss*count/tokens).backward();loss_sum+=float(loss.detach())*count/tokens
        return loss_sum,tokens
    # Only public training examples enter either gradient; evaluation is never replayed.
    new_loss,tokens=supervise([encode(row['prompt'],row['answer'],1024,512) for row in rows])
    new={name:p.grad.detach().float().cpu().clone() for name,p in parameters.items()}
    model.zero_grad(set_to_none=True)
    replay_rows=public_corpus()['train'][:MATH_POLICY['replay_examples']]
    replay_loss,replay_tokens=supervise([encode(row['prompt'],row['answer'],192,192) for row in replay_rows])
    replay={name:p.grad.detach().float().cpu().clone() for name,p in parameters.items()}
    combined=project_with_replay(new,replay)
    for name,p in parameters.items():p.grad=combined[name].to(p.device,dtype=p.dtype)
    torch.nn.utils.clip_grad_norm_(parameters.values(),1,error_if_nonfinite=True)
    return new_loss,replay_loss,tokens,replay_tokens
