"""Offline bounded text-label distillation; receives no transport/publisher key.

Different architectures exchange verified public text, never parameter tensors.
The parent supervises permissions, memory, cancellation and the deadline.
"""
import json
import math
from pathlib import Path
import random
import sys
import time

from .public_bases import BASES, base_hash
from .public_conversation import public_corpus
from .public_teaching import (TRANSFER_POLICY, digest, public_teaching_corpus, transfer_verdict,
                              verify_lessons, transfer_policy, MAX_LESSON_BYTES)
from .weights_io import digest_file


def validate_request(request):
    fields = {'action', 'base_directory', 'lessons_file', 'output_directory', 'teacher',
              'deadline', 'cpu_threads', 'gpu_fraction'}
    if (not isinstance(request, dict) or set(request) != fields or request['action'] != 'distill-public-v1'
            or type(request['cpu_threads']) is not int or not 1 <= request['cpu_threads'] <= 8
            or type(request['gpu_fraction']) not in (int, float) or not 0 < request['gpu_fraction'] <= 1
            or type(request['deadline']) not in (int, float) or not math.isfinite(request['deadline'])
            or not time.time() < request['deadline'] <= time.time() + 1201):
        raise ValueError('Invalid bounded student request')
    if any(not isinstance(request[k], str) or not Path(request[k]).is_absolute()
           for k in ['base_directory', 'lessons_file', 'output_directory']):
        raise ValueError('Only locally selected absolute public inputs are allowed')
    output = Path(request['output_directory']) / 'adapter.safetensors'
    if output.exists() or output.is_symlink():
        raise ValueError('Student output must be new')
    if not isinstance(request['teacher'], dict) or set(request['teacher']) != {
            'teacher_model_id', 'teacher_release_id', 'teacher_worker_id', 'requester_id'}:
        raise ValueError('Invalid exact teacher selection')
    with Path(request['lessons_file']).open('rb') as stream:
        raw = stream.read(MAX_LESSON_BYTES + 1)
    if len(raw) > MAX_LESSON_BYTES:
        raise ValueError('Public lessons exceed their bound')
    lessons = json.loads(raw)
    checked = verify_lessons(lessons, **request['teacher'])
    if len(checked['accepted']) < TRANSFER_POLICY['minimum_labels']:
        raise ValueError('Insufficient verified teacher labels')
    return lessons, checked


def run(request):
    lessons, checked = validate_request(request)
    began = time.monotonic()
    base = Path(request['base_directory'])
    name = 'qwen3-0.6b'
    def deadline():
        if time.time() >= request['deadline']:
            raise TimeoutError('Student experiment deadline exceeded')
    for filename, expected in BASES[name].items():
        deadline()
        if digest_file(base / filename)[0] != expected:
            raise ValueError('Pinned student public base mismatch')
    from src.release.language_catalog import INITIAL_BUNDLES
    parent_id = INITIAL_BUNDLES[name]['release']['model_artifact_id']
    if digest_file(base / 'adapter.safetensors')[0] != parent_id:
        raise ValueError('Student requires the fixed original small-family parent')
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
    from .language_adapter import install_adapter, load_adapter, save_adapter
    from .language_candidate_job import evaluate
    from .public_quality import preservation_verdict as factual_verdict
    from .public_language_gate import preservation_verdict as instruction_verdict
    torch.set_num_threads(request['cpu_threads'])
    torch.manual_seed(20261014)
    torch.cuda.set_per_process_memory_fraction(request['gpu_fraction'])
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, device_map={'': 'cuda'}).eval()
    parameters = install_adapter(model)
    load_adapter(model, base / 'adapter.safetensors', base_hash(name))
    output = Path(request['output_directory'])
    from src.network.telemetry import atomic_json
    def progress(phase, **extra):
        atomic_json(output / 'status.json', {'phase': phase, 'time': time.time(), **extra})
    def encode(row):
        text = tokenizer.apply_chat_template([{'role': 'user', 'content': row['prompt']}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prefix = tokenizer.encode(text, add_special_tokens=False)
        answer = tokenizer.encode(row['answer'], add_special_tokens=False) + [tokenizer.eos_token_id]
        if len(prefix) + len(answer) > 256:
            raise ValueError('Public lesson exceeds the bounded student context')
        return (torch.tensor([prefix + answer], device='cuda'),
                torch.tensor([[-100] * len(prefix) + answer], device='cuda'), len(prefix), len(answer))
    corpus = public_teaching_corpus()
    training = [encode(row) for row in checked['accepted']]
    heldout = [encode(row) for row in corpus['evaluation']]
    class Deadline(StoppingCriteria):
        def __call__(self, *_a, **_kw):
            return time.time() >= request['deadline']
    def transfer_evaluation():
        model.eval()
        total, tokens, results = 0, 0, []
        with torch.inference_mode():
            for i, (inputs, labels, prefix, count) in enumerate(heldout):
                deadline()
                loss = float(model(input_ids=inputs, labels=labels, use_cache=False).loss)
                if not math.isfinite(loss):
                    raise ValueError('Invalid public held-out loss')
                total += loss * count
                tokens += count
                prompt = inputs[:, :prefix]
                generated = model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                    max_new_tokens=48, do_sample=False, use_cache=True,
                    stopping_criteria=StoppingCriteriaList([Deadline()]))
                deadline()
                answer = tokenizer.decode(generated[0, prefix:], skip_special_tokens=True).strip()
                results.append({'index': i, 'answer': answer})
        return {'nll': total / tokens, 'tokens': tokens, 'results': results}
    def assess():
        return {'transfer': transfer_evaluation(), 'historical': evaluate(model, tokenizer, public_corpus(), 'cuda', request['deadline'])}
    progress('student_baseline')
    baseline = assess()
    atomic_json(output / 'baseline.json', baseline)
    # Retain this exact student's preceding behaviour on public anchor text.
    anchors = [tokenizer(text, return_tensors='pt', add_special_tokens=False).input_ids.to('cuda')
               for text in public_corpus()['retention']]
    with torch.no_grad():
        parent_probabilities = [torch.softmax(model(input_ids=x, use_cache=False).logits.float(), dim=-1).cpu() for x in anchors]
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    optimizer = torch.optim.AdamW(parameters.values(), lr=TRANSFER_POLICY['learning_rate'], weight_decay=0)
    order = list(range(len(training)))
    random.Random(20261014).shuffle(order)
    losses, presented_tokens = [], 0
    for step in range(TRANSFER_POLICY['steps']):
        deadline()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        indices = [order[(step * 4 + i) % len(order)] for i in range(4)]
        tokens = sum(training[i][3] for i in indices)
        loss_sum = 0
        for i in indices:
            inputs, labels, _, count = training[i]
            loss = model(input_ids=inputs, labels=labels, use_cache=False).loss
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite public teacher-label loss')
            (loss * count / tokens).backward()
            loss_sum += float(loss.detach()) * count / tokens
        index = step % len(anchors)
        student = model(input_ids=anchors[index], use_cache=False).logits.float()
        retention = torch.nn.functional.kl_div(torch.log_softmax(student, dim=-1),
            parent_probabilities[index].to('cuda'), reduction='sum') / anchors[index].shape[1]
        if not torch.isfinite(retention):
            raise ValueError('Nonfinite public retention loss')
        (retention * TRANSFER_POLICY['retention_weight']).backward()
        torch.nn.utils.clip_grad_norm_(parameters.values(), TRANSFER_POLICY['clip_norm'], error_if_nonfinite=True)
        optimizer.step()
        presented_tokens += tokens
        losses.append({'step': step + 1, 'supervised_loss': loss_sum, 'retention_loss': float(retention.detach())})
        progress('student_training', step=step + 1)
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    progress('student_evaluation')
    candidate = assess()
    transfer = transfer_verdict(baseline['transfer'], candidate['transfer'])
    factual = factual_verdict(baseline['historical']['quality'], candidate['historical']['quality'])
    instruction = instruction_verdict(baseline['historical']['instructions'], candidate['historical']['instructions'])
    eligible = transfer['eligible'] and factual['eligible_for_promotion'] and instruction['eligible_for_promotion']
    path = output / 'adapter.safetensors'
    deadline()
    save_adapter(model, path, base_hash(name))
    adapter_id, size = digest_file(path)
    if not eligible:
        path.unlink()  # Only this freshly generated, unapproved research output.
    policy = transfer_policy()
    return {'success': True, 'scope': 'Bounded public text-label transfer from accepted 1.7B to research 0.6B; no parameter mixing or publication',
        'base_name': name, 'base_hash': base_hash(name), 'student_parent_adapter_id': parent_id,
        'student_parent_was_chat_approved': False, 'teacher': request['teacher'], 'lessons_id': digest(lessons),
        'policy': policy, 'policy_id': digest(policy), 'accepted_teacher_labels': len(training),
        'rejected_teacher_labels': checked['rejected'], 'training_presentations': 64,
        'training_unique_examples': len(training), 'presented_answer_tokens': presented_tokens,
        'baseline': baseline, 'candidate': candidate, 'transfer_verdict': transfer,
        'factual_verdict': factual, 'instruction_verdict': instruction,
        'eligible_for_research_transfer': eligible, 'candidate_adapter_id': adapter_id, 'candidate_adapter_bytes': size,
        'candidate_weights_retained': bool(eligible), 'published': False, 'private_inputs': False,
        'native_coordinator_cycles_added': 0, 'incorporated_tokens': 0,
        'seconds': time.monotonic() - began, 'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(), 'losses': losses,
        'limitations': ['Deterministic public labels also admit an oracle; no advantage over oracle-labelled SFT claimed',
            'Only synthetic Spanish instruction transfer; no broad language or scaling claim',
            'Tensor child is supervised, but full filesystem isolation remains pending',
            'Research student is not registered, selected, replicated or chat-approved by this experiment']}


def main():
    raw = sys.stdin.buffer.read(16 * 1024 + 1)
    if len(raw) > 16 * 1024:
        raise ValueError('Student request exceeds its bound')
    print(json.dumps(run(json.loads(raw)), ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
