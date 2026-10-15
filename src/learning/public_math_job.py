"""Offline evaluation child for pinned public math; no optimizer, network or keys."""
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

from .public_math import (ARTIFACTS, canonical, curriculum_id, gate_definition, gate_id,
                          load_snapshot, prompt, score_answers)


def validate_request(request):
    fields = {'action', 'base_name', 'base_directory', 'adapter_file', 'model_id', 'parent_release_id',
              'evaluation_file', 'output_directory', 'device', 'cpu_threads', 'gpu_fraction', 'deadline', 'gate_id'}
    if not isinstance(request, dict) or set(request) != fields or request['action'] != 'public-math-baseline-v1':
        raise ValueError('Unknown public math evaluation request')
    if request['base_name'] != 'qwen3-1.7b' or request['device'] not in {'cpu', 'cuda'} or request['gate_id'] != gate_id():
        raise ValueError('Math baseline requires the declared expert and immutable evaluation gate')
    if any(not isinstance(request[k], str) or not re.fullmatch('[0-9a-f]{64}', request[k]) for k in ['model_id', 'parent_release_id']):
        raise ValueError('Invalid public math model or parent identity')
    if (type(request['cpu_threads']) is not int or not 1 <= request['cpu_threads'] <= 16
            or type(request['gpu_fraction']) not in (int, float) or not 0 < request['gpu_fraction'] <= 1
            or type(request['deadline']) not in (int, float) or not math.isfinite(request['deadline'])
            or not time.time() < request['deadline'] <= time.time()+3600):
        raise ValueError('Invalid bounded math evaluation resources')
    paths = ['base_directory', 'adapter_file', 'evaluation_file', 'output_directory']
    if any(not isinstance(request[k], str) or not Path(request[k]).is_absolute() for k in paths):
        raise ValueError('Math evaluation requires locally approved absolute paths')
    output = Path(request['output_directory'])
    if output.exists() or output.is_symlink() or output.resolve().is_relative_to(Path(request['base_directory']).resolve()):
        raise ValueError('Math evaluation cannot overwrite an existing directory or public model')
    return load_snapshot(request['evaluation_file'], 'evaluation')


def evaluate_math(model, tokenizer, evaluation, device, deadline, *, progress=lambda *_: None):
    if time.time() >= deadline:raise TimeoutError('Public math evaluation deadline exceeded')
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    policy = gate_definition()
    class Deadline(StoppingCriteria):
        def __call__(self, *_a, **_kw):return time.time() >= deadline
    total = 0.;tokens = 0;answers = [];lengths = []
    model.eval()
    with torch.inference_mode():
        for index, row in enumerate(evaluation['rows']):
            if time.time() >= deadline:raise TimeoutError('Public math evaluation deadline exceeded')
            text = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt(row)}], tokenize=False,
                                                add_generation_prompt=True, enable_thinking=False)
            prefix = tokenizer.encode(text, add_special_tokens=False)
            expected = tokenizer.encode(row['answer'], add_special_tokens=False)+[tokenizer.eos_token_id]
            if len(prefix)+len(expected) > policy['maximum_sequence_tokens'] or len(expected) > policy['maximum_answer_tokens']:
                raise ValueError('Public math example exceeds the predeclared token bound')
            inputs = torch.tensor([prefix+expected], device=device)
            labels = torch.tensor([[-100]*len(prefix)+expected], device=device)
            loss = float(model(input_ids=inputs, labels=labels, use_cache=False).loss)
            if not math.isfinite(loss) or loss < 0:raise ValueError('Invalid public math held-out loss')
            total += loss*len(expected);tokens += len(expected)
            inputs = torch.tensor([prefix], device=device)
            generated = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), do_sample=False,
                                       max_new_tokens=policy['maximum_new_tokens'], use_cache=True,
                                       stopping_criteria=StoppingCriteriaList([Deadline()]))
            if time.time() >= deadline:raise TimeoutError('Public math generation deadline exceeded')
            answer = tokenizer.decode(generated[0, len(prefix):], skip_special_tokens=True).strip()
            if len(answer.encode()) > 8192:raise ValueError('Public math generation exceeds its result bound')
            answers.append({'case': row['case'], 'answer': answer});lengths.append(int(generated.shape[1]-len(prefix)))
            progress(index+1, len(evaluation['rows']))
    correct = score_answers(answers, evaluation)
    return {'nll': total/tokens, 'tokens': tokens, 'answers': answers, 'correct': sum(correct.values()),
            'total': len(correct), 'output_tokens': lengths,
            'at_output_limit': sum(n == policy['maximum_new_tokens'] for n in lengths)}


def run(request):
    evaluation = validate_request(request);started = time.monotonic()
    from .public_bases import BASES, base_hash
    from .weights_io import digest_file
    for filename, identifier in BASES[request['base_name']].items():
        if digest_file(Path(request['base_directory'])/filename)[0] != identifier:
            raise ValueError('Math evaluation upstream dependency differs from its pin')
    if digest_file(request['adapter_file'])[0] != request['model_id']:
        raise ValueError('Math evaluation adapter differs from the declared accepted model')
    output = Path(request['output_directory']);output.mkdir(parents=True, exist_ok=False)
    from src.network.telemetry import atomic_json
    def progress(done, total):
        atomic_json(output/'progress.json', {'phase': 'public-math', 'completed': done, 'total': total, 'timestamp': time.time()})
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .language_adapter import install_adapter, load_adapter
    from .language_candidate_job import evaluate
    from .public_conversation import public_corpus
    torch.set_num_threads(request['cpu_threads']);torch.manual_seed(20261014)
    device = request['device']
    if device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(request['gpu_fraction']);torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(request['base_directory'], local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(request['base_directory'], local_files_only=True, trust_remote_code=False,
                                               dtype=torch.bfloat16 if device == 'cuda' else torch.float32, device_map={'': device}).eval()
    install_adapter(model);load_adapter(model, request['adapter_file'], base_hash(request['base_name']))
    atomic_json(output/'progress.json', {'phase': 'historical-evaluation', 'completed': 0, 'timestamp': time.time()})
    legacy = evaluate(model, tokenizer, public_corpus(), device, request['deadline'])
    result = evaluate_math(model, tokenizer, evaluation, device, request['deadline'], progress=progress)
    result.update(success=True, version=1, action=request['action'], curriculum_id=curriculum_id(), gate_id=gate_id(),
                  evaluation_artifact_id=ARTIFACTS['evaluation'][0], base_name=request['base_name'], model_id=request['model_id'],
                  parent_release_id=request['parent_release_id'], legacy=legacy, training_performed=False, private_inputs=False,
                  model_published=False, pretraining_contamination_excluded=False, seconds=time.monotonic()-started,
                  peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else 0)
    raw = canonical(result)
    if len(raw) > 2*1024**2:raise ValueError('Public math evaluation report exceeds its bound')
    (output/'baseline.json').write_bytes(raw)
    return {'success': True, 'evaluation_id': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
            'model_id': request['model_id'], 'gate_id': gate_id(), 'correct': result['correct'], 'total': result['total']}


def main():
    raw = sys.stdin.buffer.read(16*1024+1)
    if len(raw) > 16*1024:raise ValueError('Public math request exceeds its bound')
    print(json.dumps(run(json.loads(raw)), allow_nan=False), flush=True)


if __name__ == '__main__':main()
