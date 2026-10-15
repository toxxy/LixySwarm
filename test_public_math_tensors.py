"""Actual CPU tensor supervision masks; no downloaded model or quality claims."""
import time
from types import SimpleNamespace

import pytest
torch = pytest.importorskip('torch')

from src.learning.public_math_job import evaluate_math
from test_public_math import snapshots


def test_public_math_masks_prompt_and_uses_untruncated_fixed_generation(snapshots):
    evaluation = snapshots['evaluation'][1];calls = []
    class Tokenizer:
        eos_token_id = 9
        def apply_chat_template(self, messages, **options):
            assert options == {'tokenize': False, 'add_generation_prompt': True, 'enable_thinking': False}
            return 'PREFIX'+messages[0]['content']
        def encode(self, text, **options):
            assert options == {'add_special_tokens': False}
            return [1, 2, 3, 4] if text.startswith('PREFIX') else [5, 6]
        def decode(self, _tokens, **_options):return '#### 1'
    class Model:
        def eval(self):return self
        def __call__(self, *, input_ids, labels, use_cache):
            assert input_ids.tolist() == [[1, 2, 3, 4, 5, 6, 9]]
            assert labels.tolist() == [[-100, -100, -100, -100, 5, 6, 9]]
            assert not use_cache and not torch.is_grad_enabled()
            return SimpleNamespace(loss=torch.tensor(.25))
        def generate(self, **options):
            assert options['input_ids'].tolist() == [[1, 2, 3, 4]]
            assert options['max_new_tokens'] == 384 and options['do_sample'] is False and options['use_cache']
            assert options['attention_mask'].tolist() == [[1, 1, 1, 1]]
            calls.append('generate')
            return torch.tensor([[1, 2, 3, 4, 7, 8]])
    report = evaluate_math(Model(), Tokenizer(), evaluation, 'cpu', time.time()+20)
    assert report['nll'] == .25 and report['tokens'] == 24 and report['correct'] == 1
    assert len(calls) == 8 and not report['at_output_limit']


def test_public_math_evaluation_expired_before_compute(snapshots):
    with pytest.raises(TimeoutError):evaluate_math(None, None, snapshots['evaluation'][1], 'cpu', time.time()-1)
