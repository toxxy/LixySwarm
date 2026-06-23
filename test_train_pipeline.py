"""Safety and reproducibility checks for clean language-foundation runs."""

import numpy as np
import torch

from train import (
    BilingualTokenDataset,
    TokenDataset,
    _portable_model_state,
    _save_checkpoint_atomic,
)
from train_swarm import BilingualDataset


def test_random_token_dataset_is_deterministic_without_giant_shuffle(tmp_path):
    values = np.arange(4096, dtype=np.uint16)
    path = tmp_path / "tokens.bin"
    values.tofile(path)
    dataset = TokenDataset(
        path,
        32,
        random_samples=True,
        seed=17,
        sample_offset=123,
    )
    x, y = dataset[0]
    rng = np.random.default_rng(17 ^ (123 * 2654435761 & 0xFFFFFFFF))
    start = int(rng.integers(0, len(values) - 33))
    assert torch.equal(x, torch.from_numpy(values[start:start + 32].astype(np.int64)))
    assert torch.equal(y, torch.from_numpy(values[start + 1:start + 33].astype(np.int64)))
    repeated_x, repeated_y = dataset[0]
    assert torch.equal(x, repeated_x)
    assert torch.equal(y, repeated_y)


def test_portable_model_state_omits_compile_wrapper_prefix():
    class CompiledLike:
        def __init__(self):
            self._orig_mod = torch.nn.Linear(3, 2)

    state = _portable_model_state(CompiledLike())
    assert set(state) == {"weight", "bias"}
    assert not any(key.startswith("_orig_mod.") for key in state)


def test_bilingual_swarm_dataset_never_requires_personal_text(tmp_path):
    english = np.arange(2048, dtype=np.uint16)
    spanish = np.arange(4096, 6144, dtype=np.uint16)
    english_path = tmp_path / "english.bin"
    spanish_path = tmp_path / "spanish.bin"
    english.tofile(english_path)
    spanish.tofile(spanish_path)
    dataset = BilingualDataset(
        english_path,
        spanish_path,
        block_size=32,
        fw_ratio=0.5,
        seed=9,
    )
    samples = [dataset[index][0] for index in range(32)]
    assert any(int(sample[0]) < 4096 for sample in samples)
    assert any(int(sample[0]) >= 4096 for sample in samples)


def test_bilingual_base_dataset_resume_offset_is_deterministic(tmp_path):
    english = np.arange(2048, dtype=np.uint16)
    spanish = np.arange(4096, 6144, dtype=np.uint16)
    english_path = tmp_path / "base-english.bin"
    spanish_path = tmp_path / "base-spanish.bin"
    english.tofile(english_path)
    spanish.tofile(spanish_path)
    first = BilingualTokenDataset(
        english_path,
        spanish_path,
        32,
        english_ratio=0.7,
        seed=21,
        sample_offset=500,
    )
    resumed = BilingualTokenDataset(
        english_path,
        spanish_path,
        32,
        english_ratio=0.7,
        seed=21,
        sample_offset=501,
    )
    assert torch.equal(first[1][0], resumed[0][0])
    assert torch.equal(first[1][1], resumed[0][1])


def test_training_checkpoint_replace_is_atomic(tmp_path):
    path = tmp_path / "foundation.pt"
    _save_checkpoint_atomic({"step": 7, "model": {"x": torch.ones(1)}}, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert loaded["step"] == 7
    assert torch.equal(loaded["model"]["x"], torch.ones(1))
    assert not path.with_suffix(".pt.tmp").exists()
