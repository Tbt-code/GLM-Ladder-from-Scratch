import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from glm_ladder import GLMLadderModel, ModelConfig
from glm_ladder.evaluation import bits_per_byte
from glm_ladder.generate import generate, sample_next

TINY = ModelConfig(vocab_size=64, dim=32, layers=4, heads=4, expert_hidden=48, experts=4,
                   sparse_window=8, sparse_stride=8, max_sequence_length=64, linear_chunk_size=8)


def rng(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


class Counter(nn.Module):
    """Fake model that always predicts (last token + 1) % vocab, with a tiny context window."""

    def __init__(self, vocab: int = 16, window: int = 4) -> None:
        super().__init__()
        self.config = SimpleNamespace(max_sequence_length=window)
        self.vocab = vocab
        self.dummy = nn.Parameter(torch.zeros(1))          # so next(model.parameters()) works
        self.seen_lengths: list[int] = []

    def forward(self, ids: torch.Tensor):
        self.seen_lengths.append(ids.shape[1])
        logits = torch.full((*ids.shape, self.vocab), -1e9)
        logits.scatter_(-1, ((ids + 1) % self.vocab)[..., None], 0.0)
        return SimpleNamespace(logits=logits)


# ----------------------------------------------------------------- sample_next

def test_temperature_zero_is_argmax() -> None:
    logits = torch.tensor([0.1, 2.0, -1.0, 1.9])
    assert sample_next(logits, temperature=0.0, top_k=None, generator=rng()) == 1


def test_top_k_one_is_argmax_at_any_temperature() -> None:
    logits = torch.tensor([0.1, 2.0, -1.0, 1.9])
    for seed in range(20):
        assert sample_next(logits, temperature=5.0, top_k=1, generator=rng(seed)) == 1


def test_top_k_never_samples_outside_the_top_k() -> None:
    logits = torch.tensor([3.0, 2.9, 2.8, 0.0, 0.0, 0.0, 0.0, 0.0])
    generator = rng()
    draws = {sample_next(logits, temperature=10.0, top_k=3, generator=generator) for _ in range(300)}
    assert draws == {0, 1, 2}            # high temperature flattens, but the cut-off still holds


def test_sampling_follows_the_softmax() -> None:
    logits = torch.log(torch.tensor([0.7, 0.2, 0.1]))
    generator = rng()
    draws = [sample_next(logits, temperature=1.0, top_k=None, generator=generator) for _ in range(4000)]
    assert draws.count(0) / len(draws) == pytest.approx(0.7, abs=0.03)
    assert draws.count(2) / len(draws) == pytest.approx(0.1, abs=0.02)


# ----------------------------------------------------------------- generate

def test_generate_greedy_and_stop_token() -> None:
    model = Counter()
    assert generate(model, [3], max_new_tokens=5, temperature=0.0, stop_id=None) == [4, 5, 6, 7, 8]
    assert generate(model, [3], max_new_tokens=5, temperature=0.0, stop_id=6) == [4, 5, 6]


def test_generate_crops_the_context_to_the_window() -> None:
    model = Counter(vocab=32, window=4)
    out = generate(model, list(range(10)), max_new_tokens=3, temperature=0.0, stop_id=None)
    assert out == [10, 11, 12]
    assert max(model.seen_lengths) == 4


def test_generate_is_reproducible_and_restores_training_mode() -> None:
    torch.manual_seed(0)
    model = GLMLadderModel(TINY)
    a = generate(model, [5, 6, 7], max_new_tokens=8, temperature=1.0, top_k=10, seed=3, stop_id=None)
    b = generate(model, [5, 6, 7], max_new_tokens=8, temperature=1.0, top_k=10, seed=3, stop_id=None)
    assert a == b and len(a) == 8
    assert all(0 <= token < TINY.vocab_size for token in a)
    assert model.training


# ----------------------------------------------------------------- bits per byte

def test_bits_per_byte() -> None:
    assert bits_per_byte(4 * math.log(2), 4.0) == pytest.approx(1.0)
    assert bits_per_byte(3.5412, 4.103) == pytest.approx(1.245, abs=1e-3)
