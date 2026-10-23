import pytest
import torch
import torch.nn as nn

from glm_ladder import GLMLadderModel, ModelConfig
from glm_ladder.code_eval import sample_many
from glm_ladder.model import (
    LinearCache, chunked_causal_linear_attention, linear_attention_step, sparse_causal_mask, sparse_mask_rows,
)

TINY = ModelConfig(vocab_size=64, dim=32, layers=4, heads=4, expert_hidden=48, experts=4,
                   sparse_window=8, sparse_stride=8, max_sequence_length=64, linear_chunk_size=8)


def model_and_ids(length: int = 40, batch: int = 2, seed: int = 0):
    torch.manual_seed(seed)
    model = GLMLadderModel(TINY).eval()
    ids = torch.randint(0, TINY.vocab_size, (batch, length), generator=torch.Generator().manual_seed(seed + 1))
    return model, ids


# ----------------------------------------------------------------- the two attention caches

def test_linear_attention_step_matches_the_chunked_form() -> None:
    generator = torch.Generator().manual_seed(0)
    q = torch.rand(2, 21, 3, 4, generator=generator) + 0.1          # positive, like ELU + 1
    k = torch.rand(2, 21, 3, 4, generator=generator) + 0.1
    v = torch.randn(2, 21, 3, 5, generator=generator)
    full = chunked_causal_linear_attention(q, k, v, chunk_size=8)
    # The cache holds the sums over the first 20 positions; the step adds position 20.
    cache = LinearCache(state=torch.einsum("bthd,bthe->bhde", k[:, :20], v[:, :20]), normalizer=k[:, :20].sum(1))
    step = linear_attention_step(q[:, 20:], k[:, 20:], v[:, 20:], cache)
    assert step.shape == (2, 1, 3, 5)
    torch.testing.assert_close(step, full[:, 20:], atol=1e-5, rtol=1e-5)
    # ... and the cache now holds the sums over all 21 positions.
    torch.testing.assert_close(cache.normalizer, k.sum(1))


@pytest.mark.parametrize("start, length", [(0, 1), (0, 20), (7, 1), (8, 1), (31, 1), (16, 5)])
def test_mask_rows_match_the_full_mask(start, length) -> None:
    full = sparse_causal_mask(start + length, window=8, stride=8, device=torch.device("cpu"))
    assert torch.equal(sparse_mask_rows(start, length, 8, 8, torch.device("cpu")), full[start:])


# ----------------------------------------------------------------- the whole model

@torch.no_grad()
@pytest.mark.parametrize("prompt", [1, 7, 8, 13])
def test_prefill_then_decode_gives_the_full_forward_logits(prompt) -> None:
    # Prompt lengths below, at, and above a linear-attention chunk and a sparse window (8);
    # decoding then crosses several chunks, windows and anchors up to position 39.
    model, ids = model_and_ids()
    full = model(ids).logits
    cache = model.new_cache()
    pieces = [model(ids[:, :prompt], cache=cache).logits]
    for position in range(prompt, ids.shape[1]):
        pieces.append(model(ids[:, position:position + 1], cache=cache).logits)
    assert cache.length == ids.shape[1]
    torch.testing.assert_close(torch.cat(pieces, dim=1), full, atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_cache_refuses_more_than_the_context() -> None:
    model, ids = model_and_ids(length=TINY.max_sequence_length)
    cache = model.new_cache()
    model(ids, cache=cache)
    with pytest.raises(ValueError):
        model(ids[:, :1], cache=cache)


@torch.no_grad()
def test_linear_cache_must_start_at_position_zero_and_decode_one_token() -> None:
    model, ids = model_and_ids()
    cache = model.new_cache()
    model(ids[:, :5], cache=cache)
    with pytest.raises(ValueError):
        model(ids[:, 5:7], cache=cache)               # two new tokens at once after the prompt


# ----------------------------------------------------------------- generation

class NoCache(nn.Module):
    """The same model without new_cache(), so sample_many takes the uncached path."""

    def __init__(self, model: GLMLadderModel) -> None:
        super().__init__()
        self.model, self.config = model, model.config

    def forward(self, ids):
        return self.model(ids)


@pytest.mark.parametrize("temperature, top_k", [(0.0, None), (1.0, 20)])
def test_cached_generation_matches_uncached(temperature, top_k) -> None:
    model, ids = model_and_ids(length=6)
    settings = dict(n=4, max_new_tokens=20, temperature=temperature, top_k=top_k, seed=3, stop=lambda out: 7 in out)
    cached = sample_many(model, ids[0].tolist(), **settings)
    uncached = sample_many(NoCache(model), ids[0].tolist(), **settings)
    assert cached == uncached
