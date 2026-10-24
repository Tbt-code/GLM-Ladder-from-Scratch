import pytest
import torch

from glm_ladder import GLMLadderModel, ModelConfig
from glm_ladder.code_eval import sample_many
from glm_ladder.model import (
    LinearCache, SparseCache, StaticSparseCache, cached_attention, linear_attention_step, sparse_mask_rows,
    static_cached_attention,
)

TINY = ModelConfig(vocab_size=64, dim=32, layers=4, heads=4, expert_hidden=48, experts=4,
                   sparse_window=8, sparse_stride=8, max_sequence_length=64, linear_chunk_size=8)
CUDA = torch.cuda.is_available()


def model_and_ids(length: int = 40, batch: int = 2, seed: int = 0, device: str = "cpu"):
    torch.manual_seed(seed)
    model = GLMLadderModel(TINY).to(device).eval()
    ids = torch.randint(0, TINY.vocab_size, (batch, length), generator=torch.Generator().manual_seed(seed + 1))
    return model, ids.to(device)


# ----------------------------------------------------------------- the pieces

def test_linear_step_updates_the_cache_in_place() -> None:
    generator = torch.Generator().manual_seed(0)
    q, k = (torch.rand(2, 1, 3, 4, generator=generator) + 0.1 for _ in range(2))
    v = torch.randn(2, 1, 3, 4, generator=generator)
    cache = LinearCache(state=torch.zeros(2, 3, 4, 4), normalizer=torch.zeros(2, 3, 4))
    state, normalizer = cache.state, cache.normalizer
    linear_attention_step(q, k, v, cache)
    # Same tensors, now holding the update: a CUDA graph would see it.
    assert cache.state is state and cache.normalizer is normalizer
    torch.testing.assert_close(state, torch.einsum("bhd,bhe->bhde", k[:, 0], v[:, 0]))


@pytest.mark.parametrize("position", [0, 5, 8, 9, 16, 19])
def test_static_attention_matches_the_growing_cache(position) -> None:
    generator = torch.Generator().manual_seed(position)
    keys = torch.randn(2, 3, position + 1, 4, generator=generator)
    values = torch.randn(2, 3, position + 1, 4, generator=generator)
    q = torch.randn(2, 3, 1, 4, generator=generator)
    growing = SparseCache(keys=keys[:, :, :position], values=values[:, :, :position]) if position else SparseCache()
    expected = cached_attention(q, keys[:, :, position:], values[:, :, position:], growing,
                                sparse_mask_rows(position, 1, 8, 8, torch.device("cpu")))
    # Static buffer of 20 slots: earlier positions filled, later ones full of garbage.
    static = StaticSparseCache(keys=torch.randn(2, 3, 20, 4) * 100, values=torch.randn(2, 3, 20, 4) * 100)
    static.keys[:, :, :position] = keys[:, :, :position]
    static.values[:, :, :position] = values[:, :, :position]
    got = static_cached_attention(q, keys[:, :, position:], values[:, :, position:], static,
                                  torch.tensor([position]), window=8, stride=8)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(static.keys[:, :, position], keys[:, :, position])     # written into its slot


# ----------------------------------------------------------------- the whole model, eagerly

@torch.no_grad()
@pytest.mark.parametrize("prompt", [1, 8, 13])
def test_static_cache_decoding_gives_the_full_forward_logits(prompt) -> None:
    model, ids = model_and_ids()
    full = model(ids).logits
    cache = model.new_cache()
    pieces = [model(ids[:, :prompt], cache=cache).logits]
    static = model.new_static_cache(batch=2, max_length=48)
    model.fill_static_cache(static, cache)
    assert static.position.item() == prompt
    for position in range(prompt, ids.shape[1]):
        pieces.append(model(ids[:, position:position + 1], cache=static).logits)
    assert static.position.item() == ids.shape[1]
    torch.testing.assert_close(torch.cat(pieces, dim=1), full, atol=1e-4, rtol=1e-4)


# ----------------------------------------------------------------- CUDA graphs (GPU only)

@pytest.mark.skipif(not CUDA, reason="CUDA graphs need a GPU")
@torch.no_grad()
def test_graph_replay_gives_the_full_forward_logits() -> None:
    from glm_ladder.graph_decode import GraphDecoder

    model, ids = model_and_ids(device="cuda")
    full = model(ids).logits
    decoder = GraphDecoder(model, batch=2, max_length=48)
    pieces = [decoder.prefill(ids[:, :8]).clone()]
    for position in range(8, ids.shape[1] - 1):
        pieces.append(decoder.step(ids[:, position]).clone())
    torch.testing.assert_close(torch.stack(pieces, dim=1), full[:, 7:-1], atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not CUDA, reason="CUDA graphs need a GPU")
def test_graph_decoder_reused_for_a_second_prompt_and_in_sample_many() -> None:
    from glm_ladder.graph_decode import GraphDecoder

    model, ids = model_and_ids(device="cuda", length=6)
    decoder = GraphDecoder(model, batch=4, max_length=40)
    settings = dict(n=4, max_new_tokens=20, temperature=0.0, top_k=None, seed=0, stop=lambda out: False)
    for prompt in (ids[0].tolist(), ids[1].tolist()):        # the same recorded graph, two prompts
        assert sample_many(model, prompt, decoder=decoder, **settings) == sample_many(model, prompt, **settings)
