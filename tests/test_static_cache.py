import pytest
import torch

from glm_ladder import GLMLadderModel, ModelConfig
from glm_ladder.code_eval import sample_many
from glm_ladder.graph_decode import GraphDecoder, generate_batch
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


def growing_reference(q, keys, values, position):
    """What one row's token at `position` sees with the growing (non-static) cache."""
    growing = SparseCache(keys=keys[:, :, :position], values=values[:, :, :position]) if position else SparseCache()
    return cached_attention(q, keys[:, :, position:position + 1], values[:, :, position:position + 1], growing,
                            sparse_mask_rows(position, 1, 8, 8, torch.device("cpu")))


@pytest.mark.parametrize("position", [0, 5, 8, 9, 16, 19])
def test_static_attention_with_one_shared_position(position) -> None:
    generator = torch.Generator().manual_seed(position)
    keys = torch.randn(2, 3, 20, 4, generator=generator)
    values = torch.randn(2, 3, 20, 4, generator=generator)
    q = torch.randn(2, 3, 1, 4, generator=generator)
    expected = growing_reference(q, keys, values, position)
    # Static buffer of 20 slots: earlier positions filled, later ones full of garbage.
    static = StaticSparseCache(keys=torch.randn(2, 3, 20, 4) * 100, values=torch.randn(2, 3, 20, 4) * 100)
    static.keys[:, :, :position] = keys[:, :, :position]
    static.values[:, :, :position] = values[:, :, :position]
    got = static_cached_attention(q, keys[:, :, position:position + 1], values[:, :, position:position + 1],
                                  static, torch.tensor([position]), window=8, stride=8)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(static.keys[:, :, position], keys[:, :, position])     # written into its slot


def test_static_attention_with_a_position_per_row() -> None:
    generator = torch.Generator().manual_seed(7)
    positions = [3, 12, 0]
    keys = torch.randn(3, 2, 20, 4, generator=generator)
    values = torch.randn(3, 2, 20, 4, generator=generator)
    q = torch.randn(3, 2, 1, 4, generator=generator)
    static = StaticSparseCache(keys=torch.randn(3, 2, 20, 4) * 100, values=torch.randn(3, 2, 20, 4) * 100)
    new_k = torch.stack([keys[row, :, p] for row, p in enumerate(positions)])[:, :, None]
    new_v = torch.stack([values[row, :, p] for row, p in enumerate(positions)])[:, :, None]
    for row, p in enumerate(positions):
        static.keys[row, :, :p] = keys[row, :, :p]
        static.values[row, :, :p] = values[row, :, :p]
    got = static_cached_attention(q, new_k, new_v, static, torch.tensor(positions), window=8, stride=8)
    for row, p in enumerate(positions):
        expected = growing_reference(q[row:row + 1], keys[row:row + 1], values[row:row + 1], p)
        torch.testing.assert_close(got[row:row + 1], expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(static.keys[row, :, p], keys[row, :, p])


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
    assert static.position.tolist() == [prompt, prompt]
    for position in range(prompt, ids.shape[1]):
        pieces.append(model(ids[:, position:position + 1], cache=static).logits)
    assert static.position.tolist() == [ids.shape[1]] * 2
    torch.testing.assert_close(torch.cat(pieces, dim=1), full, atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_prompts_of_different_lengths_share_one_batch() -> None:
    # Three prompts (lengths below, at and past a chunk / window of 8), each continued with
    # 12 fixed tokens. Every row must get exactly the logits of its own sequence run alone.
    model, ids = model_and_ids(length=30, batch=3)
    lengths = [3, 8, 14]
    decoder = GraphDecoder(model, batch=3, max_length=40, graph=False)
    first = decoder.prefill([ids[row, :n].tolist() for row, n in enumerate(lengths)])
    steps = [first.clone()]
    for offset in range(12):
        tokens = torch.stack([ids[row, n + offset] for row, n in enumerate(lengths)])
        steps.append(decoder.step(tokens).clone())
    for row, n in enumerate(lengths):
        alone = model(ids[row:row + 1, :n + 12]).logits[0, n - 1:]      # predictions after token n-1 ... n+10
        got = torch.stack([s[row] for s in steps])[:12 + 1]
        torch.testing.assert_close(got, alone[:13], atol=1e-4, rtol=1e-4)


def test_batched_generation_matches_one_prompt_at_a_time() -> None:
    model, ids = model_and_ids(length=10, batch=2)
    prompts = [ids[0, :4].tolist(), ids[1, :9].tolist()]
    settings = dict(max_new_tokens=15, temperature=0.0, top_k=None, seed=0, stop=lambda out: 5 in out)
    decoder = GraphDecoder(model, batch=4, max_length=32, graph=False)
    batched = generate_batch(decoder, [prompts[0], prompts[0], prompts[1], prompts[1]], **settings)
    one_by_one = [out for prompt in prompts for out in sample_many(model, prompt, n=2, **settings)]
    assert batched == one_by_one


# ----------------------------------------------------------------- CUDA graphs (GPU only)

@pytest.mark.skipif(not CUDA, reason="CUDA graphs need a GPU")
@torch.no_grad()
def test_graph_replay_gives_the_full_forward_logits() -> None:
    model, ids = model_and_ids(device="cuda")
    full = model(ids).logits
    decoder = GraphDecoder(model, batch=2, max_length=48)
    pieces = [decoder.prefill(ids[:, :8].tolist()).clone()]
    for position in range(8, ids.shape[1] - 1):
        pieces.append(decoder.step(ids[:, position]).clone())
    torch.testing.assert_close(torch.stack(pieces, dim=1), full[:, 7:-1], atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not CUDA, reason="CUDA graphs need a GPU")
def test_graph_batch_of_different_prompts_matches_eager() -> None:
    model, ids = model_and_ids(device="cuda", length=12, batch=3)
    prompts = [ids[0, :3].tolist(), ids[1, :7].tolist(), ids[2, :12].tolist()]
    settings = dict(max_new_tokens=20, temperature=0.0, top_k=None, seed=0, stop=lambda out: False)
    graph = generate_batch(GraphDecoder(model, batch=3, max_length=40), prompts, **settings)
    eager = generate_batch(GraphDecoder(model, batch=3, max_length=40, graph=False), prompts, **settings)
    assert graph == eager
