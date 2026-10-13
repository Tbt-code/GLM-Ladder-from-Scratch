import pytest
import torch

from glm_ladder import PRESETS, GLMLadderModel, ModelConfig, expected_parameter_counts

TINY = ModelConfig(
    vocab_size=64, dim=32, layers=4, heads=4, expert_hidden=48, experts=4,
    sparse_window=8, sparse_stride=8, max_sequence_length=64, linear_chunk_size=8,
)


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_parameter_counts_match_lesson_1_formula(name: str) -> None:
    config = PRESETS[name]
    with torch.device("meta"):
        model = GLMLadderModel(config)
    counts = model.parameter_counts()
    # Other code (benchmark, receipts) reads these exact keys.
    assert set(counts) == {"total", "active_per_token_estimate"}
    assert counts == expected_parameter_counts(config)


def test_output_is_tied_to_embedding() -> None:
    model = GLMLadderModel(TINY)
    assert model.output.weight is model.embedding.weight


def test_model_is_causal() -> None:
    # TODO(you): changing tokens at positions >= 25 must not change the logits at
    # positions < 25, and SHOULD change the logits at positions >= 25.
    # Hint: build ids, copy them, modify the copy from position 25 on,
    # run both through the model in eval mode under torch.no_grad(), compare slices.
    torch.manual_seed(0)
    model = GLMLadderModel(TINY)
    model.eval()

    with torch.no_grad():
        ids1 = torch.randint(0,TINY.vocab_size, (2,40))
        ids2 = ids1.clone()

        ids2[:, 25:] = (ids2[:, 25:] + 1) % TINY.vocab_size

        logits1 = model(ids1).logits
        logits2 = model(ids2).logits

        torch.testing.assert_close(logits1[:, :25, :], logits2[:, :25, :])
        assert not torch.allclose(logits1[:, 25:, :], logits2[:, 25:, :])