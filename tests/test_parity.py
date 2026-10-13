"""The rewritten model must reproduce the original 25.7M model exactly."""
import pytest
import torch

from glm_ladder import GLMLadderModel, ModelConfig
from tests.reference import original_model


@pytest.mark.parametrize("length", [1, 37, 64, 150, 192])
def test_logits_match_original_model(length: int) -> None:
    torch.manual_seed(0)
    reference = original_model.GLM53FlashFromScratch(original_model.ModelConfig()).eval()
    model = GLMLadderModel(ModelConfig()).eval()

    # strict=True: every parameter name and shape must match, none missing, none extra.
    model.load_state_dict(reference.state_dict(), strict=True)

    ids = torch.randint(0, 260, (2, length), generator=torch.Generator().manual_seed(length))
    with torch.no_grad():
        expected, _ = reference(ids)        # the original returns (logits, usage)
        actual = model(ids).logits
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_original_config_json_loads() -> None:
    # Exactly what the original repository writes into each checkpoint's config.json.
    legacy = {
        "dim": 192, "expert_hidden": 384, "experts": 8, "heads": 6, "layers": 12,
        "max_sequence_length": 192, "sparse_stride": 32, "sparse_window": 32,
        "streams": 4, "top_k": 2, "vocab_size": 260,
    }
    assert ModelConfig.from_dict(legacy) == ModelConfig()