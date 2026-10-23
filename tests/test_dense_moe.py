import pytest
import torch

import glm_ladder.model as model_module
from glm_ladder.config import ModelConfig
from glm_ladder.model import SparseMoE

CONFIG = ModelConfig(dim=16, heads=4, expert_hidden=24, experts=4, top_k=2)


def moe_and_input(tokens: int = 10, seed: int = 0):
    torch.manual_seed(seed)
    return SparseMoE(CONFIG), torch.randn(2, tokens // 2, 16)


@torch.no_grad()
def test_dense_path_gives_the_sparse_result() -> None:
    moe, x = moe_and_input()
    sparse_out, sparse_balance, sparse_usage = moe.train()(x)        # training mode: always sparse
    dense_out, dense_balance, dense_usage = moe.eval()(x)            # eval, 10 tokens: dense
    torch.testing.assert_close(dense_out, sparse_out, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(dense_balance, sparse_balance)
    torch.testing.assert_close(dense_usage, sparse_usage)


@torch.no_grad()
def test_large_batches_stay_sparse(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("dense path used")

    monkeypatch.setattr(model_module, "dense_expert_mix", fail)
    moe, x = moe_and_input(tokens=10)
    moe.dense_max_tokens = 4
    moe.eval()(x)                                                     # 10 > 4 tokens: sparse, no error
    moe.dense_max_tokens = 256
    with pytest.raises(AssertionError, match="dense path used"):
        moe(x)


@torch.no_grad()
def test_dense_path_under_bf16_autocast() -> None:
    moe, x = moe_and_input()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output, _, _ = moe.eval()(x)
    assert torch.isfinite(output).all()
