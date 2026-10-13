import torch

from glm_ladder.config import ModelConfig
from glm_ladder.model import SparseMoE

CONFIG = ModelConfig(dim=16, heads=4, expert_hidden=24, experts=4, top_k=2)


def test_moe_matches_per_token_reference() -> None:
    torch.manual_seed(0)
    moe = SparseMoE(CONFIG)
    x = torch.randn(2, 5, 16)
    output, _, _ = moe(x)

    flat = x.reshape(-1, 16)
    for t in range(flat.shape[0]):
        logits = moe.router(flat[t])
        values, indices = logits.topk(2)
        weights = torch.softmax(values, dim=-1)
        expected = moe.shared(flat[t])
        for w, e in zip(weights, indices):
            expected = expected + w * moe.experts[e](flat[t])
        torch.testing.assert_close(output.reshape(-1, 16)[t], expected, rtol=1e-5, atol=1e-5)


def test_balance_loss_reaches_router() -> None:
    torch.manual_seed(0)
    moe = SparseMoE(CONFIG)
    x = torch.randn(2, 5, 16, requires_grad=True)
    
    _, balance_loss, _ = moe(x)
    
    balance_loss.backward()
    
    assert moe.router.weight.grad is not None, "Router weights didn't take gradients."
    assert not torch.all(moe.router.weight.grad == 0), "Router gradients cannot be 0."

def test_moe_runs_under_bf16_autocast() -> None:
    torch.manual_seed(0)
    moe = SparseMoE(CONFIG)
    x = torch.randn(2, 5, 16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output, balance_loss, _ = moe(x)
    assert torch.isfinite(output).all()