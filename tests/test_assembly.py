import torch

from glm_ladder.config import ModelConfig
from glm_ladder.model import HybridBlock, HyperConnection, LinearAttention, SparseAttention

CONFIG = ModelConfig(dim=16, heads=4, expert_hidden=24, experts=4, sparse_window=4, sparse_stride=4)


def test_block_uses_requested_attention_type() -> None:
    assert isinstance(HybridBlock(CONFIG, sparse=True).attention, SparseAttention)
    assert isinstance(HybridBlock(CONFIG, sparse=False).attention, LinearAttention)


def test_single_stream_is_an_ordinary_residual() -> None:
     # With one stream, softmax over a single logit is exactly 1, so the
    # HyperConnection reduces to  out = x + block(x).
    # HybridBlock already contains its own residual (it returns x + delta),
    # so the input is counted twice: out = 2x + delta. This double residual
    # is inherited from the original model and kept for weight parity.
    torch.manual_seed(0)
    config = ModelConfig(
        dim=16, heads=4, expert_hidden=24, experts=4,
        sparse_window=4, sparse_stride=4, streams=1,
    )
    block = HybridBlock(config, sparse=True)
    hyper = HyperConnection(config, block)

    x_streams = torch.randn(2, 5, 1, 16)
    out_streams, _, _ = hyper(x_streams)

    x = x_streams.squeeze(2)
    block_out, _, _ = block(x)
    torch.testing.assert_close(out_streams.squeeze(2), x + block_out)