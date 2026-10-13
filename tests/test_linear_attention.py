import torch
import torch.nn.functional as F
import pytest
from glm_ladder.model import chunked_causal_linear_attention
from glm_ladder.config import ModelConfig
from glm_ladder.model import LinearAttention

def quadratic_linear_attention(q, k, v):
    """q, k: [B, T, H, d] (already positive); v: [B, T, H, e]."""
    q_bh = q.transpose(1,2)
    k_bh = k.transpose(1,2)
    scores = torch.einsum("bhid,bhjd->bhij",q_bh,k_bh)

    scores = torch.tril(scores)
    v_bh = v.transpose(1,2)
    numerator = torch.einsum("bhij,bhje->bhie",scores,v_bh)
    numerator = numerator.transpose(1,2)

    denominator = scores.sum(dim=-1)
    denominator = denominator.transpose(1,2)

    return numerator / denominator.clamp_min(1e-6)[...,None]
    


def recurrent_linear_attention(q, k, v):
    """The original repository's formulation: one d x e state per position."""
    kv_outer = torch.einsum('bthd,bthe->bthde', k, v)
    S = kv_outer.cumsum(dim=1)  # [B, T, H, d, e]
    z = k.cumsum(dim=1)  # [B, T, H, d]

    numerator = torch.einsum('bthd,bthde->bthe', q, S)
    denominator = torch.einsum('bthd,bthd->bth', q, z)

    return numerator / denominator.clamp_min(1e-6)[..., None]


def test_quadratic_and_recurrent_forms_agree() -> None:
    torch.manual_seed(0)
    q = F.elu(torch.randn(2, 17, 3, 8, dtype=torch.float64)) + 1
    k = F.elu(torch.randn(2, 17, 3, 8, dtype=torch.float64)) + 1
    v = torch.randn(2, 17, 3, 8, dtype=torch.float64)
    torch.testing.assert_close(
        quadratic_linear_attention(q, k, v), recurrent_linear_attention(q, k, v)
    )



@pytest.mark.parametrize("length,chunk", [(1, 4), (7, 4), (16, 4), (17, 4), (33, 8), (20, 64)])
def test_chunked_matches_recurrent(length: int, chunk: int) -> None:
    torch.manual_seed(length * 31 + chunk)
    q = F.elu(torch.randn(2, length, 3, 8, dtype=torch.float64)) + 1
    k = F.elu(torch.randn(2, length, 3, 8, dtype=torch.float64)) + 1
    v = torch.randn(2, length, 3, 8, dtype=torch.float64)
    torch.testing.assert_close(
        chunked_causal_linear_attention(q, k, v, chunk), recurrent_linear_attention(q, k, v)
    )

def test_linear_attention_output_does_not_depend_on_chunk_size() -> None:
    torch.manual_seed(0)
    small = LinearAttention(ModelConfig(dim=32, heads=4, linear_chunk_size=4))
    large = LinearAttention(ModelConfig(dim=32, heads=4, linear_chunk_size=64))
    large.load_state_dict(small.state_dict())  # identical weights
    x = torch.randn(2, 30, 32)
    torch.testing.assert_close(small(x), large(x))