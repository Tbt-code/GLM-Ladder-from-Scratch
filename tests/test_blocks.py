import torch
import torch.nn as nn

from glm_ladder.model import RMSNorm
from glm_ladder.model import apply_rope
from glm_ladder.model import Expert
from glm_ladder.model import sparse_causal_mask
from glm_ladder.config import ModelConfig
from glm_ladder.model import SparseAttention
import torch.nn.functional as F


def test_rmsnorm_output_has_unit_rms() -> None:
    norm = RMSNorm(16)
    x = torch.randn(4, 7, 16) * 50  # deliberately large scale
    y = norm(x)
    rms = y.pow(2).mean(-1).sqrt()
    torch.testing.assert_close(rms, torch.ones(4, 7), rtol=1e-4, atol=1e-4)


def test_rmsnorm_is_scale_invariant() -> None:
    # TODO(you): show that norm(c * x) == norm(x) for a positive constant c.
    norm = RMSNorm(10)
    x = torch.randn(4, 7, 10) * 20
    y = norm(x)
    torch.testing.assert_close(norm(x * 10), y)

def test_rmsnorm_preserves_dtype() -> None:
    norm = RMSNorm(16)
    x = torch.randn(2, 3, 16).to(torch.bfloat16)
    assert norm(x).dtype == torch.bfloat16

def test_rope_preserves_vector_length() -> None:
    # A rotation never changes the length of a vector.
    q = torch.randn(2, 9, 3, 8)
    k = torch.randn(2, 9, 3, 8)
    q_rot, k_rot = apply_rope(q, k)
    torch.testing.assert_close(q_rot.norm(dim=-1), q.norm(dim=-1))


def test_rope_score_depends_only_on_relative_position() -> None:
    # TODO(you): this is RoPE's key property.
    # Put the SAME query vector at every position and the SAME key vector at
    # every position, apply RoPE, and compute scores[m, n] = q_rot[m] . k_rot[n].
    # Then check that scores[m, n] == scores[m + 1, n + 1].
    ...
    time_len = 16
    head_dim = 16

    q_single = torch.randn(1,1,1,head_dim)
    k_single = torch.randn(1,1,1,head_dim)
    q = q_single.expand(-1, time_len, -1, -1)
    k = k_single.expand(-1, time_len, -1, -1)
    
    q_rot, k_rot = apply_rope(q, k)
    
    q_rot_2d = q_rot.squeeze(0).squeeze(1)
    k_rot_2d = k_rot.squeeze(0).squeeze(1)
    
    scores = q_rot_2d @ k_rot_2d.T
    torch.testing.assert_close(scores[:-1, :-1], scores[1:, 1:])

import math

def test_rope_uses_base_10000() -> None:
    head_dim = 8
    q = torch.zeros(1, 2, 1, head_dim)  # positions 0 and 1
    q[0, 1, 0, 2] = 1.0                 # second pair = (1, 0) at position 1
    q_rot, _ = apply_rope(q, q)
    theta_1 = 10000 ** (-2 / head_dim)
    expected = torch.tensor([math.cos(theta_1), math.sin(theta_1)])
    torch.testing.assert_close(q_rot[0, 1, 0, 2:4], expected)

def test_expert_shapes_and_parameter_count() -> None:
    expert = Expert(dim=192, hidden=384)
    x = torch.randn(5, 192)
    assert expert(x).shape == (5, 192)
    assert sum(p.numel() for p in expert.parameters()) == 3 * 192 * 384


def test_sparse_mask_matches_original_pattern() -> None:
    length, window, stride = 150, 5, 3
    mask = sparse_causal_mask(length, window, stride, torch.device("cpu"))
    for i in range(length):
        # The original repository built these index sets with Python loops.
        anchors = set(range(0, i + 1, stride))
        local = set(range(max(0, i - window + 1), i + 1))
        assert set(torch.nonzero(mask[i]).flatten().tolist()) == anchors | local


def test_sparse_mask_every_row_can_see_itself() -> None:
    length,window,stride = 64,16,16
    device = torch.device("cpu")
    mask = sparse_causal_mask(length,window,stride,device)

    diagonal = torch.diagonal(mask)
    assert torch.all(diagonal), "Every row must see itself."


def manual_attention(q, k, v, mask):
    # q, k, v: [B, H, T, hd]; mask: [T, T] boolean
    # TODO(you):
    #   scores = q @ k^T / sqrt(hd)           -> [B, H, T, T]
    #   masked positions -> -inf              (hint: masked_fill with ~mask)
    #   softmax over the LAST dim
    #   weights @ v                           -> [B, H, T, hd]
    head_dim = q.shape[-1]

    scores = q @ k.transpose(-1,-2) / math.sqrt(head_dim)
    scores = scores.masked_fill(~mask,float("-inf"))

    weights = F.softmax(scores,dim=-1)
    output = weights @ v
    return output

    


def test_sdpa_with_mask_matches_manual_attention() -> None:
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, 20, 8) for _ in range(3))
    mask = sparse_causal_mask(20, window=4, stride=5, device=torch.device("cpu"))
    expected = manual_attention(q, k, v, mask)
    actual = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    torch.testing.assert_close(actual, expected)


def test_sparse_attention_output_shape() -> None:
    config = ModelConfig(dim=32, heads=4, sparse_window=4, sparse_stride=4, max_sequence_length=64)
    attention = SparseAttention(config)
    x = torch.randn(2, 10, 32)
    assert attention(x).shape == (2, 10, 32)

