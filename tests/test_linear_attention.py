import torch
import torch.nn.functional as F


def quadratic_linear_attention(q, k, v):
    """q, k: [B, T, H, d] (already positive); v: [B, T, H, e]."""
    # TODO(you):
    #   scores = phi(q_i) . phi(k_j) for all i, j      -> [B, H, T, T]   (einsum)
    #   zero out j > i                                  (torch.tril)
    #   numerator   = scores @ v                        -> [B, T, H, e]   (einsum)
    #   denominator = scores summed over j              -> [B, T, H]
    #   return numerator / denominator.clamp_min(1e-6)[..., None]
    ...


def recurrent_linear_attention(q, k, v):
    """The original repository's formulation: one d x e state per position."""
    # TODO(you):
    #   kv     = outer product k_t v_t^T, cumulative over t   -> [B, T, H, d, e]
    #            einsum "bthd,bthe->bthde", then .cumsum(dim=1)
    #   k_sum  = k cumulative over t                          -> [B, T, H, d]
    #   numerator   = q_t^T S_t   (einsum)
    #   denominator = q_t . z_t   (einsum)
    ...


def test_quadratic_and_recurrent_forms_agree() -> None:
    torch.manual_seed(0)
    q = F.elu(torch.randn(2, 17, 3, 8, dtype=torch.float64)) + 1
    k = F.elu(torch.randn(2, 17, 3, 8, dtype=torch.float64)) + 1
    v = torch.randn(2, 17, 3, 8, dtype=torch.float64)
    torch.testing.assert_close(
        quadratic_linear_attention(q, k, v), recurrent_linear_attention(q, k, v)
    )