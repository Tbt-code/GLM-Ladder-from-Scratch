"""Scaled GLM-5.3-Flash-style hybrid-attention MoE language model."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inv_rms = torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True) + self.eps)
        return (x.float() * inv_rms * self.weight.float()).to(x.dtype)


def apply_rope(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to [batch, time, heads, head_dim]."""
    length, width = q.shape[1], q.shape[-1]
    if width % 2:
        raise ValueError("head dimension must be even for RoPE")

    positions = torch.arange(length,device = q.device,dtype = torch.float32)
    frequencies = 1 / (10000 ** (torch.arange(0,width,2,device=q.device).float() / width))
    angles = positions[:,None] * frequencies[None,:]
    cos = angles.cos()[None, :, None, :].to(q.dtype)
    sin = angles.sin()[None, :, None, :].to(q.dtype)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        even, odd = x[..., 0::2], x[..., 1::2]
        new_even = even * cos - odd * sin
        new_odd = even * sin + odd * cos
        return torch.stack((new_even, new_odd), dim=-1).flatten(-2)

    return rotate(q), rotate(k)

class Expert(nn.Module):
    """SwiGLU feed-forward expert."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.up = nn.Linear(dim, 2*hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate,value = self.up(x).chunk(2,dim=-1)
        return self.down(F.silu(gate) * value)
        

def sparse_causal_mask(length: int, window: int, stride: int, device: torch.device) -> torch.Tensor:
    """Boolean [length, length] mask: True where position i may attend to position j."""
    rows = torch.arange(length, device=device)[:, None]  # i, shape [length, 1]
    cols = torch.arange(length, device=device)[None, :]  # j, shape [1, length]
    causal = cols <= rows
    local = (rows - cols) < window
    anchor = (cols % stride) == 0

    return causal & (local | anchor)


class SparseAttention(nn.Module):
    """Causal attention restricted to a local window plus strided anchors."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.head_dim
        self.window = config.sparse_window
        self.stride = config.sparse_stride
        self.qkv = nn.Linear(config.dim, config.dim * 3, bias=False)
        self.out = nn.Linear(config.dim, config.dim, bias=False)
        self._mask_cache: dict[tuple[int, str], torch.Tensor] = {}

    def mask(self, length: int, device: torch.device) -> torch.Tensor:
        # Build the mask once per (length, device) instead of every forward pass.
        key = (length, str(device))
        if key not in self._mask_cache:
            self._mask_cache[key] = sparse_causal_mask(length, self.window, self.stride, device)
        return self._mask_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
       
        q = q.view(batch, length, self.heads, self.head_dim)
        k = k.view(batch, length, self.heads, self.head_dim)
        v = v.view(batch,length,self.heads,self.head_dim)

        q,k = apply_rope(q,k)

        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=self.mask(length, x.device))
    
        return self.out(output.transpose(1,2).reshape(batch,length,dim))
 