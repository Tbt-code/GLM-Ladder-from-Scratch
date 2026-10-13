from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig


class ModelOutput(NamedTuple):
    logits: torch.Tensor        # [batch, time, vocab]
    balance_loss: torch.Tensor  # scalar, mean over layers; 1.0 means perfectly balanced
    expert_usage: torch.Tensor  # [layers, experts], fraction of routed slots (detached)


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
def chunked_causal_linear_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    """q, k: [B, T, H, d] (positive); v: [B, T, H, e]. Same result as the recurrent form."""
    batch, length, heads, width = q.shape
    value_width = v.shape[-1]

    # 1. Pad T up to a multiple of chunk_size.
    pad = (-length) % chunk_size
    if pad:
        # F.pad pads from the LAST dim backwards: (d_left, d_right, H_left, H_right, T_left, T_right)
        q = F.pad(q,(0,0,0,0,0,pad))
        k = F.pad(k,(0,0,0,0,0,pad))
        v = F.pad(v,(0,0,0,0,0,pad))
    chunks = (length + pad) // chunk_size

    # Split time into chunks: [B, T, H, d] -> [B, N, C, H, d]
    q = q.view(batch,chunks,chunk_size,heads,width)
    k = k.view(batch,chunks,chunk_size,heads,width)
    v = v.view(batch,chunks,chunk_size,heads,value_width)

    # Intra-chunk: causal quadratic attention inside each chunk.

    scores = torch.einsum("bnihd,bnjhd->bnhij", q, k)        # [B, N, H, C, C]
    causal = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device).tril()
    scores = scores.masked_fill(~causal,0.0)
    numerator = torch.einsum("bnhij,bnjhe->bnihe", scores, v)  # [B, N, C, H, e]
    denominator = scores.sum(dim=-1).permute(0, 1, 3, 2)       # [B, N, C, H]

    # 4. Inter-chunk: one state per chunk, then exclusive prefix over chunks.
    chunk_kv = torch.einsum("bnjhd,bnjhe->bnhde", k, v)       # [B, N, H, d, e]
    prefix_kv = chunk_kv.cumsum(dim=1) - chunk_kv
    chunk_k = k.sum(dim=2)                                     # [B, N, H, d]
    prefix_k = chunk_k.cumsum(dim=1) - chunk_k
    numerator = numerator + torch.einsum("bnihd,bnhde->bnihe", q, prefix_kv)
    denominator = denominator + torch.einsum("bnihd,bnhd->bnih", q, prefix_k)

    # 5. Normalise, merge chunks back into time, drop the padding.
    output = numerator / denominator.clamp_min(1e-6).unsqueeze(-1)
    return output.reshape(batch,chunks * chunk_size,heads,value_width)[:,:length]

class LinearAttention(nn.Module):
    """Causal positive-feature (ELU + 1) linear attention."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.head_dim
        self.chunk_size = config.linear_chunk_size
        self.qkv = nn.Linear(config.dim, config.dim * 3, bias=False)
        self.out = nn.Linear(config.dim, config.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch,length,self.heads,self.head_dim)
        k = k.view(batch,length,self.heads,self.head_dim)
        v = v.view(batch,length,self.heads,self.head_dim)
        q,k = apply_rope(q,k) #apply_rope to q and k

        with torch.autocast(device_type=x.device.type, enabled=False):
            q = F.elu(q.float()) + 1.0
            k = F.elu(k.float()) + 1.0
            output = chunked_causal_linear_attention(q,k,v.float(),self.chunk_size)
        return self.out(output.to(x.dtype).reshape(batch, length, dim))


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

 
class Expert(nn.Module):
    """SwiGLU feed-forward expert."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.up = nn.Linear(dim, 2*hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate,value = self.up(x).chunk(2,dim=-1)
        return self.down(F.silu(gate) * value)

class SparseMoE(nn.Module):
    """Top-k routed experts plus one always-active shared expert."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.router = nn.Linear(config.dim, config.experts, bias=False)
        self.experts = nn.ModuleList(
            [Expert(config.dim, config.expert_hidden) for _ in range(config.experts)]
        )
        self.shared = Expert(config.dim, config.expert_hidden)
        self.top_k = config.top_k

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])          # [tokens, dim]
        tokens, experts = flat.shape[0], len(self.experts)

        logits = self.router(flat).float()                # [tokens, E]
        top_values, top_indices = logits.topk(self.top_k, dim=-1)
        top_weights = torch.softmax(top_values, dim=-1)   # [tokens, k]

        counts = torch.bincount(top_indices.reshape(-1),minlength=experts).float()
        fraction = counts / (tokens * self.top_k)
        mean_probability = torch.softmax(logits,dim=-1).mean(dim=0)
        balance_loss = experts * (fraction * mean_probability).sum()


        output = self.shared(flat)
        for expert_id, expert in enumerate(self.experts):
            positions, slots = torch.where(top_indices == expert_id)
            
            token_slice = flat[positions]
            contribution = expert(token_slice) * top_weights[positions,slots][:,None]
            contribution = contribution.to(output.dtype)
            output.index_add_(0,positions,contribution)
        return output.view(original_shape), balance_loss, fraction.detach()



class HybridBlock(nn.Module):
    def __init__(self, config: ModelConfig, sparse: bool):
        super().__init__()
        self.attention_norm = RMSNorm(config.dim)
        self.ffn_norm = RMSNorm(config.dim)
        self.attention = SparseAttention(config) if sparse else LinearAttention(config)
        self.moe = SparseMoE(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x + self.attention(self.attention_norm(x)) #pre-norm residual attention, then pre-norm residual MoE.
        ffn,balance_loss,usage = self.moe(self.ffn_norm(x))
        return x + ffn,balance_loss,usage

class HyperConnection(nn.Module):
    """Residual streams with learned input mixing and output routing."""

    def __init__(self, config: ModelConfig, block: HybridBlock):
        super().__init__()
        self.block = block
        self.input_logits = nn.Parameter(torch.linspace(0.1, -0.1, config.streams))
        self.output_logits = nn.Parameter(torch.linspace(-0.1, 0.1, config.streams))

    def forward(self, streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mixed = torch.einsum("s,btsd->btd", torch.softmax(self.input_logits, dim=0).to(streams.dtype), streams)
        transformed, balance_loss, usage = self.block(mixed)
        routed = torch.softmax(self.output_logits,dim=0).to(streams.dtype)
        streams = streams + transformed.unsqueeze(2) * routed[None, None, :, None]
        return streams,balance_loss,usage

class GLMLadderModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([
            HyperConnection(config, HybridBlock(config, sparse=config.is_sparse_layer(index)))
            for index in range(config.layers)
        ])
        self.final_norm = RMSNorm(config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.output.weight = self.embedding.weight # tied input/output matrix
        self.apply(self._init_weights)
        if config.scaled_residual_init:
            self._scale_residual_projections()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_projections(self) -> None:
        # Projections that write into the residual stream get a smaller init
        # so the stream variance does not grow with depth (GPT-2 recipe).
        std = 0.02 / math.sqrt(2 * self.config.layers)
        for layer in self.layers:
            nn.init.normal_(layer.block.attention.out.weight, mean=0.0, std=std)
            for expert in [*layer.block.moe.experts, layer.block.moe.shared]:
                nn.init.normal_(expert.down.weight, mean=0.0, std=std)

    def forward_embeddings(self, embedded: torch.Tensor) -> ModelOutput:
        """Run the language model from precomputed token-like embeddings."""
        if embedded.ndim != 3 or embedded.shape[-1] != self.config.dim:
            raise ValueError("embedded inputs must have shape [batch, time, dim]")
        if embedded.shape[1] > self.config.max_sequence_length:
            raise ValueError("input exceeds max_sequence_length")

        # One embedding copied into every stream: [B, T, D] -> [B, T, S, D]
        streams = embedded.unsqueeze(2).expand(-1, -1, self.config.streams, -1).contiguous()

        balance_losses, usages = [], []
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                streams, balance_loss, usage = checkpoint(layer, streams, use_reentrant=False)
            else:
                streams,balance_loss,usage = layer(streams)
            balance_losses.append(balance_loss)
            usages.append(usage)
        hidden = self.final_norm(streams.mean(dim=2))
        return ModelOutput(self.output(hidden), torch.stack(balance_losses).mean(), torch.stack(usages))


    def forward(self, input_ids: torch.Tensor) -> ModelOutput:
        return self.forward_embeddings(self.embedding(input_ids))

    def parameter_counts(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        # Active count is an architectural estimate: all non-routed parameters,
        # the shared expert, and top-k of the routed expert parameters per layer.
        routed = sum(parameter.numel() for layer in self.layers
                    for expert in layer.block.moe.experts
                    for parameter in expert.parameters()
                )
        active = total - routed + routed * self.config.top_k // self.config.experts
        return {"total": total, "active_per_token_estimate": active}
