from __future__ import annotations

import math
from dataclasses import dataclass, field
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


# ----------------------------------------------------------------- generation cache
#
# Without a cache, every new token re-runs the model over the whole sequence. With it, the
# prompt is processed once ("prefill") and each new token is a forward pass of length 1
# ("decode") that reads what earlier tokens left behind:
#   linear attention: a running sum per head, state = sum_j k_j v_j^T and normalizer = sum_j k_j
#                     (fixed size, however long the sequence is)
#   sparse attention: the keys and values of every earlier position (grows by one per token)
# Everything else in the model (norms, MoE, residual streams) works on each token on its own.

@dataclass
class LinearCache:
    state: torch.Tensor | None = None        # [B, H, d, e] float32
    normalizer: torch.Tensor | None = None   # [B, H, d] float32


@dataclass
class SparseCache:
    keys: torch.Tensor | None = None         # [B, H, T, d]
    values: torch.Tensor | None = None       # [B, H, T, d]


@dataclass
class ModelCache:
    layers: list = field(default_factory=list)   # one LinearCache or SparseCache per layer
    length: int = 0                              # positions already processed


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inv_rms = torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True) + self.eps)
        return (x.float() * inv_rms * self.weight.float()).to(x.dtype)


def apply_rope(q: torch.Tensor, k: torch.Tensor, start: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to [batch, time, heads, head_dim].

    `start` is the position of the first token: 0 for a whole sequence, the number of
    cached positions when decoding with a cache.
    """
    length, width = q.shape[1], q.shape[-1]
    if width % 2:
        raise ValueError("head dimension must be even for RoPE")

    positions = torch.arange(start, start + length, device=q.device, dtype=torch.float32)
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

def linear_attention_step(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cache: LinearCache) -> torch.Tensor:
    """One new position with the recurrent form of linear attention.

    q, k: [B, 1, H, d] (already positive, ELU + 1), v: [B, 1, H, e], all float32.
    cache.state [B, H, d, e] and cache.normalizer [B, H, d] hold the sums over every earlier
    position. Update them with this position first (attention includes the token itself),
    then return the output [B, 1, H, e]:

        state      <- state + k^T v              (outer product per head)
        normalizer <- normalizer + k
        output      = (q . state) / (q . normalizer)

    This must give exactly what chunked_causal_linear_attention gives for the last position
    of the whole sequence (the test checks it).
    """

    q, k, v = q[:, 0], k[:, 0], v[:, 0]
    cache.state = cache.state + torch.einsum("bhd,bhe->bhde", k, v)
    cache.normalizer = cache.normalizer + k

    numerator = torch.einsum("bhd,bhde->bhe", q, cache.state)
    denominator = (q * cache.normalizer).sum(-1)
    output = numerator / denominator.clamp_min(1e-6)[..., None]
    
    return output.unsqueeze(1)



class LinearAttention(nn.Module):
    """Causal positive-feature (ELU + 1) linear attention."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.head_dim
        self.chunk_size = config.linear_chunk_size
        self.qkv = nn.Linear(config.dim, config.dim * 3, bias=False)
        self.out = nn.Linear(config.dim, config.dim, bias=False)

    def forward(self, x: torch.Tensor, cache: LinearCache | None = None, start: int = 0) -> torch.Tensor:
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch,length,self.heads,self.head_dim)
        k = k.view(batch,length,self.heads,self.head_dim)
        v = v.view(batch,length,self.heads,self.head_dim)
        q,k = apply_rope(q,k,start) #apply_rope to q and k

        with torch.autocast(device_type=x.device.type, enabled=False):
            q = F.elu(q.float()) + 1.0
            k = F.elu(k.float()) + 1.0
            v = v.float()
            if cache is None or cache.state is None:
                # Whole sequence, or the prompt when generating ("prefill").
                if cache is not None and start != 0:
                    raise ValueError("a linear-attention cache must be filled from position 0")
                output = chunked_causal_linear_attention(q, k, v, self.chunk_size)
                if cache is not None:
                    cache.state = torch.einsum("bthd,bthe->bhde", k, v)
                    cache.normalizer = k.sum(dim=1)
            else:
                if length != 1:
                    raise ValueError("decode one position at a time once the cache is filled")
                output = linear_attention_step(q, k, v, cache)
        return self.out(output.to(x.dtype).reshape(batch, length, dim))


def sparse_causal_mask(length: int, window: int, stride: int, device: torch.device) -> torch.Tensor:
    """Boolean [length, length] mask: True where position i may attend to position j."""
    rows = torch.arange(length, device=device)[:, None]  # i, shape [length, 1]
    cols = torch.arange(length, device=device)[None, :]  # j, shape [1, length]
    causal = cols <= rows
    local = (rows - cols) < window
    anchor = (cols % stride) == 0

    return causal & (local | anchor)

def cached_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cache: SparseCache,
                     mask_rows: torch.Tensor) -> torch.Tensor:
    """Attention for new positions that may also look at every cached position.

    q, k, v: [B, H, T, d] for the T new positions (RoPE already applied).
    cache.keys / cache.values: [B, H, P, d] for the P earlier positions, or None when empty.
    mask_rows: [T, P + T] booleans, the rows of the sparse causal mask for the new positions.
    Append the new keys and values to the cache, then attend. Returns [B, H, T, d].
    """
    if cache.keys is None or cache.values is None:
        cache.keys = k
        cache.values = v
    else:
        cache.keys = torch.cat([cache.keys,k],dim=2)
        cache.values = torch.cat([cache.values,v],dim=2)
    return F.scaled_dot_product_attention(q,cache.keys,cache.values,attn_mask=mask_rows)


def sparse_mask_rows(start: int, length: int, window: int, stride: int, device: torch.device) -> torch.Tensor:
    """Rows start .. start + length - 1 of sparse_causal_mask(start + length), built directly.

    Caching a full mask for every length a decode passes through would hold ~0.5 GB of
    booleans per layer at 1024 tokens; one row per step is tiny.
    """
    rows = torch.arange(start, start + length, device=device)[:, None]
    cols = torch.arange(start + length, device=device)[None, :]
    return (cols <= rows) & (((rows - cols) < window) | ((cols % stride) == 0))


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

    def forward(self, x: torch.Tensor, cache: SparseCache | None = None, start: int = 0) -> torch.Tensor:
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
       
        q = q.view(batch, length, self.heads, self.head_dim)
        k = k.view(batch, length, self.heads, self.head_dim)
        v = v.view(batch,length,self.heads,self.head_dim)

        q,k = apply_rope(q,k,start)

        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)
        if cache is None:
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=self.mask(length, x.device))
        else:
            mask_rows = sparse_mask_rows(start, length, self.window, self.stride, x.device)
            output = cached_attention(q, k, v, cache, mask_rows)
    
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

def dense_expert_mix(flat: torch.Tensor, experts: nn.ModuleList, top_indices: torch.Tensor,
                     top_weights: torch.Tensor) -> torch.Tensor:
    """Routed-expert output for a few tokens, computed without data-dependent shapes.

    flat: [tokens, dim]; top_indices, top_weights: [tokens, k] from the router.
    Every expert runs on every token, and each result is weighted by that expert's gate:
    the router weight if the token picked the expert, 0 otherwise. The result is the same
    as sending each token only to its k experts, but nothing depends on how many tokens an
    expert got, so the GPU never has to report a count back to the CPU.
    Returns [tokens, dim] (without the shared expert).
    """
    tokens = flat.shape[0]
    experts_count = len(experts)
    gates = torch.zeros(tokens, experts_count, dtype=top_weights.dtype, device=flat.device).scatter(1, top_indices, top_weights)
    
    output = torch.zeros_like(flat)
    for e, expert in enumerate(experts):
        expert_out = expert(flat)
        output = output + expert_out * gates[:, e:e+1].to(expert_out.dtype)
    return output


class SparseMoE(nn.Module):
    """Top-k routed experts plus one always-active shared expert.

    Two ways to compute the same thing. Training and large batches send each token only to
    its k experts (sparse). In eval mode with at most `dense_max_tokens` tokens (one decode
    step of generation) every expert runs on every token instead (dense): more arithmetic,
    but no torch.where / bincount, whose output sizes depend on the data and force the CPU
    to wait for the GPU, ~9 times per layer per generated token.
    """

    dense_max_tokens = 256

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

        dense = not self.training and tokens <= self.dense_max_tokens
        if dense:   # same counts as bincount, without a data-dependent output size
            counts = F.one_hot(top_indices.reshape(-1), experts).sum(dim=0).float()
        else:
            counts = torch.bincount(top_indices.reshape(-1),minlength=experts).float()
        fraction = counts / (tokens * self.top_k)
        mean_probability = torch.softmax(logits,dim=-1).mean(dim=0)
        balance_loss = experts * (fraction * mean_probability).sum()

        if dense:
            output = self.shared(flat)
            output = output + dense_expert_mix(flat, self.experts, top_indices, top_weights).to(output.dtype)
            return output.view(original_shape), balance_loss, fraction.detach()

        output = self.shared(flat)
        # torch.compile kernels fail (ZeroDivisionError) on zero-sized data-dependent shapes,
        # which happen whenever the router sends no token to an expert. Every expert therefore
        # also gets one zero "dummy" row (index `tokens`) with weight 0; it is dropped at the end.
        padded = torch.cat([flat, flat.new_zeros(1, flat.shape[-1])])
        output = torch.cat([output, output.new_zeros(1, output.shape[-1])])
        dummy = torch.full((1,), tokens, dtype=torch.long, device=flat.device)
        for expert_id, expert in enumerate(self.experts):
            positions, slots = torch.where(top_indices == expert_id)
            weights = torch.cat([top_weights[positions, slots], top_weights.new_zeros(1)])
            positions = torch.cat([positions, dummy])
            contribution = expert(padded[positions]) * weights[:, None]
            output.index_add_(0, positions, contribution.to(output.dtype))
        return output[:tokens].view(original_shape), balance_loss, fraction.detach()



class HybridBlock(nn.Module):
    def __init__(self, config: ModelConfig, sparse: bool):
        super().__init__()
        self.attention_norm = RMSNorm(config.dim)
        self.ffn_norm = RMSNorm(config.dim)
        self.attention = SparseAttention(config) if sparse else LinearAttention(config)
        self.moe = SparseMoE(config)

    def forward(self, x: torch.Tensor, cache=None, start: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normed = self.attention_norm(x)
        attended = self.attention(normed) if cache is None else self.attention(normed, cache, start)
        x = x + attended #pre-norm residual attention, then pre-norm residual MoE.
        ffn,balance_loss,usage = self.moe(self.ffn_norm(x))
        return x + ffn,balance_loss,usage

class HyperConnection(nn.Module):
    """Residual streams with learned input mixing and output routing."""

    def __init__(self, config: ModelConfig, block: HybridBlock):
        super().__init__()
        self.block = block
        self.input_logits = nn.Parameter(torch.linspace(0.1, -0.1, config.streams))
        self.output_logits = nn.Parameter(torch.linspace(-0.1, 0.1, config.streams))

    def forward(self, streams: torch.Tensor, cache=None, start: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mixed = torch.einsum("s,btsd->btd", torch.softmax(self.input_logits, dim=0).to(streams.dtype), streams)
        transformed, balance_loss, usage = self.block(mixed) if cache is None else self.block(mixed, cache, start)
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

    def new_cache(self) -> ModelCache:
        """An empty generation cache; pass it to forward() for the prompt, then one token at a time."""
        return ModelCache(layers=[SparseCache() if self.config.is_sparse_layer(index) else LinearCache()
                                  for index in range(self.config.layers)])

    def forward_embeddings(self, embedded: torch.Tensor, cache: ModelCache | None = None) -> ModelOutput:
        """Run the language model from precomputed token-like embeddings.

        With `cache`, the inputs are the next positions after cache.length (all of the prompt
        first, then one token per call), and the cache is updated in place.
        """
        if embedded.ndim != 3 or embedded.shape[-1] != self.config.dim:
            raise ValueError("embedded inputs must have shape [batch, time, dim]")
        start = 0 if cache is None else cache.length
        if start + embedded.shape[1] > self.config.max_sequence_length:
            raise ValueError("input exceeds max_sequence_length")

        # One embedding copied into every stream: [B, T, D] -> [B, T, S, D]
        streams = embedded.unsqueeze(2).expand(-1, -1, self.config.streams, -1).contiguous()

        balance_losses, usages = [], []
        for index, layer in enumerate(self.layers):
            if cache is not None:
                streams, balance_loss, usage = layer(streams, cache.layers[index], start)
            elif self.gradient_checkpointing and self.training:
                streams, balance_loss, usage = checkpoint(layer, streams, use_reentrant=False)
            else:
                streams,balance_loss,usage = layer(streams)
            balance_losses.append(balance_loss)
            usages.append(usage)
        hidden = self.final_norm(streams.mean(dim=2))
        return ModelOutput(self.output(hidden), torch.stack(balance_losses).mean(), torch.stack(usages))


    def forward(self, input_ids: torch.Tensor, cache: ModelCache | None = None) -> ModelOutput:
        output = self.forward_embeddings(self.embedding(input_ids), cache)
        if cache is not None:
            cache.length += input_ids.shape[1]
        return output

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
