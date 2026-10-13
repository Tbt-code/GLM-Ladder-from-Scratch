"""Model configurations for the GLM ladder."""
from __future__ import annotations
from dataclasses import asdict,dataclass,fields,replace
from typing import Any

@dataclass(frozen=True)
class ModelConfig:
    # Same names and defaults as the original repo -> its config.json loads as-is.
    vocab_size: int = 260 # byte tokenizer: 256 bytes + 4 special tokens
    dim: int = 192 # width of the residual stream
    layers: int = 12
    heads: int = 6
    expert_hidden: int = 384 # hidden width inside each SwiGLU expert
    experts: int = 8 # routed experts per layer
    top_k: int = 2 # experts each token is sent to
    streams: int = 4  # parallel residual streams (hyper-connections)
    sparse_window: int = 32
    sparse_stride: int = 32
    max_sequence_length: int = 192


    # New fields; defaults reproduce the original behaviour.
    sparse_every: int = 4
    linear_chunk_size: int = 64
    scaled_residual_init: bool =False

    def __post_init__(self) -> None:
        if self.dim % self.heads:
            raise ValueError("dim must be divisible by heads")

        if (self.dim // self.heads) % 2:
            raise ValueError("Head dims must be even in RoPE.")

        if not 1 <= self.top_k <= self.experts:
            raise ValueError("Top k must be between 1 and experts.")

        if self.linear_chunk_size < 1:
            raise ValueError("Linear chunk size must be positive")

        if self.sparse_window < 1 and self.sparse_stride < 1:
            raise ValueError("All masks are -inf, NaN")
        

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads

    def is_sparse_layer(self,index:int) -> bool:
        return (index + 1) % self.sparse_every == 0

    def to_dict(self) -> dict[str,Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls,values:dict[str,Any]) -> "ModelConfig":
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)} ")
        return cls(**values)

    

def expected_parameter_counts(config:ModelConfig) -> dict[str,int]:
    """Closed-form parameter count, used to cross-check the PyTorch module.
    
        Per block: attention qkv + out (4 d^2), router (d * E), routed and shared
        SwiGLU experts ((E + 1) * 3 * d * h), two RMSNorms (2 d), and the
        stream-mixing logits (2 * streams). The output matrix is tied to the
        embedding, so it adds no parameters.
        """
    d,e,h,s,l,v = config.dim,config.experts, config.expert_hidden,config.streams,config.layers,config.vocab_size
    param_per_block = 4 * d**2 + d * e + 3 * (e + 1) * d * h + 2 * d + 2 * s
    total_params = l * param_per_block + v * d + d
    total_routed_params = l * (e * 3 * d * h)
    active_params = total_params - total_routed_params + total_routed_params * config.top_k // e

    return {"total": total_params, "active_per_token_estimate": active_params}

BPE_VOCAB_SIZE = 16_384

PRESETS: dict[str, ModelConfig] = {
    # Exact architecture of the original repository (byte tokenizer).
    "baseline-25m": ModelConfig(),
    # Ladder rungs (BPE tokenizer, 1024-token context).
    "s-50m": ModelConfig(
        vocab_size=BPE_VOCAB_SIZE, dim=384, layers=12, heads=6, expert_hidden=320,
        sparse_window=128, sparse_stride=128, max_sequence_length=1024,
        scaled_residual_init=True,
    ),
    "m-100m": ModelConfig(
        vocab_size=BPE_VOCAB_SIZE, dim=512, layers=16, heads=8, expert_hidden=320,
        sparse_window=128, sparse_stride=128, max_sequence_length=1024,
        scaled_residual_init=True,
    ),
    "l-200m": ModelConfig(
        vocab_size=BPE_VOCAB_SIZE, dim=640, layers=18, heads=10, expert_hidden=512,
        sparse_window=128, sparse_stride=128, max_sequence_length=1024,
        scaled_residual_init=True,
    ),
}

def get_preset(name:str) -> ModelConfig:
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}") from None 

if __name__ == "__main__":
    print("Model Param Breakdowns")
    for name, config in PRESETS.items():
        d,e,h,s,l,v =  config.dim,config.experts, config.expert_hidden,config.streams,config.layers,config.vocab_size

        attention = l * (4 * d**2)
        
        experts = l * ((e + 1) * 3 * d * h)
        
        embedding = v * d
        
        rest = l * (d * e + 2 * d + 2 * s) + d

        total = attention + experts + embedding + rest

        print(f"\nPreset: {name}")
        print(f"  - Attention:               {attention:,}")
        print(f"  - Experts (routed+shared): {experts:,}")
        print(f"  - Embedding:               {embedding:,}")
        print(f"  - Rest (router/norms/etc): {rest:,}")
        print(f"  - Total Calculated:        {total:,}")
