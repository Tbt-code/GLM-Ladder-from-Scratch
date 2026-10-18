"""LoRA: train a small low-rank update instead of the full weight matrix.

A frozen linear layer computes y = W x with W of shape [out, in]. LoRA adds a trainable
update of rank r:

    y = W x + (alpha / r) * B (A x)        A: [r, in], B: [out, r], r << min(in, out)

Only A and B are trained, so a 512 x 1536 matrix (786k numbers) gets 8 x (512 + 1536)
= 16k trainable numbers at rank 8. B starts at zero, so at step 0 the model is exactly
the pretrained one; A starts random, so the gradient of B is not zero.

After training the update can be folded into W (`merge_lora`): W' = W + (alpha / r) B A.
The merged model is an ordinary GLMLadderModel with no extra cost at inference.

Parameter names stay those of the original model ("...attention.qkv.weight"), plus
"...lora_A" and "...lora_B", so a pretrained checkpoint loads before LoRA is added and
a LoRA checkpoint loads into a model that has the same adapters.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TARGETS = ("attention", "experts", "all")


class LoRALinear(nn.Module):
    """A frozen nn.Linear (no bias) plus a trainable low-rank update."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if base.bias is not None:
            raise ValueError("LoRALinear supports bias-free layers only (the model has no biases)")
        if rank < 1:
            raise ValueError("rank must be >= 1")
        self.in_features, self.out_features = base.in_features, base.out_features
        self.rank, self.alpha = rank, alpha
        self.scale = alpha / rank
        self.weight = base.weight                     # same tensor, same name; frozen
        self.weight.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))     # nn.Linear's own init, as in the LoRA paper

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x,self.weight)
        update = F.linear(F.linear(x,self.lora_A), self.lora_B) * self.scale
        return base + update

    def merged_weight(self) -> torch.Tensor:
        y = self.weight + self.scale * (self.lora_B @ self.lora_A)
        return y

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, rank={self.rank}, alpha={self.alpha}"


def target_names(model: nn.Module, targets: str) -> list[str]:
    """Names of the nn.Linear modules that get adapters.

    attention: qkv and output projections of every attention layer
    experts:   up and down projections of every routed and shared expert
    all:       both. The router and the tied embedding/output matrix are never adapted:
               the router decides which experts run, and the embedding is shared with the output.
    """
    if targets not in TARGETS:
        raise ValueError(f"targets must be one of {TARGETS}")
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        parts = name.split(".")
        is_attention = len(parts) >= 2 and parts[-2] == "attention" and parts[-1] in ("qkv", "out")
        is_expert = parts[-1] in ("up", "down") and ("experts" in parts or "shared" in parts)
        if (targets in ("attention", "all") and is_attention) or (targets in ("experts", "all") and is_expert):
            names.append(name)
    return names


def _set_module(model: nn.Module, name: str, module: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child, module)


def apply_lora(model: nn.Module, *, rank: int, alpha: float, targets: str = "all") -> list[str]:
    """Freeze every parameter, then wrap the target layers in LoRALinear (in place).

    Returns the adapted module names. Only lora_A / lora_B are trainable afterwards.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    names = target_names(model, targets)
    if not names:
        raise ValueError("no layers matched; nothing to adapt")
    for name in names:
        _set_module(model, name, LoRALinear(model.get_submodule(name), rank, alpha))
    return names


def merge_lora(model: nn.Module) -> int:
    """Replace every LoRALinear by a plain nn.Linear with the merged weight (in place).

    Returns the number of merged layers. The result has the original parameter names,
    so it can be saved and loaded as an ordinary model.
    """
    lora_names = [name for name, module in model.named_modules() if isinstance(module, LoRALinear)]
    for name in lora_names:
        lora = model.get_submodule(name)
        linear = nn.Linear(lora.in_features, lora.out_features, bias=False, device=lora.weight.device)
        with torch.no_grad():
            linear.weight.copy_(lora.merged_weight())
        _set_module(model, name, linear)
    return len(lora_names)


def trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
