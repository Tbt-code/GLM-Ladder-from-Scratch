"""One optimizer step with gradient accumulation, plus the LR schedule and AdamW groups."""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .loader import IGNORE_INDEX


def lr_at(step: int, *, max_lr: float, warmup_steps: int, total_steps: int, min_ratio: float = 0.1) -> float:
    """Linear warmup to max_lr, then cosine decay to min_ratio * max_lr at total_steps.

    step 0 is the first optimizer step. After total_steps the rate stays at the minimum.
    """
    min_lr = min_ratio * max_lr

    if step < warmup_steps:
        # Warmup: lr grows linearly; step 0 should NOT be zero -> use (step + 1) / warmup_steps
        return max_lr * (step + 1) / warmup_steps
    elif step >= total_steps:
        return min_lr
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        progress = min(max(progress,0.0),1.0)

        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = min_lr + (max_lr - min_lr) * cosine_factor
        return lr


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Two AdamW groups: decay for matrices (>= 2-D), no decay for vectors and scalars."""
    decay = []
    no_decay = []

    # Loop over model.named_parameters() (it lists the tied embedding/output matrix once)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Put p.ndim >= 2 into the decay list, everything else into the no-decay list.
        if p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
            
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def lm_loss(logits: torch.Tensor, targets: torch.Tensor, *, per_window: bool = False) -> torch.Tensor:
    """Next-token cross-entropy over the learned targets (targets != IGNORE_INDEX).

    per_window=False: mean over every learned token in the batch. A window with 600 learned
        tokens weighs 300x more than one with 2 (pretraining: all windows are full, so this is
        the same as per_window=True).
    per_window=True: first the mean over each window's own learned tokens, then the mean over
        windows. Every window counts equally, whatever it learns: when chat windows (~600 learned
        tokens) and review windows (~2 per review) are mixed, each source then contributes in
        proportion to its share of windows. Windows without any learned token are left out;
        if no window has one, the loss is 0 (still connected to the graph, so backward works).
    """
    flat_logits = logits.float().flatten(0, 1)
    if not per_window:
        return F.cross_entropy(flat_logits, targets.flatten())

    B,T = targets.shape
    losses = F.cross_entropy(flat_logits,targets.flatten(),reduction="none").view(B,T)

    learned = targets != IGNORE_INDEX

    counts = learned.sum(dim=1)

    valid_mask = counts > 0
    if not valid_mask.any():
        return flat_logits.sum() * 0.0

    window_losses = losses.sum(dim=1)[valid_mask] / counts[valid_mask].float()
    return window_losses.mean()


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    micro_batches: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    device: torch.device,
    balance_coef: float = 0.01,
    grad_clip: float = 1.0,
    forward: nn.Module | None = None,
    per_window: bool = False,
) -> dict[str, float]:
    """Accumulate gradients over `micro_batches`, clip, and take one optimizer step.

    `per_window`: how the language-model loss averages, see `lm_loss`.
    `forward` may be a torch.compile'd wrapper of `model`; gradients still land in
    `model`'s parameters because the wrapper shares them.
    """
    forward = model if forward is None else forward
    model.train()
    optimizer.zero_grad(set_to_none=True)
    steps = len(micro_batches)
    lm_sum = torch.zeros((), device=device)
    balance_sum = torch.zeros((), device=device)
    for inputs, targets in micro_batches:
        x = torch.as_tensor(inputs, device=device)
        y = torch.as_tensor(targets, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = forward(x)
            language_loss = lm_loss(output.logits, y, per_window=per_window)
            loss = (language_loss + balance_coef * output.balance_loss) / steps
        loss.backward()
        lm_sum += language_loss.detach() / steps
        balance_sum += output.balance_loss.detach() / steps
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    if not torch.isfinite(grad_norm):
        raise FloatingPointError(f"non-finite gradient norm: {grad_norm}")
    optimizer.step()
    return {
        "loss": float(lm_sum),          # one GPU->CPU sync per step, not per micro-batch
        "balance_loss": float(balance_sum),
        "grad_norm": float(grad_norm),
    }