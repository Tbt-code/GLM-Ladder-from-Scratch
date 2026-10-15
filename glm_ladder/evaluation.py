"""Validation loss on fixed windows, so values are comparable across steps and runs."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .loader import PackedSource


@torch.no_grad()
def evaluate(
    model: nn.Module, sources: Mapping[str, PackedSource], *,
    windows: int, batch_size: int, device: torch.device,
) -> dict[str, float]:
    """Mean next-token cross-entropy over the first `windows` windows of each source."""
    model.eval()
    results: dict[str, float] = {}
    for name,source in sources.items():
        num_windows = min(windows,source.num_windows)
        if num_windows == 0:
            results[name] = 0.0
            continue

        total_loss = 0.0
        total_count = 0

        indices = list(range(num_windows))
        for i in range(0,num_windows,batch_size):
            chunk = indices[i : i + batch_size]
            rows = np.stack([source.window(idx) for idx in chunk]).astype(np.int64)
            x = torch.as_tensor(rows[:,:-1],device=device)
            y = torch.as_tensor(rows[:,1:],device=device)

            with torch.autocast(device_type = device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(x)
                logits = output.logits

                loss_sum = F.cross_entropy(logits.float().flatten(0,1),y.flatten(), reduction="sum")

            total_loss += loss_sum.item()
            total_count += y.numel()

        results[name] = total_loss / total_count if total_count > 0 else 0.0
    model.train()
    return results