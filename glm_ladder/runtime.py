"""Device selection and seeding helpers shared by notebooks and scripts."""
from __future__ import annotations

import random

import numpy as np
import torch


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment_summary() -> dict[str, object]:
    """Hardware and software facts recorded in every receipt."""
    summary: dict[str, object] = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        summary.update({
            "gpu": properties.name,
            "gpu_memory_gib": round(properties.total_memory / 1024**3, 2),
            "cuda": torch.version.cuda,
            "bf16_supported": torch.cuda.is_bf16_supported(),
        })
    return summary
