"""Save and restore everything needed to continue training exactly where it stopped.

A checkpoint directory holds:
  model.safetensors  weights (no pickle; tied embedding/output stored once)
  optimizer.pt       AdamW moments and RNG states, loaded with weights_only=True
  trainer.json       step, loader state, config, and the SHA-256 of model.safetensors
It is written to `<name>.partial` and renamed when complete, so a directory
without the suffix is never half-written.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_model, save_model

MODEL_FILE = "model.safetensors"
OPTIMIZER_FILE = "optimizer.pt"
STATE_FILE = "trainer.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_checkpoint(
    directory: str | Path, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer, trainer_state: dict,
) -> dict:
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    staging = directory.with_name(directory.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)              # leftover from an interrupted save
    staging.mkdir(parents=True)
    save_model(model, str(staging / MODEL_FILE))
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        staging / OPTIMIZER_FILE,
    )
    state = {**trainer_state, "model_sha256": sha256_file(staging / MODEL_FILE)}
    (staging / STATE_FILE).write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    staging.rename(directory)
    return state


def load_checkpoint(
    directory: str | Path, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device,
) -> dict:
    """Restore model, optimizer and RNG in place; return the trainer state (step, loader, ...)."""
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {directory}")
    
    model_path = directory / MODEL_FILE
    optimizer_path = directory / OPTIMIZER_FILE
    state_path = directory / STATE_FILE

    state = json.loads(state_path.read_text())

    actual_sha256 = sha256_file(model_path)
    if actual_sha256 != state.get("model_sha256"):
        raise ValueError(
            f"Model checkpoint integrity check failed! Expected SHA-256: {state.get('model_sha256')}, got: {actual_sha256}"
        )
    load_model(model,str(model_path), strict=True, device=str(device))

    extras = torch.load(optimizer_path, map_location=device, weights_only=True)
    optimizer.load_state_dict(extras["optimizer"])

    
    torch_rng = extras["torch_rng"]
    if isinstance(torch_rng, torch.Tensor):
        torch_rng = torch_rng.cpu()
    torch.set_rng_state(torch_rng)
    
    if torch.cuda.is_available() and extras.get("cuda_rng"):
        cuda_rngs = [state.cpu() if isinstance(state, torch.Tensor) else state for state in extras["cuda_rng"]]
        torch.cuda.set_rng_state_all(cuda_rngs)
        
    return state