"""Measure real training throughput of each preset on the current GPU.

The numbers feed the compute plan: tokens/second -> hours -> Colab compute
units for each rung's token budget. Inputs are random token ids, so data
loading is not included; real runs will be slightly slower.
"""
from __future__ import annotations

import gc
import time
from typing import Any

import torch
import torch.nn.functional as F

from .config import ModelConfig, expected_parameter_counts, get_preset
from .model import GLMLadderModel
from .runtime import environment_summary

# Dense bf16 tensor-core peak per GPU (TFLOP/s), used only for the MFU estimate.
PEAK_BF16_TFLOPS = {"H100": 989.0, "A100": 312.0, "L4": 121.0, "A10": 125.0, "T4": 65.0}

# Approximate Colab compute units per hour (checked March 2026; prices change).
COLAB_UNITS_PER_HOUR = {"A100": 5.40, "L4": 1.71, "T4": 1.19}

# Planned pretraining token budgets per rung.
TOKEN_TARGETS = {"s-50m": 1.5e9, "m-100m": 3.0e9, "l-200m": 4.0e9}


def _gpu_key(table: dict[str, float]) -> str | None:
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name(0)
    return next((key for key in table if key in name), None)


def training_flops_per_token(config: ModelConfig, sequence_length: int) -> float:
    """6 * active parameters + causal attention-score FLOPs of the sparse layers.

    The sparse layers run a masked dense kernel, so we count the full causal
    score matrix. Linear-attention chunk products are small and ignored.
    """
    active = expected_parameter_counts(config)["active_per_token_estimate"]
    sparse_layers = sum(config.is_sparse_layer(index) for index in range(config.layers))
    attention = 6 * sparse_layers * config.dim * sequence_length  # fwd+bwd, causal half
    return 6.0 * active + attention


def _try_steps(
    config: ModelConfig, *, micro_batch: int, sequence_length: int, steps: int, warmup: int,
    gradient_checkpointing: bool, use_compile: bool, device: torch.device,
) -> dict[str, float]:
    model = GLMLadderModel(config).to(device)
    model.gradient_checkpointing = gradient_checkpointing
    step_model = torch.compile(model) if use_compile else model
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1,
        fused=device.type == "cuda",
    )
    ids = torch.randint(0, config.vocab_size, (micro_batch, sequence_length + 1), device=device)
    inputs, labels = ids[:, :-1], ids[:, 1:]
    torch.cuda.reset_peak_memory_stats()
    timings = []
    for step in range(warmup + steps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = step_model(inputs)
            loss = F.cross_entropy(output.logits.float().flatten(0, 1), labels.flatten())
            loss = loss + 0.01 * output.balance_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        if step >= warmup:
            timings.append(time.perf_counter() - started)
    return {
        "step_seconds": sum(timings) / len(timings),
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "final_loss": float(loss.detach()),
    }


def _free_gpu_memory() -> None:
    # Called after the failed or finished attempt's frame is gone, so its
    # tensors are unreachable and can actually be released.
    gc.collect()
    torch.cuda.empty_cache()


def benchmark_preset(
    name: str, *, sequence_length: int = 1024, micro_batch: int = 32, steps: int = 10,
    warmup: int = 3, gradient_checkpointing: bool = False, use_compile: bool = False,
) -> dict[str, Any]:
    """Benchmark one preset, halving the micro-batch until it fits in memory."""
    if not torch.cuda.is_available():
        raise RuntimeError("the throughput benchmark needs a CUDA GPU")
    torch.set_float32_matmul_precision("high")  # TF32 tensor cores for the float32 linear-attention core
    torch._dynamo.config.recompile_limit = 64            # one compiled variant per block is fine
    torch._dynamo.config.capture_dynamic_output_shape_ops = True  # let torch.where stay inside the graph
    config = get_preset(name)
    sequence_length = min(sequence_length, config.max_sequence_length)
    device = torch.device("cuda")
    batch = micro_batch
    measured = None
    while batch >= 1 and measured is None:
        try:
            measured = _try_steps(
                config, micro_batch=batch, sequence_length=sequence_length, steps=steps,
                warmup=warmup, gradient_checkpointing=gradient_checkpointing,
                use_compile=use_compile, device=device,
            )
        except torch.cuda.OutOfMemoryError:
            batch //= 2
        _free_gpu_memory()
    if measured is None:
        return {"preset": name, "status": "out_of_memory_at_batch_1"}

    tokens_per_second = batch * sequence_length / measured["step_seconds"]
    flops_per_token = training_flops_per_token(config, sequence_length)
    peak_key = _gpu_key(PEAK_BF16_TFLOPS)
    mfu = (
        tokens_per_second * flops_per_token / (PEAK_BF16_TFLOPS[peak_key] * 1e12)
        if peak_key else None
    )
    result: dict[str, Any] = {
        "preset": name,
        "status": "ok",
        "parameters": expected_parameter_counts(config),
        "micro_batch": batch,
        "sequence_length": sequence_length,
        "gradient_checkpointing": gradient_checkpointing,
        "torch_compile": use_compile,
        "step_seconds": round(measured["step_seconds"], 4),
        "tokens_per_second": round(tokens_per_second),
        "peak_vram_gib": round(measured["peak_vram_gib"], 2),
        "mfu_estimate": round(mfu, 3) if mfu is not None else None,
        "random_token_loss": round(measured["final_loss"], 3),
    }
    if name in TOKEN_TARGETS:
        hours = TOKEN_TARGETS[name] / tokens_per_second / 3600
        units_key = _gpu_key(COLAB_UNITS_PER_HOUR)
        result["pretraining_estimate"] = {
            "target_tokens": TOKEN_TARGETS[name],
            "hours_on_this_gpu": round(hours, 2),
            "colab_compute_units": round(hours * COLAB_UNITS_PER_HOUR[units_key], 1) if units_key else None,
        }
    return result


def run_benchmark(presets: list[str], **kwargs: Any) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "kind": "throughput_benchmark",
        "environment": environment_summary(),
        "settings": kwargs,
        "results": [benchmark_preset(name, **kwargs) for name in presets],
    }
