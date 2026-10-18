"""The pretraining loop: schedule, evaluation, checkpoints and resume in one function.

Everything the loop needs to continue after an interruption lives in the latest
checkpoint on the Hub: weights, AdamW state, step, tokens seen and the loader
state. Rerunning `run_pretraining` with the same config continues exactly there.
"""
from __future__ import annotations

import math
import shutil
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
from safetensors.torch import load_model

from .checkpoint import load_checkpoint, save_checkpoint
from .config import ModelConfig
from .evaluation import evaluate
from .loader import MixedLoader, PackedSource
from .lora import apply_lora, trainable_parameters
from .model import GLMLadderModel
from .runs import MetricsLog, RunStore
from .train import lr_at, param_groups, train_step


@dataclass(frozen=True)
class PretrainConfig:
    run_name: str
    total_tokens: int
    seq_len: int = 1024
    batch_rows: int = 256              # sequences per optimizer step: 256 x 1024 = 262,144 tokens
    micro_batch: int = 32              # sequences per forward pass; batch_rows / micro_batch = accumulation
    max_lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    balance_coef: float = 0.01
    grad_clip: float = 1.0
    seed: int = 1234
    eval_every: int = 250
    eval_windows: int = 128            # per source
    log_every: int = 10
    checkpoint_minutes: float = 25.0
    checkpoint_every_steps: int | None = None
    compile: bool = False
    lora_rank: int = 0                 # 0: train every parameter; > 0: freeze the model and train LoRA adapters
    lora_alpha: float = 16.0
    lora_targets: str = "all"          # "attention", "experts" or "all" (see glm_ladder/lora.py)

    @property
    def tokens_per_step(self) -> int:
        return self.batch_rows * self.seq_len

    @property
    def total_steps(self) -> int:
        return math.ceil(self.total_tokens / self.tokens_per_step)


# Fields that define *what* is trained. Changing any of them mid-run would silently
# produce a different experiment, so a resume refuses. Everything else (micro_batch,
# compile, logging and checkpoint cadence) may change, e.g. when moving from L4 to A100.
RESUME_FIELDS = (
    "run_name", "total_tokens", "seq_len", "batch_rows", "max_lr", "min_lr_ratio",
    "warmup_steps", "weight_decay", "beta1", "beta2", "balance_coef", "grad_clip", "seed",
    "lora_rank", "lora_alpha", "lora_targets",
)
# A checkpoint saved before a field existed is compared with that field's default.
DEFAULTS = {field.name: field.default for field in fields(PretrainConfig)}


def check_resume_compatible(saved: Mapping, current: Mapping) -> None:
    """Raise ValueError naming every RESUME_FIELDS entry whose value differs."""
    # collect the fields in RESUME_FIELDS
    diffs = [f"{field}: {saved.get(field, DEFAULTS[field])!r} -> {current.get(field)!r}"
             for field in RESUME_FIELDS
             if saved.get(field, DEFAULTS[field]) != current.get(field)
        ]
    if diffs:
        raise ValueError(", ".join(diffs))

    


def run_pretraining(
    cfg: PretrainConfig,
    model_config: ModelConfig,
    *,
    train_sources: Mapping[str, PackedSource],
    val_sources: Mapping[str, PackedSource],
    weights: Mapping[str, float],
    store: RunStore | None,
    workdir: str | Path,
    device: torch.device,
    metadata: Mapping | None = None,
    log: Callable[[str], None] = print,
    stop_after_steps: int | None = None,
    init_from: str | Path | None = None,
    extra_val_sources: Mapping[str, PackedSource] | None = None,
) -> dict:
    """Train until cfg.total_steps, resuming from the store's latest checkpoint if there is one.

    `init_from`: a model.safetensors to start a *new* run from (fine-tuning); ignored when resuming,
    because the checkpoint already holds the fine-tuned weights.
    `extra_val_sources`: also evaluated at every eval and logged as "eval_extra", e.g. the
    pretraining validation sets during SFT, to measure forgetting.
    `cfg.lora_rank > 0`: the model is frozen and only LoRA adapters train (see glm_ladder/lora.py);
    checkpoints then hold the adapters next to the frozen weights.
    `stop_after_steps` stops without saving, to simulate an interrupted session in tests.
    """
    if cfg.batch_rows % cfg.micro_batch:
        raise ValueError("batch_rows must be a multiple of micro_batch")
    if cfg.seq_len > model_config.max_sequence_length:
        raise ValueError("seq_len exceeds the model's max_sequence_length")
    for source in [*train_sources.values(), *val_sources.values(), *(extra_val_sources or {}).values()]:
        if source.seq_len != cfg.seq_len:
            raise ValueError("every source must use cfg.seq_len")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")      # TF32 for the float32 linear-attention core

    torch.manual_seed(cfg.seed)
    workdir = Path(workdir)
    run_dir = workdir / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    model = GLMLadderModel(model_config).to(device)
    latest = store.latest() if store is not None else None
    # Order matters for fine-tuning: pretrained weights load into the plain model (their names
    # match), then LoRA wraps it, then the optimizer sees only the parameters that train.
    if latest is None and init_from is not None:
        load_model(model, str(init_from), strict=True, device=str(device))
        log(f"initialised from {init_from}")
    if cfg.lora_rank > 0:
        adapted = apply_lora(model, rank=cfg.lora_rank, alpha=cfg.lora_alpha, targets=cfg.lora_targets)
        log(f"LoRA rank {cfg.lora_rank}, alpha {cfg.lora_alpha} on {len(adapted)} layers ({cfg.lora_targets})")
    groups = [group for group in param_groups(model, cfg.weight_decay) if group["params"]]
    optimizer = torch.optim.AdamW(groups, lr=0.0, betas=(cfg.beta1, cfg.beta2), fused=device.type == "cuda")
    # The loader yields whole optimizer steps, so the data order does not depend on micro_batch.
    loader = MixedLoader(train_sources, weights, batch_size=cfg.batch_rows, seed=cfg.seed)
    metrics = MetricsLog(run_dir / "metrics.jsonl")
    step, tokens_seen = 0, 0

    if latest is not None:
        pulled = workdir / "pulled"
        if pulled.exists():
            shutil.rmtree(pulled)
        checkpoint_dir = store.pull(latest, pulled)
        state = load_checkpoint(checkpoint_dir, model=model, optimizer=optimizer, device=device)
        check_resume_compatible(state["pretrain_config"], asdict(cfg))
        if state["model_config"] != model_config.to_dict():
            raise ValueError("model config differs from the checkpoint")
        loader.load_state_dict(state["loader"])
        step, tokens_seen = state["step"], state["tokens_seen"]
        remote_metrics = pulled / "runs" / cfg.run_name / "metrics.jsonl"
        if remote_metrics.exists():
            shutil.copyfile(remote_metrics, metrics.path)
        metrics.truncate(step)
        shutil.rmtree(pulled)
        log(f"resumed {cfg.run_name} from {latest}: step {step}/{cfg.total_steps}, {tokens_seen:,} tokens")
    else:
        if metrics.path.exists():
            metrics.path.unlink()                         # leftovers from a run that never checkpointed
        counts = model.parameter_counts()
        log(f"new run {cfg.run_name}: {counts['total']:,} params ({trainable_parameters(model):,} trained), "
            f"{cfg.total_steps} steps x {cfg.tokens_per_step:,} tokens")

    if cfg.compile:
        torch._dynamo.config.recompile_limit = 64
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        forward = torch.compile(model)
    else:
        forward = model

    def save(final: bool) -> None:
        name = f"step-{step:06d}"
        local = run_dir / name
        if local.exists():
            shutil.rmtree(local)
        save_checkpoint(local, model=model, optimizer=optimizer, trainer_state={
            "step": step,
            "tokens_seen": tokens_seen,
            "loader": loader.state_dict(),
            "pretrain_config": asdict(cfg),
            "model_config": model_config.to_dict(),
            "metadata": dict(metadata or {}),
        })
        if store is not None:
            store.push(local, metrics=metrics.path, milestone=final)
            shutil.rmtree(local)
        log(f"checkpoint {name}{' (final)' if final else ''} saved")

    micro = cfg.micro_batch
    last_save = time.monotonic()
    while step < cfg.total_steps:
        lr = lr_at(step, max_lr=cfg.max_lr, warmup_steps=cfg.warmup_steps,
                   total_steps=cfg.total_steps, min_ratio=cfg.min_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = lr
        inputs, targets = loader.next_batch()
        batches = [(inputs[i:i + micro], targets[i:i + micro]) for i in range(0, cfg.batch_rows, micro)]
        started = time.perf_counter()
        stats = train_step(model, optimizer, batches, device=device, balance_coef=cfg.balance_coef,
                           grad_clip=cfg.grad_clip, forward=forward)
        seconds = time.perf_counter() - started
        step += 1
        tokens_seen += inputs.size
        metrics.append({"step": step, "lr": lr, "tokens": tokens_seen, "seconds": round(seconds, 4),
                        "tokens_per_second": round(inputs.size / seconds), **stats})
        if step == 1 or step % cfg.log_every == 0:
            log(f"step {step}/{cfg.total_steps}  loss {stats['loss']:.4f}  lr {lr:.2e}  "
                f"grad {stats['grad_norm']:.2f}  {inputs.size / seconds:,.0f} tok/s")

        final = step == cfg.total_steps
        if final or step % cfg.eval_every == 0:
            val = evaluate(model, val_sources, windows=cfg.eval_windows, batch_size=micro, device=device)
            mixed = sum(weights[name] * val[name] for name in val) / sum(weights[name] for name in val)
            row = {"step": step, "eval": val, "eval_mixed": mixed}
            message = f"eval @ {step}: mixed {mixed:.4f}  " + "  ".join(f"{k} {v:.4f}" for k, v in val.items())
            if extra_val_sources:
                row["eval_extra"] = evaluate(model, extra_val_sources, windows=cfg.eval_windows,
                                             batch_size=micro, device=device)
                message += "  | extra " + "  ".join(f"{k} {v:.4f}" for k, v in row["eval_extra"].items())
            metrics.append(row)
            log(message)

        due = (cfg.checkpoint_every_steps is not None and step % cfg.checkpoint_every_steps == 0) \
            or time.monotonic() - last_save >= cfg.checkpoint_minutes * 60
        if final or due:
            save(final)
            last_save = time.monotonic()
        if stop_after_steps is not None and step >= stop_after_steps and not final:
            log(f"stopping at step {step} without saving (simulated interruption)")
            break

    return {"step": step, "tokens_seen": tokens_seen, "finished": step == cfg.total_steps, "model": model}