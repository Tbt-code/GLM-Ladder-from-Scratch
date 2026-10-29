"""Reinforcement learning with executable rewards: RLOO with a KL penalty.

One RL step:
  1. rollout   pick `prompts_per_step` training tasks; the current model writes `group`
               answers to each (one batch, decoded from a CUDA graph)
  2. reward    1 if the answer's code passes all the task's tests, else 0
  3. advantage RLOO: each answer's reward minus the mean reward of the other answers to the
               same task. Better than its siblings -> positive, worse -> negative; a task
               where all answers score the same gives no signal at all
  4. update    one AdamW step on  -advantage * log p(answer)  plus a KL penalty that keeps
               the model close to where it started (the reference model)
On-policy: every update uses answers the current model just wrote, once, so the importance
ratio is 1 and no PPO clipping is needed.
"""
from __future__ import annotations

import copy
import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .chat import END_TOKEN, prompt_ids, special_id
from .checkpoint import load_checkpoint, save_checkpoint
from .code_eval import CodeTask, extract_from_chat, pass_at_k, run_tests_many
from .graph_decode import GraphDecoder, generate_batch
from .tokenizer import EOS_ID, PAD_ID


@dataclass(frozen=True)
class RLConfig:
    run_name: str
    steps: int = 200
    prompts_per_step: int = 16
    group: int = 16                    # answers per task; prompts_per_step * group rows in one decode batch
                                       # (16 answers make a hard task's rare success, and so a signal, likelier)
    max_new_tokens: int = 192
    temperature: float = 1.0           # sample from the model's own distribution (no top-k): the
                                       # gradient below is for exactly that distribution
    lr: float = 5e-6                   # chosen with a 20-step trial run (README)
    warmup_steps: int = 10
    kl_coef: float = 0.02              # 0: no KL penalty (the ablation)
    balance_coef: float = 0.01         # MoE load-balancing loss, as in every earlier phase
    grad_clip: float = 1.0
    micro_batch: int = 32
    seed: int = 1234
    log_every: int = 1


# ----------------------------------------------------------------- the two pieces you write

def rloo_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Leave-one-out advantages. rewards: [tasks, group] -> [tasks, group].

    A_i = r_i - mean of the other (group - 1) rewards of the same task.
    """
    group = rewards.shape[1]
    other_sum = rewards.sum(dim=1,keepdim=True) - rewards
    other_mean = other_sum / (group - 1)
    return rewards - other_mean



def policy_loss(logp: torch.Tensor, ref_logp: torch.Tensor, mask: torch.Tensor, advantages: torch.Tensor,
                kl_coef: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(loss, mean KL per answer token) for one micro-batch.

    logp, ref_logp: [B, T] log-probabilities of the tokens actually generated, under the model
    being trained and under the frozen reference model. mask: [B, T] True on answer tokens
    (prompt and padding are False). advantages: [B], one per answer.

        policy term:  - sum over answer tokens of  advantage * logp
        KL term:      kl_coef * sum over answer tokens of  exp(d) - d - 1,  where d = ref_logp - logp
                      (the "k3" estimate of KL(model || reference): never negative, 0 when equal)
    Both sums are divided by the number of answer tokens in the micro-batch: every answer
    token in the batch counts the same (the trainer weights micro-batches by their token count).
    """
    tokens = mask.sum().clamp_min(1)
    pg = -(advantages[:,None] * logp * mask).sum() / tokens
    d = ref_logp - logp
    kl = ((torch.exp(d) - d - 1) * mask).sum() / tokens

    return pg + kl_coef * kl,kl.detach()


# ----------------------------------------------------------------- batches and log-probabilities

def pad_batch(sequences: Sequence[Sequence[int]], prompt_lengths: Sequence[int]
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(inputs, targets, mask) [B, T]: next-token pairs, right-padded; mask True where the
    target is an answer token (position >= prompt length - 1 of the inputs)."""
    width = max(len(seq) for seq in sequences) - 1
    inputs = torch.full((len(sequences), width), PAD_ID, dtype=torch.long)
    targets = torch.full((len(sequences), width), PAD_ID, dtype=torch.long)
    mask = torch.zeros(len(sequences), width, dtype=torch.bool)
    for row, (seq, prompt) in enumerate(zip(sequences, prompt_lengths)):
        seq = torch.as_tensor(seq)
        inputs[row, :len(seq) - 1] = seq[:-1]
        targets[row, :len(seq) - 1] = seq[1:]
        mask[row, prompt - 1:len(seq) - 1] = True
    return inputs, targets, mask


def token_logprobs(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(log p(target) [B, T] in float32, MoE balance loss) from one forward pass."""
    device = inputs.device
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        output = model(inputs)
    logp = torch.log_softmax(output.logits.float(), dim=-1).gather(-1, targets[..., None]).squeeze(-1)
    return logp, output.balance_loss


# ----------------------------------------------------------------- rewards and evaluation

def code_rewards(tokenizer, tasks: Sequence[CodeTask], answers: Sequence[Sequence[int]],
                 workers: int | None = None, timeout: float = 2.0) -> list[float]:
    """1.0 if the answer's code passes all of its task's tests, else 0.0 (one per answer).

    timeout: these functions run in milliseconds; 2 s only cuts off endless loops, which with
    the default 5 s made some RL steps 4 s longer.
    """
    end_id = special_id(tokenizer, END_TOKEN)
    jobs = []
    for task, ids in zip(tasks, answers):
        text = tokenizer.decode([t for t in ids if t not in (end_id, EOS_ID)])
        jobs.append((extract_from_chat(text, task.name), task.tests))
    return [float(ok) for ok in run_tests_many(jobs, workers=workers, timeout=timeout)]


def chat_prompts(tokenizer, tasks: Sequence[CodeTask]) -> list[list[int]]:
    return [prompt_ids(tokenizer, [{"role": "user", "content": task.chat_prompt()}]) for task in tasks]


def stop_at_end(tokenizer) -> Callable[[list[int]], bool]:
    end_id = special_id(tokenizer, END_TOKEN)
    return lambda out: out[-1] in (end_id, EOS_ID)


class Sampler:
    """Generates for any list of prompts, `batch` rows at a time, reusing one recorded decoder."""

    def __init__(self, model: nn.Module, batch: int, max_length: int) -> None:
        self.model, self.batch = model, batch
        self.decoder = GraphDecoder(model, batch=batch, max_length=max_length)

    @torch.no_grad()
    def __call__(self, prompts: Sequence[Sequence[int]], *, max_new_tokens: int, temperature: float,
                 seed: int, stop: Callable[[list[int]], bool]) -> list[list[int]]:
        was_training = self.model.training
        self.model.eval()                         # the decoder was recorded in eval mode (dense MoE)
        outputs: list[list[int]] = []
        for start in range(0, len(prompts), self.batch):
            chunk = list(prompts[start:start + self.batch])
            real = len(chunk)
            chunk += [chunk[-1]] * (self.batch - real)          # fill the last batch; extra rows are dropped
            outputs += generate_batch(self.decoder, chunk, max_new_tokens=max_new_tokens, temperature=temperature,
                                      top_k=None, seed=seed + start, stop=stop)[:real]
        self.model.train(was_training)
        return outputs


def pass_counts(sampler: Sampler, tokenizer, tasks: Sequence[CodeTask], *, n: int, temperature: float,
                max_new_tokens: int = 192, seed: int = 0) -> list[int]:
    """How many of n answers pass, per task (temperature 0 with n = 1 gives greedy pass/fail)."""
    rows = [task for task in tasks for _ in range(n)]
    answers = sampler(chat_prompts(tokenizer, rows), max_new_tokens=max_new_tokens, temperature=temperature,
                      seed=seed, stop=stop_at_end(tokenizer))
    rewards = code_rewards(tokenizer, rows, answers)
    return [int(sum(rewards[i * n:(i + 1) * n])) for i in range(len(tasks))]


def summarize_counts(counts: Sequence[int], n: int, ks: Sequence[int] = (1, 5, 10)) -> dict[str, float]:
    return {f"pass@{k}": float(np.mean([pass_at_k(n, c, k) for c in counts])) for k in ks if k <= n}


def paired_bootstrap(before: Sequence[float], after: Sequence[float], iterations: int = 10_000,
                     seed: int = 0) -> dict[str, float]:
    """Mean per-task difference (after - before) with a 95% interval from resampling tasks."""
    difference = np.asarray(after, dtype=float) - np.asarray(before, dtype=float)
    rng = np.random.default_rng(seed)
    means = difference[rng.integers(0, len(difference), size=(iterations, len(difference)))].mean(axis=1)
    return {"mean": float(difference.mean()), "low": float(np.quantile(means, 0.025)),
            "high": float(np.quantile(means, 0.975)), "tasks": len(difference)}


# ----------------------------------------------------------------- the trainer

class RLTrainer:
    """RLOO on code tasks. `reward_fn(tasks, answers) -> rewards` can be swapped for tests."""

    def __init__(self, cfg: RLConfig, model: nn.Module, tokenizer, tasks: Sequence[CodeTask], *,
                 device: torch.device, log: Callable[[str], None] = print,
                 reward_fn: Callable[[Sequence[CodeTask], Sequence[Sequence[int]]], list[float]] | None = None,
                 prompts_fn: Callable[[Sequence[CodeTask]], list[list[int]]] | None = None,
                 stop: Callable[[list[int]], bool] | None = None, reference: nn.Module | None = None) -> None:
        """reference: the frozen starting model for the KL penalty. When resuming, pass the
        original starting model here (the model being trained has already moved)."""
        if cfg.group < 2:
            raise ValueError("RLOO needs at least two answers per task")
        self.cfg, self.model, self.tokenizer, self.tasks = cfg, model, tokenizer, list(tasks)
        self.device, self.log = device, log
        self.reward_fn = reward_fn or (lambda tasks, answers: code_rewards(tokenizer, tasks, answers))
        self.prompts_fn = prompts_fn or (lambda tasks: chat_prompts(tokenizer, tasks))
        self.stop = stop or stop_at_end(tokenizer)
        torch.manual_seed(cfg.seed)
        self.rng = random.Random(cfg.seed)
        self.reference = None
        if cfg.kl_coef > 0:
            self.reference = (reference if reference is not None else copy.deepcopy(model)).eval()
        if self.reference is not None:
            for parameter in self.reference.parameters():
                parameter.requires_grad_(False)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0)
        longest = max(len(p) for p in self.prompts_fn(self.tasks))
        self.sampler = Sampler(model, cfg.prompts_per_step * cfg.group, longest + cfg.max_new_tokens + 1)
        self.order: list[int] = []
        self.history: list[dict] = []

    def next_tasks(self) -> list[CodeTask]:
        """Tasks without replacement; reshuffled every pass over the training set."""
        picked = []
        while len(picked) < self.cfg.prompts_per_step:
            if not self.order:
                self.order = list(range(len(self.tasks)))
                self.rng.shuffle(self.order)
            picked.append(self.tasks[self.order.pop()])
        return picked

    def step(self, step: int) -> dict:
        cfg = self.cfg
        started = time.perf_counter()
        tasks = self.next_tasks()
        rows = [task for task in tasks for _ in range(cfg.group)]
        prompts = self.prompts_fn(rows)
        answers = self.sampler(prompts, max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature,
                               seed=cfg.seed * 1_000_003 + step, stop=self.stop)
        generated = time.perf_counter()
        rewards = torch.tensor(self.reward_fn(rows, answers), dtype=torch.float32).view(len(tasks), cfg.group)
        advantages = rloo_advantages(rewards).view(-1)
        rewarded = time.perf_counter()

        lr = cfg.lr * min(1.0, (step + 1) / cfg.warmup_steps)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        sequences = [list(p) + list(a) for p, a in zip(prompts, answers)]
        lengths = [len(p) for p in prompts]
        total_tokens = sum(len(a) for a in answers)
        kl_sum, chunks = 0.0, math.ceil(len(sequences) / cfg.micro_batch)
        for start in range(0, len(sequences), cfg.micro_batch):
            inputs, targets, mask = (t.to(self.device) for t in
                                     pad_batch(sequences[start:start + cfg.micro_batch], lengths[start:start + cfg.micro_batch]))
            logp, balance = token_logprobs(self.model, inputs, targets)
            if self.reference is not None:
                with torch.no_grad():
                    ref_logp, _ = token_logprobs(self.reference, inputs, targets)
            else:
                ref_logp = logp.detach()
            adv = advantages[start:start + cfg.micro_batch].to(self.device)
            loss, kl = policy_loss(logp, ref_logp, mask, adv, cfg.kl_coef)
            # Weight each micro-batch by its share of answer tokens, so the step equals one big batch.
            weight = mask.sum().item() / max(total_tokens, 1)
            (loss * weight + cfg.balance_coef * balance / chunks).backward()
            kl_sum += kl.item() * weight
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient norm at step {step}")
        self.optimizer.step()
        finished = time.perf_counter()

        row = {
            "step": step + 1, "lr": lr, "reward": float(rewards.mean()),
            "tasks_with_signal": float(((rewards.sum(1) > 0) & (rewards.sum(1) < cfg.group)).float().mean()),
            "answer_tokens": total_tokens / len(answers), "kl": kl_sum, "grad_norm": float(grad_norm),
            "seconds_generate": round(generated - started, 2), "seconds_reward": round(rewarded - generated, 2),
            "seconds_update": round(finished - rewarded, 2),
        }
        self.history.append(row)
        if step == 0 or (step + 1) % cfg.log_every == 0:
            self.log(f"step {row['step']}/{cfg.steps}  reward {row['reward']:.3f}  signal {row['tasks_with_signal']:.2f}  "
                     f"kl {row['kl']:.4f}  len {row['answer_tokens']:.0f}  grad {row['grad_norm']:.2f}  "
                     f"time {row['seconds_generate']:.1f}+{row['seconds_reward']:.1f}+{row['seconds_update']:.1f} s")
        return row

    def run(self, on_checkpoint: Callable[[int], None] | None = None, checkpoint_every: int = 25) -> list[dict]:
        """Train until cfg.steps, starting after the steps already in history (a resumed run).

        on_checkpoint(step) is called every `checkpoint_every` steps and after the last one.
        """
        for step in range(len(self.history), self.cfg.steps):
            self.step(step)
            done = step + 1
            if on_checkpoint is not None and (done % checkpoint_every == 0 or done == self.cfg.steps):
                on_checkpoint(done)
        return self.history

    # ------------------------------------------------------------- resuming

    def state_dict(self) -> dict:
        """Everything besides weights, optimizer and torch RNG that a resumed run needs. JSON only."""

        return {"rl_config":asdict(self.cfg),
                "history":list(self.history),
                "order":self.order,
                "python_rng":self.rng.getstate()}

    def load_state_dict(self, state: dict) -> None:
        if state["rl_config"] != asdict(self.cfg):
            raise ValueError("Configuration mismatch when resuming training.")
        self.history = list(state["history"])
        self.order = list(state["order"])

        rng_state = state["python_rng"]
        version = rng_state[0]
        internal = tuple(rng_state[1])
        gauss = rng_state[2]
        self.rng.setstate((version, internal, gauss))


    def save(self, directory, extra: dict | None = None) -> None:
        """Weights, AdamW state and torch RNG (save_checkpoint) plus state_dict() and `extra`."""
        save_checkpoint(directory, model=self.model, optimizer=self.optimizer,
                        trainer_state={**(extra or {}), **self.state_dict(), "step": len(self.history)})

    def restore(self, directory) -> dict:
        """Load a checkpoint written by save() into this trainer; returns its trainer state."""
        state = load_checkpoint(directory, model=self.model, optimizer=self.optimizer, device=self.device)
        self.load_state_dict(state)
        return state

    def config_dict(self) -> dict:
        return asdict(self.cfg)
