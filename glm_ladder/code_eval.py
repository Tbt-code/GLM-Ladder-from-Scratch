"""Executable code evaluation: generate a Python function, run unit tests on it, report pass@k.

Two ways to ask a model for a function:
  completion  `def name(args):\n    \"\"\"doc\"\"\"` and the model writes the body (works for
              pretrained and fine-tuned models: it measures raw coding ability)
  chat        a user message "Write a Python function `name(args)`. doc" in the chat template
              (fine-tuned models only: it also measures instruction following)

Generated code runs in a separate Python process with a timeout, in an empty temporary
directory, after a simple deny-list check. That is enough for a small model's output on a
personal machine, but it is NOT a security sandbox; RL at scale should run in a container.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Code containing any of these is not executed and counts as a failure.
UNSAFE_PATTERNS = (
    "import os", "from os", "import sys", "from sys", "import subprocess", "from subprocess",
    "import shutil", "from shutil", "import socket", "from socket", "import pathlib", "from pathlib",
    "__import__", "open(", "eval(", "exec(", "compile(", "input(", "globals(", "breakpoint(",
    # RL runs thousands of generated programs: also no raising of resource limits, no foreign
    # function calls, no network, no new processes or threads, no dynamic imports.
    "import resource", "from resource", "import ctypes", "from ctypes", "import importlib", "from importlib",
    "import multiprocessing", "from multiprocessing", "import threading", "from threading",
    "import signal", "from signal", "import urllib", "from urllib", "import http", "from http",
    "import requests", "import asyncio", "from asyncio", "import pickle", "from pickle",
)

# On Linux (Colab) every test program first limits itself: 1 GB of memory, a few CPU seconds,
# no files larger than 1 MB. The limits are set inside the child, before the model's code runs.
LIMITS_PRELUDE = (
    "import resource as _r\n"
    "_r.setrlimit(_r.RLIMIT_AS, (1 << 30, 1 << 30))\n"
    "_r.setrlimit(_r.RLIMIT_CPU, ({cpu}, {cpu}))\n"
    "_r.setrlimit(_r.RLIMIT_FSIZE, (1 << 20, 1 << 20))\n"
    "del _r\n"
)


@dataclass(frozen=True)
class CodeTask:
    name: str
    args: str
    doc: str
    tests: tuple[str, ...]
    reference: str

    def completion_prompt(self) -> str:
        # No trailing newline: "\n" plus indentation is one BPE token in training text, so the
        # model should produce it (see the token-boundary note in the README).
        return f'def {self.name}({self.args}):\n    """{self.doc}"""'

    def chat_prompt(self) -> str:
        return f"Write a Python function `{self.name}({self.args})`. {self.doc}"


def load_tasks(path: str | Path) -> list[CodeTask]:
    tasks = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            tasks.append(CodeTask(row["name"], row["args"], row["doc"], tuple(row["tests"]), row["reference"]))
    return tasks


def is_safe(code: str) -> bool:
    return not any(pattern in code for pattern in UNSAFE_PATTERNS)


def run_tests(code: str, tests: Sequence[str], timeout: float = 5.0) -> bool:
    """True if `code` followed by the assert statements exits cleanly within `timeout` seconds."""
    if not is_safe(code):
        return False
    program = code + "\n\n" + "\n".join(tests) + "\n"
    environment = None
    if os.name == "posix":
        program = LIMITS_PRELUDE.format(cpu=int(math.ceil(timeout)) + 1) + program
        environment = {"PATH": "/usr/bin:/bin"}        # nothing from the parent's environment
    with tempfile.TemporaryDirectory() as workdir:
        try:
            result = subprocess.run([sys.executable, "-I", "-c", program], cwd=workdir, capture_output=True,
                                    stdin=subprocess.DEVNULL, timeout=timeout, env=environment)
        except subprocess.TimeoutExpired:
            return False
    return result.returncode == 0


def run_tests_many(jobs: Sequence[tuple[str | None, Sequence[str]]], *, timeout: float = 5.0,
                   workers: int | None = None) -> list[bool]:
    """run_tests for many (code, tests) pairs at once, in order; code None counts as a failure.

    Each program is its own subprocess; threads only wait for them, so they run on all CPU cores.
    """
    from concurrent.futures import ThreadPoolExecutor

    def one(job):
        code, tests = job
        return code is not None and run_tests(code, tests, timeout=timeout)

    with ThreadPoolExecutor(max_workers=workers or os.cpu_count() or 4) as pool:
        return list(pool.map(one, jobs))


def extract_from_completion(prompt: str, continuation: str) -> str:
    """The function = prompt + continuation, cut at the first non-indented line after the header."""
    lines = (prompt + continuation).split("\n")
    kept = [lines[0]]
    for line in lines[1:]:
        if line.strip() and not line[0].isspace():
            break
        kept.append(line)
    return "\n".join(kept).rstrip() + "\n"


def extract_from_chat(answer: str, name: str) -> str | None:
    """The code that defines `name`: a fenced block if there is one, else text from `def name(`."""
    for block in re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", answer, flags=re.S):
        if f"def {name}(" in block:
            return block
    start = answer.find(f"def {name}(")
    if start < 0:
        return None
    return answer[start:].split("```")[0]           # also handles an answer cut before its closing fence


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k (Chen et al., 2021): the chance that at least one of k samples, drawn
    without replacement from n generated samples of which c pass, is correct."""
    if n - c < k:
        return 1.0

    return 1.0 - math.comb(n-c,k) / math.comb(n,k)


@torch.no_grad()
def sample_many(
    model: nn.Module, prompt_ids: list[int], *, n: int, max_new_tokens: int, temperature: float,
    top_k: int | None, seed: int, stop: Callable[[list[int]], bool], decoder=None,
) -> list[list[int]]:
    """n continuations of one prompt, generated as one batch; each stops when stop(its new ids) is True.

    Models with a generation cache (GLMLadderModel.new_cache) process the prompt once and then
    one token per step, as long as prompt + new tokens fit in the context; otherwise every step
    re-runs the whole (sliding) window. With `decoder` (a GraphDecoder recorded for batch n),
    each new token is one CUDA-graph replay. All paths draw the same random numbers.

    Sampling happens on the model's device, and only the n chosen token ids go to the CPU:
    copying [n, 16384] logits and sampling them on the CPU took ~70 ms per step at n = 128,
    far longer than the model step itself. The random numbers therefore come from a generator
    on that device (same seed -> same samples on the same device, but GPU and CPU differ).
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    window = model.config.max_sequence_length
    generator = torch.Generator(device=device).manual_seed(seed)
    context = torch.tensor([prompt_ids] * n, dtype=torch.long, device=device)
    outputs: list[list[int]] = [[] for _ in range(n)]
    done = [False] * n
    if decoder is not None:                  # CUDA-graph decoding: see glm_ladder/graph_decode.py
        from .graph_decode import generate_batch

        if decoder.batch != n:
            raise ValueError("decoder was built for a different batch size")
        outputs = generate_batch(decoder, [prompt_ids] * n, max_new_tokens=max_new_tokens,
                                 temperature=temperature, top_k=top_k, seed=seed, stop=stop)
        model.train(was_training)
        return outputs
    cache = model.new_cache() if hasattr(model, "new_cache") and len(prompt_ids) + max_new_tokens <= window else None
    step_input = context
    for _ in range(max_new_tokens):
        if cache is not None:
            logits = model(step_input, cache=cache).logits[:, -1].float()
        else:
            logits = model(context[:, -window:]).logits[:, -1].float()
        if temperature == 0:
            tokens = logits.argmax(dim=-1)
        else:
            logits = logits / temperature
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            tokens = torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator).squeeze(1)
        for i, token in enumerate(tokens.tolist()):           # the one GPU -> CPU copy per step: n ids
            if not done[i]:
                outputs[i].append(token)
                done[i] = stop(outputs[i])
        if all(done):
            break
        context = torch.cat([context, tokens[:, None]], dim=1)
        step_input = tokens[:, None]
    model.train(was_training)
    return outputs


@torch.no_grad()
def answer_loss(model: nn.Module, prompt_ids: list[int], answer_ids: list[int]) -> float:
    """Mean cross-entropy (nats per token) the model assigns to answer_ids right after prompt_ids.

    Used with the reference solutions: a continuous measure of coding ability that still moves
    when every generated sample fails (pass@k has a floor at zero).
    """
    ids = prompt_ids + answer_ids
    if len(ids) > model.config.max_sequence_length:
        raise ValueError("prompt + answer longer than the model's context")
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    logits = model(torch.tensor([ids], dtype=torch.long, device=device)).logits[0].float()
    start = len(prompt_ids)
    # logits at position t predict token t + 1, so the answer is predicted from start - 1 to the end.
    loss = F.cross_entropy(logits[start - 1:-1], torch.tensor(answer_ids, dtype=torch.long, device=device))
    model.train(was_training)
    return loss.item()

def summarize(passes: Sequence[Sequence[bool]], ks: Sequence[int]) -> dict[str, float]:
    """Mean pass@k over tasks, from each task's list of per-sample results."""
    return {f"pass@{k}": sum(pass_at_k(len(p), sum(p), k) for p in passes) / len(passes)
            for k in ks if all(len(p) >= k for p in passes)}