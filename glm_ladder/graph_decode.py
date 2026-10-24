"""Decoding from a CUDA graph: one recorded decode step, replayed for every new token.

In eager PyTorch every operation costs the CPU 10-30 microseconds to launch, and one decode
step of these models is ~1,600 small operations, so the GPU waits for the CPU most of the
time (~40 ms per token measured, for a few microseconds of arithmetic). A CUDA graph records
all the kernels of one step once; `replay()` then launches them with a single call.

What a recorded step needs, and how it gets it:
  - fixed shapes:        a fixed batch size, and a StaticCache with full-length buffers
  - fixed memory:        inputs are copied into the same tensors before each replay, and
                         every cache update is in place
  - no CPU-GPU syncs:    the dense MoE path (no torch.where / bincount), the position as a
                         GPU tensor instead of a Python int
The prompt itself ("prefill") runs eagerly with an ordinary cache and is then copied into
the static buffers.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GraphDecoder:
    """Replays one decode step of `model` for a fixed batch size from a CUDA graph.

        decoder = GraphDecoder(model, batch=8)
        logits = decoder.prefill(prompt_batch)        # [8, vocab] for the token after the prompt
        logits = decoder.step(next_tokens)            # [8, vocab], one call per new token

    The returned logits tensor is overwritten by the next step: copy or use it first.
    """

    def __init__(self, model: nn.Module, batch: int, max_length: int | None = None) -> None:
        device = next(model.parameters()).device
        if device.type != "cuda":
            raise ValueError("CUDA graphs need the model on a CUDA device")
        if batch > model.layers[0].block.moe.dense_max_tokens:
            raise ValueError("batch too large for the dense MoE path, which the graph needs (no syncs)")
        self.model = model.eval()
        self.batch = batch
        self.max_length = max_length or model.config.max_sequence_length
        self.cache = model.new_static_cache(batch, self.max_length, device)
        self.tokens = torch.zeros(batch, 1, dtype=torch.long, device=device)
        self.position = 0

        # Warm up on a side stream first (lazy initialisation must not happen during capture),
        # then record one step. Both write into the cache; prefill overwrites everything used.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(2):
                self._step()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.graph):
            self.logits = self._step()

    def _step(self) -> torch.Tensor:
        return self.model(self.tokens, cache=self.cache).logits[:, -1]

    @torch.no_grad()
    def prefill(self, prompt: torch.Tensor) -> torch.Tensor:
        """Run the prompt [batch, P] eagerly, copy its cache into the static buffers, return [batch, vocab]."""
        if prompt.shape[0] != self.batch:
            raise ValueError(f"this decoder was recorded for batch {self.batch}")
        if prompt.shape[1] >= self.max_length:
            raise ValueError("prompt fills the whole static cache")
        cache = self.model.new_cache()
        logits = self.model(prompt.to(self.tokens.device), cache=cache).logits[:, -1]
        self.model.fill_static_cache(self.cache, cache)
        self.position = prompt.shape[1]          # mirrors cache.position on the CPU, without a sync
        return logits

    @torch.no_grad()
    def step(self, tokens: torch.Tensor) -> torch.Tensor:
        """Feed one token per sequence ([batch]) and return the next logits [batch, vocab]."""
        if self.position >= self.max_length:
            raise ValueError("static cache is full")
        self.tokens.copy_(tokens.view(self.batch, 1))
        self.graph.replay()
        self.position += 1
        return self.logits
