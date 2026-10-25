"""Batched decoding from a CUDA graph: one recorded decode step, replayed for every new token.

In eager PyTorch every operation costs the CPU 10-30 microseconds to launch, and one decode
step of these models is ~1,600 small operations, so the GPU waits for the CPU most of the
time (~40 ms per token measured, for a few microseconds of arithmetic). A CUDA graph records
all the kernels of one step once; `replay()` then launches them with a single call.

What a recorded step needs, and how it gets it:
  - fixed shapes:        a fixed batch size, and a StaticCache with full-length buffers
  - fixed memory:        inputs are copied into the same tensors before each replay, and
                         every cache update is in place
  - no CPU-GPU syncs:    the dense MoE path (no torch.where / bincount), positions as a
                         GPU tensor instead of Python ints
The prompts ("prefill") run eagerly with an ordinary cache, right-padded to one length, and
are then copied into the static buffers. Each row keeps its own position, so one batch can
hold many different prompts (RL: 32 tasks x 8 answers in one batch).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn as nn

from .tokenizer import PAD_ID


class GraphDecoder:
    """Decode steps of `model` for a fixed batch size, replayed from a CUDA graph.

        decoder = GraphDecoder(model, batch=8)
        logits = decoder.prefill(prompts)            # 8 token lists, any lengths -> [8, vocab]
        logits = decoder.step(next_tokens)           # [8] -> [8, vocab], one call per new token

    The returned logits tensor is overwritten by the next step: use or copy it first.
    graph=False runs the same static-cache step eagerly (CPU tests, debugging).
    """

    def __init__(self, model: nn.Module, batch: int, max_length: int | None = None,
                 graph: bool | None = None) -> None:
        device = next(model.parameters()).device
        self.graph_enabled = device.type == "cuda" if graph is None else graph
        if self.graph_enabled and device.type != "cuda":
            raise ValueError("CUDA graphs need the model on a CUDA device")
        if batch > model.layers[0].block.moe.dense_max_tokens:
            raise ValueError("batch too large for the dense MoE path, which the graph needs (no syncs)")
        self.model = model.eval()
        self.batch = batch
        self.device = device
        self.max_length = max_length or model.config.max_sequence_length
        self.cache = model.new_static_cache(batch, self.max_length, device)
        self.tokens = torch.zeros(batch, 1, dtype=torch.long, device=device)
        self.furthest = 0                             # largest position of any row, tracked on the CPU
        if not self.graph_enabled:
            return
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
    def prefill(self, prompts: Sequence[Sequence[int]]) -> torch.Tensor:
        """Run `batch` prompts (any lengths) eagerly, copy the cache into the static buffers.

        Returns [batch, vocab]: the logits after each prompt's own last token.
        """
        if len(prompts) != self.batch:
            raise ValueError(f"this decoder was built for batch {self.batch}, got {len(prompts)} prompts")
        lengths = [len(prompt) for prompt in prompts]
        if min(lengths) < 1 or max(lengths) >= self.max_length:
            raise ValueError("every prompt needs 1 .. max_length - 1 tokens")
        width = max(lengths)
        ids = torch.full((self.batch, width), PAD_ID, dtype=torch.long)
        for row, prompt in enumerate(prompts):
            ids[row, :len(prompt)] = torch.as_tensor(prompt)
        ids = ids.to(self.device)
        length_tensor = torch.as_tensor(lengths, device=self.device)
        cache = self.model.new_cache()
        cache.valid = torch.arange(width, device=self.device)[None, :] < length_tensor[:, None]
        logits = self.model(ids, cache=cache).logits
        self.model.fill_static_cache(self.cache, cache, length_tensor)
        self.furthest = width
        return logits[torch.arange(self.batch, device=self.device), length_tensor - 1]

    @torch.no_grad()
    def step(self, tokens: torch.Tensor) -> torch.Tensor:
        """Feed one token per sequence ([batch]) and return the next logits [batch, vocab]."""
        if self.furthest >= self.max_length:
            raise ValueError("static cache is full")
        self.tokens.copy_(tokens.view(self.batch, 1))
        if self.graph_enabled:
            self.graph.replay()
            logits = self.logits
        else:
            logits = self._step()
        self.furthest += 1
        return logits


def pick_tokens(logits: torch.Tensor, *, temperature: float, top_k: int | None,
                generator: torch.Generator) -> torch.Tensor:
    """Greedy (temperature 0) or temperature + top-k sampling, on the logits' device. [B, V] -> [B]."""
    if temperature == 0:
        return logits.argmax(dim=-1)
    logits = logits.float() / temperature
    if top_k:
        kth = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    return torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator).squeeze(1)


@torch.no_grad()
def generate_batch(decoder: GraphDecoder, prompts: Sequence[Sequence[int]], *, max_new_tokens: int,
                   temperature: float, top_k: int | None, seed: int,
                   stop: Callable[[list[int]], bool]) -> list[list[int]]:
    """One continuation per prompt (rows may repeat a prompt), all decoded as one batch.

    Each row stops when stop(its new ids) is True; the batch stops when every row has.
    Only the chosen token ids go to the CPU, once per step.
    """
    if max(len(prompt) for prompt in prompts) + max_new_tokens > decoder.max_length:
        raise ValueError("prompt + new tokens longer than the decoder's buffers")
    generator = torch.Generator(device=decoder.device).manual_seed(seed)
    outputs: list[list[int]] = [[] for _ in prompts]
    done = [False] * len(prompts)
    logits = decoder.prefill(prompts)
    for step in range(max_new_tokens):
        tokens = pick_tokens(logits, temperature=temperature, top_k=top_k, generator=generator)
        for row, token in enumerate(tokens.tolist()):
            if not done[row]:
                outputs[row].append(token)
                done[row] = stop(outputs[row])
        if all(done) or step == max_new_tokens - 1:
            break
        logits = decoder.step(tokens)
    return outputs
