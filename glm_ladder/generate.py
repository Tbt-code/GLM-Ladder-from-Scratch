"""Autoregressive text generation: temperature, top-k sampling, stop at <|endoftext|>.

The model has no KV cache, so every new token re-runs the model over the whole
context (at most max_sequence_length tokens). That is slow but simple, and fast
enough for a few hundred tokens from a 50-100M model, even on a CPU.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .tokenizer import EOS_ID


def sample_next(logits: torch.Tensor, *, temperature: float, top_k: int | None,
                generator: torch.Generator) -> int:
    """Pick the next token id from 1-D float `logits` over the vocabulary.

    temperature == 0 means greedy (argmax). Otherwise logits are divided by the
    temperature; if top_k is set, only the top_k largest logits stay possible.
    """
    #   1. temperature == 0 -> return the index of the largest logit (int)
    if temperature == 0:
        return int(torch.argmax(logits).item())

    # 2. logits = logits / temperature
    logits = logits / temperature

    #   3. if top_k: find the k-th largest value (torch.topk) and set every logit
    #  below it to -inf, so softmax gives those tokens probability exactly 0
    if top_k is not None and top_k > 0:
        k = min(top_k,logits.size(-1))
        values,_ = torch.topk(logits,k)
        threshold = values[-1]
        logits[logits < threshold] = float("-inf")

    #   4. probs = softmax(logits); draw one index with
    #  torch.multinomial(probs, 1, generator=generator) and return it as int
    probs = torch.softmax(logits,dim=-1)
    next_token = torch.multinomial(probs,1,generator=generator)
    return int(next_token.item())


@torch.no_grad()
def generate(
    model: nn.Module, prompt_ids: list[int], *, max_new_tokens: int,
    temperature: float = 0.8, top_k: int | None = 50, seed: int = 0, stop_id: int | None = EOS_ID,
) -> list[int]:
    """Return only the newly generated ids (the stop token included, if it was produced)."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    window = model.config.max_sequence_length
    generator = torch.Generator().manual_seed(seed)      # CPU generator: same samples on any device
    ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        context = torch.tensor([ids[-window:]], dtype=torch.long, device=device)
        logits = model(context).logits[0, -1].float().cpu()
        token = sample_next(logits, temperature=temperature, top_k=top_k, generator=generator)
        ids.append(token)
        if token == stop_id:
            break
    model.train(was_training)
    return ids[len(prompt_ids):]
