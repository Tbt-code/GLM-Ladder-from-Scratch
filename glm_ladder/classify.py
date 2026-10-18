"""Review rating (1-5 stars) as a chat task: prompts, scoring without generation, metrics.

The model is asked in the SFT chat format and should answer with one digit:

    <|user|>Rate the following review from 1 to 5 stars. ...<|end|><|assistant|>4<|end|>

To evaluate, nothing is generated. One forward pass over the prompt gives the next-token
distribution after <|assistant|>; the prediction is the most likely of the five digit
tokens. This is exact (no sampling), fast (one pass per review), and also says how much
probability the model puts on the digits at all (`label_mass`): a zero-shot model that
wants to write "The review is positive" puts little mass there.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn

from .chat import prompt_ids
from .tokenizer import PAD_ID

STARS = (1, 2, 3, 4, 5)
LANGUAGES = ("tr", "en")
INSTRUCTIONS = {
    "tr": "Aşağıdaki yorumu 1 ile 5 yıldız arasında puanla. Sadece rakamı yaz.\n\nYorum: {text}",
    "en": "Rate the following review from 1 to 5 stars. Answer with the digit only.\n\nReview: {text}",
}


def conversation(text: str, language: str, stars: int | None = None) -> list[dict[str, str]]:
    """The user request, plus the answer when `stars` is given (training data)."""
    messages = [{"role": "user", "content": INSTRUCTIONS[language].format(text=text.strip())}]
    if stars is not None:
        if stars not in STARS:
            raise ValueError(f"stars must be in {STARS}, got {stars}")
        messages.append({"role": "assistant", "content": str(stars)})
    return messages


def label_token_ids(tokenizer) -> list[int]:
    """Token ids of "1" ... "5"; each must be a single token, as in the training answers."""
    ids = []
    for stars in STARS:
        encoded = tokenizer.encode(str(stars)).ids
        if len(encoded) != 1:
            raise ValueError(f"{stars!r} is not a single token: {encoded}")
        ids.append(encoded[0])
    return ids


def label_log_probs(logits: torch.Tensor, label_ids: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Logits [N, vocab] at the answer position -> (log-probs over the labels [N, L], label mass [N]).

    label mass: total probability (over the full vocabulary) of the L label tokens, in [0, 1].
    log-probs: renormalised among the labels only, so each row's exp() sums to 1.
    """
    log_probs = torch.log_softmax(logits.float(),dim=-1)
    selected = log_probs[:,label_ids]
    mass = selected.exp().sum(-1)
    normalized_log_probs = selected - torch.logsumexp(selected, dim=-1, keepdim=True)
    return normalized_log_probs,mass


@torch.no_grad()
def score(
    model: nn.Module, prompts: Sequence[Sequence[int]], label_ids: Sequence[int], *,
    batch_size: int = 64, device: torch.device = torch.device("cpu"),
) -> tuple[np.ndarray, np.ndarray]:
    """(label log-probs [N, L], label mass [N]) for every prompt, in the given order.

    Prompts are sorted by length and right-padded, so each batch wastes little compute.
    Padding sits after the last real token, and the model is causal (attention, linear
    attention, and per-token MoE), so padding cannot change the logits read at that token.
    """
    model.eval()
    order = sorted(range(len(prompts)), key=lambda index: len(prompts[index]))
    log_probs = np.zeros((len(prompts), len(label_ids)), dtype=np.float32)
    mass = np.zeros(len(prompts), dtype=np.float32)
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        lengths = [len(prompts[index]) for index in chunk]
        batch = torch.full((len(chunk), max(lengths)), PAD_ID, dtype=torch.long)
        for row, index in enumerate(chunk):
            batch[row, :lengths[row]] = torch.as_tensor(prompts[index])
        batch = batch.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(batch).logits
        last = torch.as_tensor(lengths, device=device) - 1
        answer_logits = logits[torch.arange(len(chunk), device=device), last].float()
        chunk_log_probs, chunk_mass = label_log_probs(answer_logits, label_ids)
        log_probs[chunk] = chunk_log_probs.cpu().numpy()
        mass[chunk] = chunk_mass.cpu().numpy()
    return log_probs, mass


def review_prompts(tokenizer, texts: Sequence[str], language: str) -> list[list[int]]:
    return [prompt_ids(tokenizer, conversation(text, language)) for text in texts]


def metrics(predicted: Sequence[int], true: Sequence[int]) -> dict:
    """Accuracy, macro-F1 (each star counts equally, whatever its frequency), mean absolute
    error in stars, and the confusion matrix (rows: true stars, columns: predicted)."""
    predicted, true = np.asarray(predicted), np.asarray(true)
    if predicted.shape != true.shape or predicted.size == 0:
        raise ValueError("predicted and true must be non-empty and the same length")
    confusion = np.zeros((len(STARS), len(STARS)), dtype=np.int64)
    for t, p in zip(true, predicted):
        confusion[t - 1, p - 1] += 1
    f1s = []
    for k in range(len(STARS)):
        tp = confusion[k, k]
        precision = tp / confusion[:, k].sum() if confusion[:, k].sum() else 0.0
        recall = tp / confusion[k, :].sum() if confusion[k, :].sum() else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "n": int(true.size),
        "accuracy": float((predicted == true).mean()),
        "macro_f1": float(np.mean(f1s)),
        "mae_stars": float(np.abs(predicted - true).mean()),
        "confusion": confusion.tolist(),
    }


def evaluate_reviews(model: nn.Module, tokenizer, texts: Sequence[str], stars: Sequence[int], language: str, *,
                     batch_size: int = 64, device: torch.device = torch.device("cpu")) -> dict:
    """Score every review and return metrics plus the mean label mass."""
    log_probs, mass = score(model, review_prompts(tokenizer, texts, language), label_token_ids(tokenizer),
                            batch_size=batch_size, device=device)
    predicted = log_probs.argmax(axis=1) + STARS[0]
    return {**metrics(predicted, stars), "label_mass": float(mass.mean())}


def majority_baseline(train_stars: Sequence[int], test_stars: Sequence[int]) -> dict:
    """Always predict the most frequent training label: the floor any model must beat."""
    most_common = int(np.bincount(np.asarray(train_stars), minlength=len(STARS) + 1).argmax())
    return {**metrics([most_common] * len(test_stars), test_stars), "predicts": most_common}
