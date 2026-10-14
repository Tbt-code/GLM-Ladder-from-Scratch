"""Byte-level BPE tokenizer shared by every BPE rung (Hugging Face `tokenizers`).

Same algorithm as toy_bpe.py, plus GPT-2-style pre-tokenization and a fast Rust
implementation. Token ids must fit in uint16, so the vocabulary is capped at 65,536.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

VOCAB_SIZE = 16_384
SPECIAL_TOKENS = [
    "<|endoftext|>",  # document separator in pretraining
    "<|pad|>",
    "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",  # chat format for SFT
    *[f"<|reserved_{i}|>" for i in range(10)],
]
# Special tokens are added first, in this order, so their ids are fixed.
EOS_ID = 0
PAD_ID = 1


def train_bpe_tokenizer(texts: Sequence[str], vocab_size: int = VOCAB_SIZE):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    if vocab_size > 65_536:
        raise ValueError("token ids must fit in uint16")
    tokenizer = Tokenizer(models.BPE())
    # GPT-2 regex split (words, numbers, punctuation, whitespace), then bytes.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes, always
        min_frequency=2,
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer, length=len(texts))
    return tokenizer


def load_tokenizer(path: str | Path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))


def bytes_per_token(tokenizer, texts: Iterable[str]) -> float:
    """Average UTF-8 bytes per token over `texts`: higher means better compression.

    The byte tokenizer of the original repository scores exactly 1.0 by definition.
    """
    # TODO(you): total UTF-8 bytes / total tokens.
    # hint: tokenizer.encode_batch(list_of_texts) returns encodings; each has `.ids`.
    text_list = list(texts)
    if not text_list:
        return 0.0

    total_bytes = sum(len(text.encode("utf-8")) for text in text_list)
    encodings = tokenizer.encode_batch(text_list)
    total_tokens = sum(len(enc.ids) for enc in encodings)


    if total_tokens == 0:
        return 0.0
    return total_bytes / total_tokens