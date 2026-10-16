"""SFT data: whole conversations packed into fixed-length windows with a loss mask.

Pretraining cut one long token stream into windows; a document could be split anywhere.
For SFT that would be harmful: the second half of a conversation would be trained
without the question it answers. Here every conversation stays inside one window:

    [conv A ... EOS][conv B ... EOS][conv C ... EOS][PAD PAD PAD]   <- one window
    [conv D .................... EOS][PAD PAD PAD PAD PAD PAD PAD]   <- next window

Conversations are placed in order ("next fit"): if the next one does not fit in the
space left, the window is padded and a new one starts. EOS and PAD are never learned.
Stored as two arrays of shape [windows, seq_len + 1]: uint16 token ids and uint8 mask.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from .tokenizer import EOS_ID, PAD_ID


def pack_conversations(
    conversations: Iterable[tuple[list[int], list[bool]]], window: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (tokens uint16 [window], learn bool [window]) windows of whole conversations.

    Each conversation is followed by EOS. A conversation that does not fit in an empty
    window (len(ids) + 1 > window) raises ValueError; filter those out before packing.
    """
    tokens: list[int] = []
    learn: list[bool] = []
    
    for ids, mask in conversations:
        # 1. length = len(ids) + 1 (the EOS); if length > window -> ValueError
        length = len(ids) + 1
        if length > window:
            raise ValueError(f"Conversation length ({length}) exceeds window size ({window}).")
            
        # 2. if it does not fit in what is left of the current window (len(tokens) + length > window):
        # yield the current window padded to `window` with PAD_ID / False, and start empty
        if len(tokens) + length > window:
            yield _finish(tokens, learn, window)
            tokens = []
            learn = []
            
        # 3. extend tokens with ids + [EOS_ID] and learn with mask + [False]
        tokens.extend(ids)
        tokens.append(EOS_ID)
        learn.extend(mask)
        learn.append(False)  # EOS is never learned
        
    # after the loop: if anything is left, yield it padded the same way
    if tokens:
        yield _finish(tokens, learn, window)
        

def _finish(tokens: list[int], learn: list[bool], window: int) -> tuple[np.ndarray, np.ndarray]:
    pad = window - len(tokens)
    return (np.array(tokens + [PAD_ID] * pad, dtype=np.uint16),
            np.array(learn + [False] * pad, dtype=bool))


class SFTSource:
    """Packed windows of one SFT source, read with memory mapping."""

    def __init__(self, directory: str | Path) -> None:
        directory = Path(directory)
        self.tokens = np.load(directory / "tokens.npy", mmap_mode="r")
        self.mask = np.load(directory / "mask.npy", mmap_mode="r")
        if self.tokens.shape != self.mask.shape:
            raise ValueError("tokens and mask shapes differ")
        self.num_windows, width = self.tokens.shape
        self.seq_len = width - 1

    def window(self, index: int) -> np.ndarray:
        """seq_len + 1 token ids; same interface as PackedSource."""
        return np.asarray(self.tokens[index])

    def learn(self, index: int) -> np.ndarray:
        """seq_len + 1 booleans: True where that token is a training target."""
        return np.asarray(self.mask[index], dtype=bool)