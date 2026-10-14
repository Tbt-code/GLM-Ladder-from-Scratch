"""Deterministic, resumable batches from uint16 token shards.

Each source's shards are viewed as one sequence of non-overlapping windows of
`seq_len + 1` tokens (consecutive windows share one token: inputs and shifted
targets). Windows are visited in a seeded permutation, so every window is seen
once per epoch. Each step draws which source every sample comes from with
rng([seed, step]); the whole loader state is `step` plus one cursor per source.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


def token_mix(byte_mix: Mapping[str, float], bytes_per_token: Mapping[str, float]) -> dict[str, float]:
    """Convert a mix defined in UTF-8 bytes into the equivalent mix in tokens."""
    #  weight_i = byte_mix[i] / bytes_per_token[i], then normalise to sum to 1.
    raw_mix = {name: byte_mix[name] / bytes_per_token[name] for name in byte_mix}
    total = sum(raw_mix.values())
    return {name:val / total for name,val in raw_mix.items()}



class PackedSource:
    """One source's shards, addressed as a flat range of windows."""

    def __init__(self, paths: Sequence[str | Path], seq_len: int) -> None:
        self.seq_len = seq_len
        self.shards = [np.memmap(path, dtype=np.uint16, mode="r") for path in paths]
        windows = [(len(shard) - 1) // seq_len for shard in self.shards]
        self.cumulative = np.cumsum(windows)          # windows before the end of each shard
        self.num_windows = int(self.cumulative[-1])

    def window(self, index: int) -> np.ndarray:
        """Return window `index` as seq_len + 1 tokens."""
        if not 0 <= index < self.num_windows:
            raise IndexError(index)
       
        #   1. find the shard: the first shard whose cumulative count is > index
        shard_idx = int(np.searchsorted(self.cumulative, index, side="right"))

        #   2. local window number = index minus the windows of all earlier shards
        prev_cumulative = int(self.cumulative[shard_idx - 1]) if shard_idx > 0 else 0
        local_window = index - prev_cumulative

        #   3. start = local * seq_len; return shard[start : start + seq_len + 1]
        start = local_window * self.seq_len
        shard = self.shards[shard_idx]
        return shard[start : start + self.seq_len + 1]


class MixedLoader:
    def __init__(
        self, sources: Mapping[str, PackedSource], weights: Mapping[str, float],
        *, batch_size: int, seed: int,
    ) -> None:
        if set(sources) != set(weights):
            raise ValueError("sources and weights must have the same names")
        self.names = sorted(sources)                 # fixed order -> reproducible choices
        self.sources = sources
        self.probabilities = np.array([weights[n] for n in self.names], dtype=np.float64)
        self.probabilities /= self.probabilities.sum()
        self.batch_size = batch_size
        self.seed = seed
        self.step = 0
        self.cursors = {name: 0 for name in self.names}
        self._permutations: dict[tuple[str, int], np.ndarray] = {}

    def _window_index(self, name: str, cursor: int) -> int:
        n = self.sources[name].num_windows
        epoch, position = divmod(cursor, n)
        key = (name, epoch)
        if key not in self._permutations:
            source_id = self.names.index(name)
            self._permutations[key] = np.random.default_rng([self.seed, source_id, epoch]).permutation(n)
        return int(self._permutations[key][position])

    def next_batch(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (inputs, targets), each int64 [batch_size, seq_len]."""
        rng = np.random.default_rng([self.seed, self.step])
        choices = rng.choice(len(self.names), size=self.batch_size, p=self.probabilities)
        rows = []
        for choice in choices:
            name = self.names[choice]
            rows.append(self.sources[name].window(self._window_index(name, self.cursors[name])))
            self.cursors[name] += 1
        self.step += 1
        batch = np.stack(rows).astype(np.int64)
        return batch[:, :-1], batch[:, 1:]

    def state_dict(self) -> dict:
        """Everything needed to continue exactly where we are. Must be JSON-serialisable."""
        return {
            "seed": self.seed,
            "names": self.names,
            "step": self.step,
            "cursors": self.cursors
        }

    def load_state_dict(self, state: dict) -> None:
        # restore step and cursors. Refuse (ValueError) a state saved with a
        # different seed or different source names: resuming it would silently change the data.
        if state.get("seed") != self.seed:
            raise ValueError("State seed does not match loader seed.")
        if state.get("names") != self.names:
            raise ValueError("State source names do not match loader source names.")

        self.step = state["step"]
        self.cursors = dict(state["cursors"])