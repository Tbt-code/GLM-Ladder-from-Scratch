"""A minimal byte-level BPE tokenizer, written to understand the algorithm.

The production tokenizer uses the Hugging Face `tokenizers` library; this file
exists for learning and for tests, not for training runs.
"""
from __future__ import annotations


def get_pair_counts(ids: list[int]) -> dict[tuple[int, int], int]:
    """Count how often each adjacent pair occurs, e.g. [1, 2, 1, 2] -> {(1,2): 2, (2,1): 1}."""
    counts = {}
    for i in range(len(ids) - 1):
        pair = (ids[i],ids[i+1])
        counts[pair] = counts.get(pair,0) + 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of `pair` with `new_id`, scanning left to right.

    merge([1, 2, 3, 1, 2], (1, 2), 99) -> [99, 3, 99]
    """
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i],ids[i+1]) == pair:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


class ToyBPE:
    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}            # pair -> new token id
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, vocab_size: int) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")
        ids = list(text.encode("utf-8"))
        for new_id in range(256, vocab_size):
            counts = get_pair_counts(ids)
            if not counts:
                break  # nothing left to merge
            pair = max(counts, key=counts.get)  # most frequent pair
            ids = merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            pairs = list(zip(ids,ids[1:]))

            best_pair = min(pairs,key=lambda p: self.merges.get(p,float("inf")))
            if best_pair not in self.merges:
                break
            new_id = self.merges[best_pair]
            ids = merge(ids,best_pair,new_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")