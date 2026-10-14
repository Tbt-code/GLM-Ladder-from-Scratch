"""Turn a stream of documents into flat uint16 token shards.

Each document becomes its tokens followed by <|endoftext|>. Shards end on
document boundaries, so resuming only needs the number of documents consumed.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .tokenizer import EOS_ID, SPECIAL_TOKENS


def contains_special_token(text: str) -> bool:
    """True if the raw text contains a special-token string such as "<|endoftext|>".

    The tokenizer would turn such text into a real special token, so these
    (rare) documents are skipped rather than silently corrupted.
    """
    return "<|" in text and any(token in text for token in SPECIAL_TOKENS)


def _encode_batch(tokenizer, texts: list[str]) -> Iterator[np.ndarray]:
    keep = [text for text in texts if not contains_special_token(text)]
    encoded = iter(tokenizer.encode_batch(keep))
    for text in texts:
        if contains_special_token(text):
            yield np.empty(0, dtype=np.uint16)  # skipped document
        else:
            ids = next(encoded).ids
            yield np.array([*ids, EOS_ID], dtype=np.uint16)


def tokenize_stream(tokenizer, documents: Iterable[str], batch_size: int = 1024) -> Iterator[np.ndarray]:
    """Yield one uint16 array per input document, in order (empty if skipped).

    Documents are encoded in batches because `encode_batch` uses every CPU core.
    """
    batch: list[str] = []
    for text in documents:
        batch.append(text)
        if len(batch) == batch_size:
            yield from _encode_batch(tokenizer, batch)
            batch = []
    if batch:
        yield from _encode_batch(tokenizer, batch)


@dataclass(frozen=True)
class ShardInfo:
    name: str
    tokens: int
    documents: int          # documents in this shard, skipped ones included
    skipped_documents: int
    documents_end: int      # documents consumed from the source stream after this shard
    sha256: str


class ShardWriter:
    """Collect documents and write a shard once it holds at least `shard_tokens` tokens."""

    def __init__(
        self, directory: Path, prefix: str, shard_tokens: int,
        *, start_index: int = 0, documents_consumed: int = 0,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.shard_tokens = shard_tokens
        self.index = start_index
        self.documents_consumed = documents_consumed
        self._parts: list[np.ndarray] = []
        self._tokens = 0
        self._documents = 0
        self._skipped = 0

    def add(self, document_tokens: np.ndarray) -> ShardInfo | None:
        """Add one document. Return the finished shard's info if this document filled it."""
        #   1. count the document as consumed (self.documents_consumed, self._documents)
        self.documents_consumed += 1
        self._documents += 1

         #   2. if it is empty, it was skipped: count it in self._skipped
        #    otherwise keep it in self._parts and add its length to self._tokens
        if document_tokens.size == 0:
            self._skipped += 1
        else:
            self._parts.append(document_tokens)
            self._tokens += document_tokens.size

        #   3. once self._tokens >= self.shard_tokens, return self.flush(); otherwise None
        if self._tokens >= self.shard_tokens:
            return self.flush()

        return None



    def flush(self) -> ShardInfo | None:
        """Write whatever is buffered as a shard (used at the end of a stream)."""
        if self._tokens == 0:
            return None
        name = f"{self.prefix}-{self.index:03d}.bin"
        path = self.directory / name
        np.concatenate(self._parts).tofile(path)
        info = ShardInfo(
            name=name,
            tokens=self._tokens,
            documents=self._documents,
            skipped_documents=self._skipped,
            documents_end=self.documents_consumed,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        self.index += 1
        self._parts, self._tokens, self._documents, self._skipped = [], 0, 0, 0
        return info