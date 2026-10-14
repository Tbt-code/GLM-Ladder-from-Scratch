"""Streaming access to the pretraining sources and a deterministic text sample.

Nothing is downloaded in full: `streaming=True` reads the Parquet files over HTTP
and yields one document at a time, so a laptop can sample a few hundred MB.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping

# name -> (dataset id, config, text field)
SOURCES: dict[str, tuple[str, str | None, str]] = {
    "english": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
    "code": ("codeparrot/codeparrot-clean", None, "content"),
    "turkish": ("epfml/FineWeb2-HQ", "tur_Latn", "text"),
}

# Fraction of UTF-8 bytes from each source (tokenizer sample and pretraining stream).
MIX: dict[str, float] = {"english": 0.6, "code": 0.3, "turkish": 0.1}

# Very long files are mostly minified code, data dumps, or generated tables.
MAX_DOCUMENT_CHARS = 100_000


def iter_documents(name: str) -> Iterator[str]:
    """Yield documents from one source in its fixed streaming order."""
    from datasets import load_dataset  # imported lazily so tests run without network

    path, config, field = SOURCES[name]
    for row in load_dataset(path, name=config, split="train", streaming=True):
        text = row[field]
        if text and len(text) <= MAX_DOCUMENT_CHARS:
            yield text


def take_bytes(texts: Iterable[str], budget: int) -> Iterator[str]:
    """Yield texts until their total UTF-8 size reaches `budget` bytes.

    The text that crosses the budget is still yielded (whole documents only),
    so the total may overshoot by at most one document.
    """
    if budget <= 0:
        return
    used = 0
    for text in texts:
        yield text
        used += len(text.encode("utf-8"))
        if used >= budget:
            return


def mixed_sample(
    total_bytes: int,
    mix: Mapping[str, float],
    sources: Mapping[str, Iterable[str]],
) -> list[str]:
    """Return roughly `total_bytes` of text, `mix[name]` of it taken from `sources[name]`."""

    #   1. check that the fractions sum to 1 (within a small tolerance) -> ValueError otherwise
    total_fraction = sum(mix.values())
    if abs(total_fraction - 1.0) > 1e-6:
        raise ValueError(f"Mix fractions must sum to 1.0, got {total_fraction}")

    result: list[str] = []

    # 2. for each name: budget = int(total_bytes * fraction); take_bytes from that source
    for name,fraction in mix.items():
        if fraction <= 0:
            continue
        budget = int(total_bytes * fraction)
        source_iterable = sources[name]

        #   3. return everything as one list
        result.extend(list(take_bytes(source_iterable,budget)))
    
    return result