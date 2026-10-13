"""Tests for the data sampling helpers. No network access needed."""
import pytest

from glm_ladder.data_sources import MIX, mixed_sample, take_bytes


def test_take_bytes_stops_after_budget_is_reached() -> None:
    texts = ["aaaa", "bbbb", "cccc", "dddd"]  # 4 bytes each
    assert list(take_bytes(texts, budget=8)) == ["aaaa", "bbbb"]
    assert list(take_bytes(texts, budget=9)) == ["aaaa", "bbbb", "cccc"]  # overshoot by one doc
    assert list(take_bytes(texts, budget=0)) == []


def test_take_bytes_counts_utf8_bytes_not_characters() -> None:
    # "ğ" is one character but two UTF-8 bytes.
    assert list(take_bytes(["ğğ", "x"], budget=4)) == ["ğğ"]


def test_take_bytes_is_lazy() -> None:
    def endless():
        while True:
            yield "abcd"

    # Must terminate even though the source never ends.
    assert len(list(take_bytes(endless(), budget=12))) == 3


def test_mixed_sample_respects_fractions() -> None:
    sources = {
        "english": ["e" * 100] * 1000,
        "code": ["c" * 100] * 1000,
        "turkish": ["t" * 100] * 1000,
    }
    sample = mixed_sample(20_000, MIX, sources)
    total = sum(len(s) for s in sample)
    for name, letter in (("english", "e"), ("code", "c"), ("turkish", "t")):
        share = sum(len(s) for s in sample if s.startswith(letter)) / total
        assert abs(share - MIX[name]) < 0.02


def test_mixed_sample_rejects_fractions_that_do_not_sum_to_one() -> None:
    with pytest.raises(ValueError):
        mixed_sample(1000, {"english": 0.5, "code": 0.3}, {"english": [], "code": []})