import json

import numpy as np
import pytest

from glm_ladder.loader import MixedLoader, PackedSource, token_mix

SEQ = 8


def write_shard(path, values) -> str:
    np.asarray(values, dtype=np.uint16).tofile(path)
    return str(path)


@pytest.fixture
def two_sources(tmp_path):
    # Source "a" holds values 1000+, source "b" values 2000+, so a row's origin is visible.
    a = [write_shard(tmp_path / "a0.bin", np.arange(1000, 1017)),   # 17 tokens -> 2 windows
         write_shard(tmp_path / "a1.bin", np.arange(1100, 1125))]   # 25 tokens -> 3 windows
    b = [write_shard(tmp_path / "b0.bin", np.arange(2000, 2081))]   # 81 tokens -> 10 windows
    return {"a": PackedSource(a, SEQ), "b": PackedSource(b, SEQ)}


def test_window_maps_across_shards(two_sources) -> None:
    a = two_sources["a"]
    assert a.num_windows == 5
    np.testing.assert_array_equal(a.window(1), np.arange(1008, 1017))   # end of shard 0
    np.testing.assert_array_equal(a.window(2), np.arange(1100, 1109))   # start of shard 1
    np.testing.assert_array_equal(a.window(4), np.arange(1116, 1125))   # last window
    with pytest.raises(IndexError):
        a.window(5)


def test_targets_are_inputs_shifted_by_one(two_sources) -> None:
    loader = MixedLoader(two_sources, {"a": 0.5, "b": 0.5}, batch_size=4, seed=0)
    x, y = loader.next_batch()
    assert x.shape == y.shape == (4, SEQ) and x.dtype == np.int64
    np.testing.assert_array_equal(x[:, 1:], y[:, :-1])


def test_each_window_seen_once_per_epoch(two_sources) -> None:
    loader = MixedLoader({"b": two_sources["b"]}, {"b": 1.0}, batch_size=5, seed=3)
    starts = [row[0] for _ in range(2) for row in loader.next_batch()[0]]
    assert sorted(starts) == list(range(2000, 2080, SEQ))   # all 10 windows, no repeats


def test_resume_gives_identical_batches(two_sources) -> None:
    reference = MixedLoader(two_sources, {"a": 0.3, "b": 0.7}, batch_size=3, seed=42)
    expected = [reference.next_batch()[0] for _ in range(20)]

    first = MixedLoader(two_sources, {"a": 0.3, "b": 0.7}, batch_size=3, seed=42)
    for _ in range(10):
        first.next_batch()
    saved = json.loads(json.dumps(first.state_dict()))       # as if written to a checkpoint

    resumed = MixedLoader(two_sources, {"a": 0.3, "b": 0.7}, batch_size=3, seed=42)
    resumed.load_state_dict(saved)
    for step in range(10, 20):
        np.testing.assert_array_equal(resumed.next_batch()[0], expected[step])


def test_refuses_state_from_another_seed(two_sources) -> None:
    saved = MixedLoader(two_sources, {"a": 0.5, "b": 0.5}, batch_size=2, seed=1).state_dict()
    other = MixedLoader(two_sources, {"a": 0.5, "b": 0.5}, batch_size=2, seed=2)
    with pytest.raises(ValueError):
        other.load_state_dict(saved)


def test_mix_is_respected(two_sources) -> None:
    loader = MixedLoader(two_sources, {"a": 0.25, "b": 0.75}, batch_size=50, seed=7)
    rows = np.concatenate([loader.next_batch()[0] for _ in range(40)])
    share_a = np.mean(rows[:, 0] < 2000)
    assert abs(share_a - 0.25) < 0.03


def test_token_mix_uses_compression() -> None:
    mix = token_mix({"english": 0.6, "code": 0.3, "turkish": 0.1},
                    {"english": 4.103, "code": 3.468, "turkish": 3.507})
    assert abs(sum(mix.values()) - 1) < 1e-12
    assert mix["english"] == pytest.approx(0.560, abs=1e-3)
    assert mix["code"] == pytest.approx(0.331, abs=1e-3)
    assert mix["turkish"] == pytest.approx(0.109, abs=1e-3)