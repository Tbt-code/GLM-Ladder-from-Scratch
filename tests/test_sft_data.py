import numpy as np
import pytest

from glm_ladder.sft_data import SFTSource, pack_conversations
from glm_ladder.tokenizer import EOS_ID, PAD_ID


def conversation(start: int, length: int, learned_from: int) -> tuple[list[int], list[bool]]:
    """Fake conversation: ids start..start+length-1 (all > PAD_ID), learned from position `learned_from`."""
    ids = list(range(start, start + length))
    return ids, [i >= learned_from for i in range(length)]


def unpack(windows) -> list[list[int]]:
    """Recover the conversations from packed windows (drop padding, split on EOS)."""
    conversations = []
    for tokens, _ in windows:
        current: list[int] = []
        for token in tokens.tolist():
            if token == PAD_ID:
                break
            if token == EOS_ID:
                conversations.append(current)
                current = []
            else:
                current.append(token)
        assert current == []                 # no conversation is cut at a window border
    return conversations


def test_conversations_stay_whole_and_in_order() -> None:
    convs = [conversation(100 * k + 10, length, 2) for k, length in enumerate([5, 7, 3, 9, 2, 6])]
    windows = list(pack_conversations(convs, window=12))
    assert unpack(windows) == [ids for ids, _ in convs]
    for tokens, learn in windows:
        assert tokens.shape == learn.shape == (12,)
        assert tokens.dtype == np.uint16 and learn.dtype == bool


def test_next_fit_layout() -> None:
    # lengths with EOS: 6, 8, 4 in a window of 12 -> [6 + pad][8 + 4]
    convs = [conversation(10, 5, 0), conversation(30, 7, 0), conversation(50, 3, 0)]
    windows = list(pack_conversations(convs, window=12))
    assert len(windows) == 2
    assert windows[0][0].tolist() == [10, 11, 12, 13, 14, EOS_ID] + [PAD_ID] * 6
    assert windows[1][0].tolist() == [30, 31, 32, 33, 34, 35, 36, EOS_ID, 50, 51, 52, EOS_ID]


def test_mask_is_kept_and_eos_and_padding_are_never_learned() -> None:
    convs = [conversation(10, 6, 3), conversation(40, 4, 1)]
    windows = list(pack_conversations(convs, window=16))
    tokens, learn = windows[0]
    assert sum(learn) == (6 - 3) + (4 - 1)
    assert not learn[tokens == EOS_ID].any()
    assert not learn[tokens == PAD_ID].any()


def test_conversation_exactly_filling_a_window() -> None:
    windows = list(pack_conversations([conversation(10, 11, 0)], window=12))
    assert len(windows) == 1 and windows[0][0][-1] == EOS_ID


def test_too_long_conversation_is_rejected() -> None:
    with pytest.raises(ValueError):
        list(pack_conversations([conversation(10, 12, 0)], window=12))    # 12 + EOS > 12


def test_empty_input_gives_no_windows() -> None:
    assert list(pack_conversations([], window=12)) == []


def test_sft_source_reads_saved_arrays(tmp_path) -> None:
    windows = list(pack_conversations([conversation(10, 5, 2), conversation(30, 8, 4)], window=12))
    np.save(tmp_path / "tokens.npy", np.stack([t for t, _ in windows]))
    np.save(tmp_path / "mask.npy", np.stack([m for _, m in windows]).astype(np.uint8))
    source = SFTSource(tmp_path)
    assert source.num_windows == 2 and source.seq_len == 11
    assert source.window(1).tolist() == windows[1][0].tolist()
    assert source.learn(1).tolist() == windows[1][1].tolist()