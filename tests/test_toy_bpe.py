from glm_ladder.toy_bpe import ToyBPE, get_pair_counts, merge


def test_get_pair_counts() -> None:
    assert get_pair_counts([1, 2, 1, 2]) == {(1, 2): 2, (2, 1): 1}
    assert get_pair_counts([5]) == {}


def test_merge() -> None:
    assert merge([1, 2, 3, 1, 2], (1, 2), 99) == [99, 3, 99]
    assert merge([1, 1, 1], (1, 1), 7) == [7, 1]  # left to right, no overlap


def test_training_matches_the_worked_example() -> None:
    bpe = ToyBPE()
    bpe.train("aaabdaaabac", vocab_size=259)
    assert bpe.vocab[256] == b"aa"
    assert bpe.vocab[257] == b"aaa"
    assert bpe.vocab[258] == b"aaab"
    assert bpe.encode("aaabdaaabac") == [258, ord("d"), 258, ord("a"), ord("c")]


def test_roundtrip_any_text() -> None:
    bpe = ToyBPE()

    bpe.train("Hello world, this is a simple English text for training the tokenizer.", vocab_size=265)
    unseen_text = "Merhaba dünya! Türkçe karakterler: ğüşİıÖçŞ ve bir emoji: 🚀🔥"

    encoded = bpe.encode(unseen_text)
    decoded = bpe.decode(encoded)

    assert decoded == unseen_text