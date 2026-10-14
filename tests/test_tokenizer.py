import numpy as np

from glm_ladder.tokenizer import EOS_ID, PAD_ID, SPECIAL_TOKENS, bytes_per_token, train_bpe_tokenizer

CORPUS = [
    "The quick brown fox jumps over the lazy dog. " * 20,
    "def add(a, b):\n    return a + b\n" * 20,
    "Türkçe metinler de doğru şekilde işlenmeli. " * 20,
]


def test_special_token_ids_are_fixed() -> None:
    tokenizer = train_bpe_tokenizer(CORPUS, vocab_size=400)
    assert tokenizer.token_to_id("<|endoftext|>") == EOS_ID
    assert tokenizer.token_to_id("<|pad|>") == PAD_ID
    for index, token in enumerate(SPECIAL_TOKENS):
        assert tokenizer.token_to_id(token) == index


def test_roundtrip_unseen_text() -> None:
    tokenizer = train_bpe_tokenizer(CORPUS, vocab_size=400)
    text = "Görüşürüz! 🚀\n\tclass Foo:\n        pass  # ñ"
    assert tokenizer.decode(tokenizer.encode(text).ids) == text


def test_ids_fit_in_uint16() -> None:
    tokenizer = train_bpe_tokenizer(CORPUS, vocab_size=400)
    ids = np.array(tokenizer.encode("anything at all").ids)
    assert ids.max() < np.iinfo(np.uint16).max


def test_bytes_per_token_is_one_without_merges() -> None:
    # 256 bytes + special tokens and nothing else: every byte is its own token.
    tokenizer = train_bpe_tokenizer(CORPUS, vocab_size=256 + len(SPECIAL_TOKENS))
    assert bytes_per_token(tokenizer, ["hello", "ğüş"]) == 1.0


def test_merges_compress_text() -> None:
    tokenizer = train_bpe_tokenizer(CORPUS, vocab_size=400)
    assert bytes_per_token(tokenizer, CORPUS) > 2.0