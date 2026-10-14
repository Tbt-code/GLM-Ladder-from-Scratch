import hashlib

import numpy as np

from glm_ladder.shards import ShardWriter, contains_special_token, tokenize_stream
from glm_ladder.tokenizer import EOS_ID, train_bpe_tokenizer


def documents(*lengths: int) -> list[np.ndarray]:
    return [np.arange(1, n + 1, dtype=np.uint16) for n in lengths]


def test_shards_end_on_document_boundaries(tmp_path) -> None:
    writer = ShardWriter(tmp_path, "train", shard_tokens=10)
    finished = [info for doc in documents(4, 4, 4, 4, 4) if (info := writer.add(doc))]
    last = writer.flush()
    # 4 + 4 + 4 = 12 >= 10, so the first shard closes after the third document.
    assert [info.tokens for info in finished] == [12]
    assert finished[0].documents_end == 3
    assert last.tokens == 8 and last.documents_end == 5
    assert [finished[0].name, last.name] == ["train-000.bin", "train-001.bin"]


def test_shard_files_contain_exactly_the_documents(tmp_path) -> None:
    docs = documents(3, 5, 2, 7)
    writer = ShardWriter(tmp_path, "train", shard_tokens=6)
    infos = [info for doc in docs if (info := writer.add(doc))] + [writer.flush()]
    written = np.concatenate([np.fromfile(tmp_path / i.name, dtype=np.uint16) for i in infos if i])
    np.testing.assert_array_equal(written, np.concatenate(docs))
    for info in infos:
        if info:
            assert hashlib.sha256((tmp_path / info.name).read_bytes()).hexdigest() == info.sha256


def test_skipped_documents_are_consumed_but_not_written(tmp_path) -> None:
    writer = ShardWriter(tmp_path, "val", shard_tokens=100, documents_consumed=50)
    writer.add(np.empty(0, dtype=np.uint16))
    writer.add(np.arange(1, 4, dtype=np.uint16))
    info = writer.flush()
    assert info.tokens == 3 and info.documents == 2 and info.skipped_documents == 1
    assert info.documents_end == 52  # resumes count from where the previous shard ended


def test_tokenize_stream_appends_eos_and_skips_special_text() -> None:
    tokenizer = train_bpe_tokenizer(["hello world " * 50], vocab_size=300)
    texts = ["hello world", "about <|endoftext|> tokens", "world"]
    arrays = list(tokenize_stream(tokenizer, texts, batch_size=2))
    assert len(arrays) == 3
    assert arrays[0][-1] == EOS_ID and arrays[2][-1] == EOS_ID
    assert arrays[1].size == 0
    assert tokenizer.decode(arrays[0][:-1].tolist()) == "hello world"


def test_contains_special_token() -> None:
    assert contains_special_token("x <|pad|> y")
    assert not contains_special_token("a < | b")