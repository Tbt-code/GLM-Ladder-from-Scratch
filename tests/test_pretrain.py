from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from glm_ladder import ModelConfig
from glm_ladder.loader import PackedSource
from glm_ladder.pretrain import PretrainConfig, check_resume_compatible, run_pretraining
from glm_ladder.runs import MetricsLog, RunStore
from tests.fakes import FakeHub

TINY = ModelConfig(vocab_size=64, dim=32, layers=4, heads=4, expert_hidden=48, experts=4,
                   sparse_window=8, sparse_stride=8, max_sequence_length=64, linear_chunk_size=8)
CPU = torch.device("cpu")
SEQ = 16
CFG = PretrainConfig(
    run_name="tiny", total_tokens=12 * 8 * SEQ, seq_len=SEQ, batch_rows=8, micro_batch=4,
    warmup_steps=2, eval_every=4, eval_windows=4, log_every=1,
    checkpoint_minutes=1e9, checkpoint_every_steps=4,     # step-based only: deterministic in tests
)


def write_tokens(path, tokens) -> str:
    np.asarray(tokens, dtype=np.uint16).tofile(path)
    return str(path)


@pytest.fixture
def data(tmp_path):
    rng = np.random.default_rng(0)
    train, val = {}, {}
    for name in ("a", "b"):
        train[name] = PackedSource([write_tokens(tmp_path / f"{name}-train.bin", rng.integers(0, 64, 4000))], SEQ)
        val[name] = PackedSource([write_tokens(tmp_path / f"{name}-val.bin", rng.integers(0, 64, 600))], SEQ)
    return {"train_sources": train, "val_sources": val, "weights": {"a": 0.7, "b": 0.3}}


def run(cfg, root, data, **kwargs) -> dict:
    store = RunStore(FakeHub(root / "hub"), "user/runs", cfg.run_name)
    return run_pretraining(cfg, TINY, store=store, workdir=root / "work", device=CPU,
                           log=lambda _: None, **data, **kwargs)


def train_steps(root, run_name="tiny") -> list[int]:
    rows = MetricsLog(root / "hub" / "runs" / run_name / "metrics.jsonl").rows()
    return [row["step"] for row in rows if "loss" in row]


def test_resume_compatibility_check() -> None:
    saved = asdict(CFG)
    check_resume_compatible(saved, asdict(replace(CFG, micro_batch=2, compile=True, log_every=50)))
    with pytest.raises(ValueError, match="max_lr"):
        check_resume_compatible(saved, asdict(replace(CFG, max_lr=1e-3)))
    with pytest.raises(ValueError) as error:
        check_resume_compatible(saved, asdict(replace(CFG, seed=7, batch_rows=16)))
    assert "seed" in str(error.value) and "batch_rows" in str(error.value)     # every field, not just the first


def test_interrupted_run_resumes_bit_identically(tmp_path, data) -> None:
    straight = run(CFG, tmp_path / "straight", data)
    assert straight["finished"] and straight["step"] == 12

    first = run(CFG, tmp_path / "resumed", data, stop_after_steps=6)     # checkpoint at 4, lost 5-6
    assert not first["finished"] and first["step"] == 6
    resumed = run(CFG, tmp_path / "resumed", data)                       # continues from step 4
    assert resumed["finished"] and resumed["tokens_seen"] == straight["tokens_seen"]

    for (name, p), q in zip(straight["model"].state_dict().items(), resumed["model"].state_dict().values()):
        assert torch.equal(p, q), name
    assert train_steps(tmp_path / "resumed") == list(range(1, 13))       # no repeated steps 5-6
    assert train_steps(tmp_path / "straight") == list(range(1, 13))


def test_store_after_a_finished_run(tmp_path, data) -> None:
    run(CFG, tmp_path, data)
    index = RunStore(FakeHub(tmp_path / "hub"), "user/runs", "tiny").index()
    assert index["latest"] == "step-000012"
    assert index["milestones"] == ["step-000012"]
    assert index["resume"] == ["step-000004", "step-000008"]
    rows = MetricsLog(tmp_path / "hub" / "runs" / "tiny" / "metrics.jsonl").rows()
    evals = [row for row in rows if "eval" in row]
    assert [row["step"] for row in evals] == [4, 8, 12]
    assert set(evals[0]["eval"]) == {"a", "b"}
    assert not (tmp_path / "work" / "tiny" / "step-000012").exists()      # local copy removed after upload


def test_resume_refuses_a_changed_run_definition(tmp_path, data) -> None:
    run(CFG, tmp_path, data, stop_after_steps=5)
    with pytest.raises(ValueError, match="max_lr"):
        run(replace(CFG, max_lr=1e-3), tmp_path, data)
    run(replace(CFG, micro_batch=2), tmp_path, data)     # a different micro-batch is fine


def test_loss_goes_down_on_learnable_data(tmp_path) -> None:
    # Every next token is the current token + 1 (mod 64): a bigram the model can learn quickly.
    sequence = np.arange(4000) % 64
    source = PackedSource([write_tokens(tmp_path / "count.bin", sequence)], SEQ)
    cfg = replace(CFG, run_name="count", total_tokens=40 * 8 * SEQ, max_lr=3e-3, warmup_steps=5,
                  eval_every=40, checkpoint_every_steps=None)
    run(cfg, tmp_path, {"train_sources": {"c": source}, "val_sources": {"c": source}, "weights": {"c": 1.0}})
    rows = MetricsLog(tmp_path / "hub" / "runs" / "count" / "metrics.jsonl").rows()
    losses = [row["loss"] for row in rows if "loss" in row]
    assert losses[0] == pytest.approx(np.log(64), abs=0.5)
    assert losses[-1] < losses[0] - 0.5
