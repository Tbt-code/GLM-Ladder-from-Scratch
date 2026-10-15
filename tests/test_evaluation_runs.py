import json
import shutil
from pathlib import Path

import math
import numpy as np
import pytest
import torch

from glm_ladder import GLMLadderModel, ModelConfig
from glm_ladder.checkpoint import load_checkpoint, save_checkpoint
from glm_ladder.evaluation import evaluate
from glm_ladder.loader import PackedSource
from glm_ladder.runs import MetricsLog, RunStore
from glm_ladder.train import param_groups
from tests.fakes import FakeHub 

TINY = ModelConfig(vocab_size=64, dim=32, layers=4, heads=4, expert_hidden=48, experts=4,
                   sparse_window=8, sparse_stride=8, max_sequence_length=64, linear_chunk_size=8)
CPU = torch.device("cpu")


@pytest.fixture
def val_source(tmp_path):
    path = tmp_path / "val-000.bin"
    np.random.default_rng(0).integers(0, 64, size=1000).astype(np.uint16).tofile(path)
    return PackedSource([str(path)], 16)


# ----------------------------------------------------------------- evaluate

def test_evaluate_is_deterministic_and_near_uniform_at_init(val_source) -> None:
    torch.manual_seed(0)
    model = GLMLadderModel(TINY)
    a = evaluate(model, {"v": val_source}, windows=10, batch_size=3, device=CPU)
    b = evaluate(model, {"v": val_source}, windows=10, batch_size=3, device=CPU)
    assert a == b
    assert a["v"] == pytest.approx(math.log(64), abs=0.3)


def test_evaluate_does_not_change_the_model(val_source) -> None:
    model = GLMLadderModel(TINY)
    before = [p.clone() for p in model.parameters()]
    evaluate(model, {"v": val_source}, windows=5, batch_size=2, device=CPU)
    assert model.training                       # returned to training mode
    for p, q in zip(before, model.parameters()):
        assert torch.equal(p, q) and q.grad is None


def test_uneven_last_batch_is_weighted_correctly(val_source) -> None:
    model = GLMLadderModel(TINY)
    one_batch = evaluate(model, {"v": val_source}, windows=7, batch_size=7, device=CPU)
    uneven = evaluate(model, {"v": val_source}, windows=7, batch_size=3, device=CPU)   # 3 + 3 + 1
    assert uneven["v"] == pytest.approx(one_batch["v"], rel=1e-5)





def make_checkpoint(directory: Path, step: int) -> Path:
    model = GLMLadderModel(TINY)
    optimizer = torch.optim.AdamW(param_groups(model, 0.1))
    save_checkpoint(directory / f"step-{step:06d}", model=model, optimizer=optimizer, trainer_state={"step": step})
    return directory / f"step-{step:06d}"


def test_store_keeps_last_two_and_milestones(tmp_path) -> None:
    hub = FakeHub(tmp_path / "hub")
    store = RunStore(hub, "user/runs", "demo", keep_last=2)
    assert store.latest() is None
    for step in (100, 200, 300):
        store.push(make_checkpoint(tmp_path / "local", step))
    store.push(make_checkpoint(tmp_path / "local", 400), milestone=True)
    index = store.index()
    assert index["latest"] == "step-000400"
    assert index["resume"] == ["step-000200", "step-000300"]
    assert index["milestones"] == ["step-000400"]
    remote = tmp_path / "hub" / "runs" / "demo"
    assert not (remote / "step-000100").exists()           # pruned
    assert (remote / "step-000200").exists() and (remote / "step-000400").exists()


def test_pull_restores_a_loadable_checkpoint(tmp_path) -> None:
    hub = FakeHub(tmp_path / "hub")
    store = RunStore(hub, "user/runs", "demo")
    store.push(make_checkpoint(tmp_path / "local", 100))
    directory = store.pull(store.latest(), tmp_path / "restored")
    model = GLMLadderModel(TINY)
    state = load_checkpoint(directory, model=model, optimizer=torch.optim.AdamW(param_groups(model, 0.1)), device=CPU)
    assert state["step"] == 100                              # also checks the SHA-256


def test_metrics_truncate_after_resume(tmp_path) -> None:
    log = MetricsLog(tmp_path / "metrics.jsonl")
    for step in range(1, 11):
        log.append({"step": step, "loss": 10.0 - step})
    log.truncate(6)
    assert [row["step"] for row in log.rows()] == [1, 2, 3, 4, 5, 6]
    log.append({"step": 7, "loss": 3.5})
    assert [row["step"] for row in log.rows()][-2:] == [6, 7]