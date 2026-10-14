import math

import numpy as np
import pytest
import torch

from glm_ladder import GLMLadderModel, ModelConfig
from glm_ladder.train import lr_at, param_groups, train_step

TINY = ModelConfig(vocab_size=64, dim=32, layers=4, heads=4, expert_hidden=48, experts=4,
                   sparse_window=8, sparse_stride=8, max_sequence_length=64, linear_chunk_size=8)
SCHEDULE = dict(max_lr=1e-3, warmup_steps=10, total_steps=100, min_ratio=0.1)


def test_lr_warmup_starts_small_and_reaches_max() -> None:
    assert 0 < lr_at(0, **SCHEDULE) < 2e-4
    assert lr_at(9, **SCHEDULE) == pytest.approx(1e-3)


def test_lr_decays_to_min_and_stays_there() -> None:
    values = [lr_at(s, **SCHEDULE) for s in range(10, 101)]
    assert all(a >= b for a, b in zip(values, values[1:]))   # never increases after warmup
    assert lr_at(100, **SCHEDULE) == pytest.approx(1e-4)
    assert lr_at(500, **SCHEDULE) == pytest.approx(1e-4)


def test_lr_midpoint_of_cosine() -> None:
    middle = 10 + 45   # halfway through the decay: cosine factor is exactly 0.5
    assert lr_at(middle, **SCHEDULE) == pytest.approx(1e-4 + 0.5 * 9e-4)


def test_param_groups_cover_every_parameter_once() -> None:
    model = GLMLadderModel(TINY)
    groups = param_groups(model, weight_decay=0.1)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    no_decay = groups[1]["params"]
    assert all(p.ndim < 2 for p in no_decay)
    assert any(p is model.final_norm.weight for p in no_decay)


def test_model_can_memorise_one_batch() -> None:
    # Karpathy's first sanity check: if a model cannot overfit a single batch, something is broken.
    torch.manual_seed(0)
    model = GLMLadderModel(TINY)
    optimizer = torch.optim.AdamW(param_groups(model, 0.0), lr=3e-3)
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, TINY.vocab_size, size=(2, 17))
    batch = [(tokens[:, :-1], tokens[:, 1:])]
    first = train_step(model, optimizer, batch, device=torch.device("cpu"))["loss"]
    assert first == pytest.approx(math.log(TINY.vocab_size), abs=0.3)   # random init ~ uniform guess
    for _ in range(80):
        last = train_step(model, optimizer, batch, device=torch.device("cpu"))["loss"]
    assert last < 0.5 * first


def test_accumulation_matches_one_big_batch() -> None:
    torch.manual_seed(0)
    a = GLMLadderModel(TINY)
    b = GLMLadderModel(TINY)
    b.load_state_dict(a.state_dict())
    rng = np.random.default_rng(1)
    tokens = rng.integers(0, TINY.vocab_size, size=(4, 17))
    x, y = tokens[:, :-1], tokens[:, 1:]
    opt_a = torch.optim.SGD(a.parameters(), lr=0.1)
    opt_b = torch.optim.SGD(b.parameters(), lr=0.1)
    train_step(a, opt_a, [(x, y)], device=torch.device("cpu"), balance_coef=0.0)
    train_step(b, opt_b, [(x[:2], y[:2]), (x[2:], y[2:])], device=torch.device("cpu"), balance_coef=0.0)
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa, pb, rtol=1e-4, atol=1e-5)