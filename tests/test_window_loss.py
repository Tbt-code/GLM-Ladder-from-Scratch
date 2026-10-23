from dataclasses import asdict, replace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from glm_ladder import ModelConfig
from glm_ladder.loader import IGNORE_INDEX
from glm_ladder.pretrain import PretrainConfig, check_resume_compatible, run_pretraining
from glm_ladder.sft_data import SFTSource
from glm_ladder.train import lm_loss

TINY = ModelConfig(vocab_size=64, dim=32, layers=4, heads=4, expert_hidden=48, experts=4,
                   sparse_window=8, sparse_stride=8, max_sequence_length=64, linear_chunk_size=8)
CPU = torch.device("cpu")
SEQ = 16
X = IGNORE_INDEX


def logits_and_targets():
    logits = torch.randn(3, 4, 10, generator=torch.Generator().manual_seed(0), requires_grad=True)
    targets = torch.tensor([[1, X, X, X],          # window 0: 1 learned token
                            [2, 3, 4, X],          # window 1: 3 learned tokens
                            [X, X, X, X]])         # window 2: nothing learned
    return logits, targets


def token_losses(logits, targets):
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none").view(targets.shape)


def test_token_mean_is_the_default_and_unchanged() -> None:
    logits, targets = logits_and_targets()
    expected = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    torch.testing.assert_close(lm_loss(logits, targets), expected)


def test_window_mean_by_hand() -> None:
    logits, targets = logits_and_targets()
    each = token_losses(logits, targets)
    window0 = each[0, 0]
    window1 = each[1, :3].mean()
    # Window 2 learns nothing and is left out: (window0 + window1) / 2, not / 3.
    torch.testing.assert_close(lm_loss(logits, targets, per_window=True), (window0 + window1) / 2)
    # The token mean weighs window 1 three times as much: (l0 + l1 + l2 + l3) / 4.
    assert not torch.isclose(lm_loss(logits, targets), lm_loss(logits, targets, per_window=True))


def test_full_windows_give_the_same_loss_either_way() -> None:
    # Pretraining: every target is learned, so both averages agree.
    logits = torch.randn(2, 5, 10, generator=torch.Generator().manual_seed(1))
    targets = torch.randint(0, 10, (2, 5), generator=torch.Generator().manual_seed(2))
    torch.testing.assert_close(lm_loss(logits, targets, per_window=True), lm_loss(logits, targets))


def test_nothing_learned_gives_zero_and_backward_still_works() -> None:
    logits, targets = logits_and_targets()
    loss = lm_loss(logits, torch.full_like(targets, X), per_window=True)
    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None and torch.count_nonzero(logits.grad) == 0


def test_short_and_long_windows_get_equal_gradient_weight() -> None:
    # Two windows with the same per-token loss pattern but 1 vs 3 learned tokens: with the
    # window mean, the total gradient each window's logits receive is the same size.
    logits, targets = logits_and_targets()
    lm_loss(logits, targets, per_window=True).backward()
    grad = logits.grad.abs().sum(dim=(1, 2))
    assert grad[2] == 0                                   # nothing learned, no gradient
    ratio = (grad[0] / grad[1]).item()
    # Token mean would give window 0 about 1/3 of window 1's gradient; window mean about 1/1.
    assert 0.5 < ratio < 2.0


# ----------------------------------------------------------------- inside the training loop

def sft_source(path, windows: int, seed: int, learned: float) -> SFTSource:
    rng = np.random.default_rng(seed)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "tokens.npy", rng.integers(2, 64, size=(windows, SEQ + 1)).astype(np.uint16))
    np.save(path / "mask.npy", (rng.random((windows, SEQ + 1)) < learned).astype(np.uint8))
    return SFTSource(path)


def test_training_with_window_mean_runs_and_is_a_run_defining_setting(tmp_path) -> None:
    sources = {"dense": sft_source(tmp_path / "dense", 30, 1, 0.9), "sparse": sft_source(tmp_path / "sparse", 30, 2, 0.1)}
    cfg = PretrainConfig(run_name="mix", total_tokens=4 * 8 * SEQ, seq_len=SEQ, batch_rows=8, micro_batch=4,
                         warmup_steps=1, eval_every=4, eval_windows=4, log_every=1, checkpoint_minutes=1e9,
                         loss_per_window=True)
    result = run_pretraining(cfg, TINY, train_sources=sources, val_sources=sources,
                             weights={"dense": 0.5, "sparse": 0.5}, store=None, workdir=tmp_path / "work",
                             device=CPU, log=lambda _: None)
    assert result["finished"]
    with pytest.raises(ValueError, match="loss_per_window"):
        check_resume_compatible(asdict(cfg), asdict(replace(cfg, loss_per_window=False)))
    # Checkpoints written before the field existed count as the token mean.
    old = {k: v for k, v in asdict(replace(cfg, loss_per_window=False)).items() if k != "loss_per_window"}
    check_resume_compatible(old, asdict(replace(cfg, loss_per_window=False)))
