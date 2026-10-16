import numpy as np
import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_model

from glm_ladder import GLMLadderModel, ModelConfig
from glm_ladder.evaluation import evaluate
from glm_ladder.loader import IGNORE_INDEX, MixedLoader, PackedSource, make_batch
from glm_ladder.pretrain import PretrainConfig, run_pretraining
from glm_ladder.runs import MetricsLog
from glm_ladder.sft_data import SFTSource

TINY = ModelConfig(vocab_size=64, dim=32, layers=4, heads=4, expert_hidden=48, experts=4,
                   sparse_window=8, sparse_stride=8, max_sequence_length=64, linear_chunk_size=8)
CPU = torch.device("cpu")
SEQ = 16


def sft_source(path, windows: int, seed: int) -> SFTSource:
    rng = np.random.default_rng(seed)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "tokens.npy", rng.integers(2, 64, size=(windows, SEQ + 1)).astype(np.uint16))
    np.save(path / "mask.npy", (rng.random((windows, SEQ + 1)) < 0.6).astype(np.uint8))
    return SFTSource(path)


def packed_source(path, tokens: int, seed: int) -> PackedSource:
    np.random.default_rng(seed).integers(0, 64, size=tokens).astype(np.uint16).tofile(path)
    return PackedSource([str(path)], SEQ)


def test_make_batch_masks_only_sources_with_a_loss_mask(tmp_path) -> None:
    sft = sft_source(tmp_path / "sft", 4, seed=0)
    plain = packed_source(tmp_path / "plain.bin", 200, seed=1)
    inputs, targets = make_batch([(sft, 2), (plain, 3)])
    np.testing.assert_array_equal(inputs[0], sft.window(2)[:-1])
    expected = sft.window(2)[1:].astype(np.int64)
    expected[~sft.learn(2)[1:]] = IGNORE_INDEX
    np.testing.assert_array_equal(targets[0], expected)
    np.testing.assert_array_equal(targets[1], plain.window(3)[1:])      # no mask -> unchanged
    assert inputs.dtype == targets.dtype == np.int64


def test_loader_yields_masked_targets(tmp_path) -> None:
    sft = sft_source(tmp_path / "sft", 10, seed=0)
    loader = MixedLoader({"chat": sft}, {"chat": 1.0}, batch_size=4, seed=0)
    inputs, targets = loader.next_batch()
    assert (targets == IGNORE_INDEX).any() and (targets != IGNORE_INDEX).any()
    learned = targets != IGNORE_INDEX
    np.testing.assert_array_equal(targets[:, :-1][learned[:, :-1]], inputs[:, 1:][learned[:, :-1]])


def test_evaluate_averages_over_learned_targets_only(tmp_path) -> None:
    torch.manual_seed(0)
    model = GLMLadderModel(TINY).eval()
    sft = sft_source(tmp_path / "sft", 5, seed=3)
    inputs, targets = make_batch([(sft, i) for i in range(5)])
    with torch.no_grad():
        logits = model(torch.as_tensor(inputs)).logits
    expected = F.cross_entropy(logits.flatten(0, 1), torch.as_tensor(targets).flatten()).item()
    got = evaluate(model, {"chat": sft}, windows=5, batch_size=2, device=CPU)["chat"]   # 2 + 2 + 1
    assert got == pytest.approx(expected, rel=1e-5)


def test_sft_run_starts_from_given_weights_and_measures_forgetting(tmp_path) -> None:
    torch.manual_seed(99)
    pretrained = GLMLadderModel(TINY)
    save_model(pretrained, str(tmp_path / "pretrained.safetensors"))
    train = {"chat": sft_source(tmp_path / "train", 40, seed=1)}
    val = {"chat": sft_source(tmp_path / "val", 6, seed=2)}
    extra = {"web": packed_source(tmp_path / "web.bin", 400, seed=4)}
    # max_lr = 0: AdamW changes nothing, so the final weights must equal the initial ones.
    cfg = PretrainConfig(run_name="sft", total_tokens=3 * 8 * SEQ, seq_len=SEQ, batch_rows=8, micro_batch=4,
                         max_lr=0.0, warmup_steps=1, eval_every=3, eval_windows=4, log_every=1,
                         checkpoint_minutes=1e9)
    result = run_pretraining(cfg, TINY, train_sources=train, val_sources=val, weights={"chat": 1.0},
                             store=None, workdir=tmp_path / "work", device=CPU, log=lambda _: None,
                             init_from=tmp_path / "pretrained.safetensors", extra_val_sources=extra)
    for (name, p), q in zip(pretrained.state_dict().items(), result["model"].state_dict().values()):
        assert torch.equal(p, q), name
    rows = MetricsLog(tmp_path / "work" / "sft" / "metrics.jsonl").rows()
    evals = [row for row in rows if "eval" in row]
    assert set(evals[-1]["eval"]) == {"chat"} and set(evals[-1]["eval_extra"]) == {"web"}
    assert all(np.isfinite(row["loss"]) for row in rows if "loss" in row)