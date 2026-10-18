from dataclasses import asdict, replace

import numpy as np
import pytest
import torch
import torch.nn as nn
from safetensors.torch import load_model, save_model

from glm_ladder import GLMLadderModel, ModelConfig
from glm_ladder.loader import PackedSource
from glm_ladder.lora import LoRALinear, apply_lora, merge_lora, target_names, trainable_parameters
from glm_ladder.pretrain import PretrainConfig, check_resume_compatible, run_pretraining
from glm_ladder.runs import RunStore
from tests.fakes import FakeHub

TINY = ModelConfig(vocab_size=64, dim=32, layers=4, heads=4, expert_hidden=48, experts=4,
                   sparse_window=8, sparse_stride=8, max_sequence_length=64, linear_chunk_size=8)
CPU = torch.device("cpu")
SEQ = 16
IDS = torch.randint(0, 64, (2, 24), generator=torch.Generator().manual_seed(0))


def model_with_lora(rank: int = 4, targets: str = "all", seed: int = 0) -> tuple[GLMLadderModel, GLMLadderModel]:
    """(original, copy with LoRA) sharing the same starting weights."""
    torch.manual_seed(seed)
    original = GLMLadderModel(TINY).eval()
    adapted = GLMLadderModel(TINY).eval()
    adapted.load_state_dict(original.state_dict())
    apply_lora(adapted, rank=rank, alpha=8.0, targets=targets)
    return original, adapted


def randomize_b(model: nn.Module, seed: int = 1) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.lora_B.copy_(torch.randn(module.lora_B.shape, generator=generator) * 0.05)


# ----------------------------------------------------------------- the layer

def test_lora_linear_matches_the_formula() -> None:
    torch.manual_seed(0)
    base = nn.Linear(6, 10, bias=False)
    layer = LoRALinear(base, rank=2, alpha=4.0)
    with torch.no_grad():
        layer.lora_B.normal_()
    x = torch.randn(3, 5, 6)
    expected = x @ base.weight.T + 2.0 * (x @ layer.lora_A.T @ layer.lora_B.T)       # alpha / rank = 2
    torch.testing.assert_close(layer(x), expected)
    torch.testing.assert_close(layer.merged_weight(), base.weight + 2.0 * layer.lora_B @ layer.lora_A)


def test_b_starts_at_zero_so_the_model_is_unchanged() -> None:
    original, adapted = model_with_lora()
    with torch.no_grad():
        torch.testing.assert_close(adapted(IDS).logits, original(IDS).logits)


def test_both_factors_get_gradient_only_because_a_is_random() -> None:
    # With B = 0, dLoss/dA = 0 (A only acts through B), but dLoss/dB != 0 (it sees A x).
    # If A also started at zero, neither factor would ever move.
    _, adapted = model_with_lora()
    adapted.train()
    adapted(IDS).logits.square().mean().backward()
    layer = adapted.layers[0].block.attention.qkv
    assert layer.lora_A.grad is not None and torch.count_nonzero(layer.lora_A.grad) == 0
    assert torch.count_nonzero(layer.lora_B.grad) > 0
    assert layer.weight.grad is None                                  # frozen


# ----------------------------------------------------------------- the model

def test_which_layers_are_adapted() -> None:
    model = GLMLadderModel(TINY)
    attention, experts, every = (target_names(model, t) for t in ("attention", "experts", "all"))
    assert len(attention) == 4 * 2                                    # qkv, out per layer
    assert len(experts) == 4 * (4 + 1) * 2                            # (routed + shared) x (up, down) per layer
    assert sorted(every) == sorted(attention + experts)
    assert not any("router" in name or name == "output" for name in every)


def test_only_adapters_train_and_names_are_kept() -> None:
    original, adapted = model_with_lora(rank=4)
    layers = [m for m in adapted.modules() if isinstance(m, LoRALinear)]
    assert trainable_parameters(adapted) == sum(4 * (m.in_features + m.out_features) for m in layers)
    trainable = {name for name, p in adapted.named_parameters() if p.requires_grad}
    assert trainable and all(name.endswith(("lora_A", "lora_B")) for name in trainable)
    # Every original name is still there, so pretrained weights load before LoRA is added.
    assert set(original.state_dict()) <= set(adapted.state_dict())


def test_merge_gives_the_same_logits_and_a_plain_model() -> None:
    _, adapted = model_with_lora(rank=4)
    randomize_b(adapted)
    with torch.no_grad():
        before = adapted(IDS).logits
    assert merge_lora(adapted) == 48
    assert not any(isinstance(m, LoRALinear) for m in adapted.modules())
    with torch.no_grad():
        torch.testing.assert_close(adapted(IDS).logits, before, atol=1e-5, rtol=1e-4)
    plain = GLMLadderModel(TINY)
    plain.load_state_dict(adapted.state_dict())                        # strict: same names, no adapters


def test_lora_checkpoint_round_trip(tmp_path) -> None:
    _, adapted = model_with_lora(rank=4)
    randomize_b(adapted)
    save_model(adapted, str(tmp_path / "lora.safetensors"))
    _, again = model_with_lora(rank=4, seed=5)                         # different random weights
    load_model(again, str(tmp_path / "lora.safetensors"), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(again(IDS).logits, adapted(IDS).logits)


# ----------------------------------------------------------------- training with LoRA

CFG = PretrainConfig(run_name="lora", total_tokens=12 * 8 * SEQ, seq_len=SEQ, batch_rows=8, micro_batch=4,
                     max_lr=1e-2, warmup_steps=2, eval_every=4, eval_windows=4, log_every=1,
                     checkpoint_minutes=1e9, checkpoint_every_steps=4, weight_decay=0.0,
                     lora_rank=4, lora_alpha=8.0)


@pytest.fixture
def data(tmp_path):
    rng = np.random.default_rng(0)
    rng.integers(0, 64, 4000).astype(np.uint16).tofile(tmp_path / "train.bin")
    rng.integers(0, 64, 600).astype(np.uint16).tofile(tmp_path / "val.bin")
    return {"train_sources": {"a": PackedSource([str(tmp_path / "train.bin")], SEQ)},
            "val_sources": {"a": PackedSource([str(tmp_path / "val.bin")], SEQ)}, "weights": {"a": 1.0}}


@pytest.fixture
def pretrained(tmp_path):
    torch.manual_seed(42)
    model = GLMLadderModel(TINY)
    save_model(model, str(tmp_path / "pretrained.safetensors"))
    return model


def run(cfg, root, data, init_from, **kwargs) -> dict:
    store = RunStore(FakeHub(root / "hub"), "user/runs", cfg.run_name)
    return run_pretraining(cfg, TINY, store=store, workdir=root / "work", device=CPU, log=lambda _: None,
                           init_from=init_from, **data, **kwargs)


def test_lora_run_changes_only_the_adapters(tmp_path, data, pretrained) -> None:
    result = run(CFG, tmp_path, data, tmp_path / "pretrained.safetensors")
    trained = result["model"].state_dict()
    for name, value in pretrained.state_dict().items():
        assert torch.equal(trained[name], value), name                 # frozen weights untouched
    assert any(torch.count_nonzero(value) > 0 for name, value in trained.items() if name.endswith("lora_B"))


def test_lora_run_resumes_bit_identically(tmp_path, data, pretrained) -> None:
    init = tmp_path / "pretrained.safetensors"
    straight = run(CFG, tmp_path / "straight", data, init)
    run(CFG, tmp_path / "resumed", data, init, stop_after_steps=6)
    resumed = run(CFG, tmp_path / "resumed", data, init)
    assert resumed["finished"]
    for (name, p), q in zip(straight["model"].state_dict().items(), resumed["model"].state_dict().values()):
        assert torch.equal(p, q), name


def test_resume_check_covers_lora_settings_and_old_checkpoints() -> None:
    saved = asdict(CFG)
    with pytest.raises(ValueError, match="lora_rank"):
        check_resume_compatible(saved, asdict(replace(CFG, lora_rank=8)))
    # A checkpoint written before the LoRA fields existed counts as a full fine-tuning run.
    full = replace(CFG, lora_rank=0, lora_alpha=16.0, lora_targets="all")      # the defaults
    old = {key: value for key, value in asdict(full).items() if not key.startswith("lora")}
    check_resume_compatible(old, asdict(full))
    with pytest.raises(ValueError, match="lora_rank"):
        check_resume_compatible(old, saved)
