import numpy as np
import pytest
import torch

from glm_ladder import GLMLadderModel, ModelConfig
from glm_ladder.checkpoint import MODEL_FILE, load_checkpoint, save_checkpoint
from glm_ladder.loader import MixedLoader, PackedSource
from glm_ladder.train import lr_at, param_groups, train_step

TINY = ModelConfig(vocab_size=64, dim=32, layers=4, heads=4, expert_hidden=48, experts=4,
                   sparse_window=8, sparse_stride=8, max_sequence_length=64, linear_chunk_size=8)
CPU = torch.device("cpu")


@pytest.fixture
def shard(tmp_path):
    path = tmp_path / "train-000.bin"
    np.random.default_rng(0).integers(0, 64, size=2000).astype(np.uint16).tofile(path)
    return str(path)


def fresh(shard_path):
    model = GLMLadderModel(TINY)
    optimizer = torch.optim.AdamW(param_groups(model, 0.1), lr=0.0, betas=(0.9, 0.95))
    loader = MixedLoader({"a": PackedSource([shard_path], 16)}, {"a": 1.0}, batch_size=2, seed=0)
    return model, optimizer, loader


def run(model, optimizer, loader, start: int, steps: int) -> list[float]:
    losses = []
    for step in range(start, start + steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, max_lr=3e-3, warmup_steps=2, total_steps=10)
        losses.append(train_step(model, optimizer, [loader.next_batch()], device=CPU)["loss"])
    return losses


def test_resume_reproduces_uninterrupted_training(tmp_path, shard) -> None:
    torch.manual_seed(0)
    ref_model, ref_opt, ref_loader = fresh(shard)
    expected = run(ref_model, ref_opt, ref_loader, 0, 10)

    torch.manual_seed(0)
    model, optimizer, loader = fresh(shard)
    run(model, optimizer, loader, 0, 5)
    save_checkpoint(tmp_path / "step-000005", model=model, optimizer=optimizer,
                    trainer_state={"step": 5, "loader": loader.state_dict()})

    torch.manual_seed(999)          # a new Colab session starts from a different random state
    model2, optimizer2, loader2 = fresh(shard)
    state = load_checkpoint(tmp_path / "step-000005", model=model2, optimizer=optimizer2, device=CPU)
    loader2.load_state_dict(state["loader"])
    resumed = run(model2, optimizer2, loader2, state["step"], 5)

    assert resumed == expected[5:]                       # bit-identical on CPU
    for p, q in zip(ref_model.parameters(), model2.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_tied_weights_survive_roundtrip(tmp_path, shard) -> None:
    model, optimizer, _ = fresh(shard)
    save_checkpoint(tmp_path / "c", model=model, optimizer=optimizer, trainer_state={"step": 0})
    model2, optimizer2, _ = fresh(shard)
    load_checkpoint(tmp_path / "c", model=model2, optimizer=optimizer2, device=CPU)
    assert model2.output.weight is model2.embedding.weight
    torch.testing.assert_close(model2.embedding.weight, model.embedding.weight)


def test_corrupted_weights_are_rejected(tmp_path, shard) -> None:
    model, optimizer, _ = fresh(shard)
    save_checkpoint(tmp_path / "c", model=model, optimizer=optimizer, trainer_state={"step": 0})
    with open(tmp_path / "c" / MODEL_FILE, "ab") as handle:
        handle.write(b"\x00")
    with pytest.raises(ValueError):
        load_checkpoint(tmp_path / "c", model=model, optimizer=optimizer, device=CPU)


def test_never_overwrites_a_checkpoint(tmp_path, shard) -> None:
    model, optimizer, _ = fresh(shard)
    save_checkpoint(tmp_path / "c", model=model, optimizer=optimizer, trainer_state={"step": 0})
    with pytest.raises(FileExistsError):
        save_checkpoint(tmp_path / "c", model=model, optimizer=optimizer, trainer_state={"step": 1})