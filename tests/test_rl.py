import math

import pytest
import torch

from glm_ladder import GLMLadderModel, ModelConfig
from glm_ladder.code_eval import CodeTask, pass_at_k
from glm_ladder.rl import (
    RLConfig, RLTrainer, pad_batch, paired_bootstrap, policy_loss, rloo_advantages, summarize_counts, token_logprobs,
)

TINY = ModelConfig(vocab_size=64, dim=32, layers=4, heads=4, expert_hidden=48, experts=4,
                   sparse_window=8, sparse_stride=8, max_sequence_length=64, linear_chunk_size=8)
CPU = torch.device("cpu")


# ----------------------------------------------------------------- advantages

def test_rloo_advantages_by_hand() -> None:
    rewards = torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
    expected = torch.tensor([[2 / 3, -2 / 3, -2 / 3, 2 / 3], [0.0, 0.0, 0.0, 0.0], [-1 / 3, -1 / 3, -1 / 3, 1.0]])
    torch.testing.assert_close(rloo_advantages(rewards), expected)


def test_rloo_advantages_sum_to_zero_per_task() -> None:
    rewards = torch.randint(0, 2, (5, 8), generator=torch.Generator().manual_seed(0)).float()
    torch.testing.assert_close(rloo_advantages(rewards).sum(dim=1), torch.zeros(5), atol=1e-6, rtol=0)


# ----------------------------------------------------------------- the loss

def test_policy_term_alone() -> None:
    logp = torch.tensor([[-1.0, -2.0, -3.0], [-0.5, -0.5, -9.0]])
    mask = torch.tensor([[False, True, True], [True, True, False]])
    advantages = torch.tensor([1.0, -2.0])
    loss, kl = policy_loss(logp, logp, mask, advantages, kl_coef=0.5)     # reference == model: KL 0
    # -(1 * (-2 - 3) + (-2) * (-0.5 - 0.5)) / 4 answer tokens = -(-5 + 2) / 4
    assert loss.item() == pytest.approx(3 / 4)
    assert kl.item() == pytest.approx(0.0)


def test_kl_term_is_positive_and_matches_the_formula() -> None:
    logp = torch.tensor([[-1.0, -2.0]])
    ref = torch.tensor([[-1.5, -1.0]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    loss, kl = policy_loss(logp, ref, mask, torch.zeros(1), kl_coef=2.0)   # advantage 0: only the KL term
    d = ref - logp
    expected = ((torch.exp(d) - d - 1).sum() / 2).item()
    assert kl.item() == pytest.approx(expected) and kl.item() > 0
    assert loss.item() == pytest.approx(2.0 * expected)


def test_gradient_raises_good_answers_and_lowers_bad_ones() -> None:
    logp = torch.tensor([[-1.0], [-1.0]], requires_grad=True)
    loss, _ = policy_loss(logp, logp.detach(), torch.ones(2, 1, dtype=torch.bool), torch.tensor([1.0, -1.0]), 0.0)
    loss.backward()
    assert logp.grad[0, 0] < 0 < logp.grad[1, 0]          # gradient descent raises logp[0], lowers logp[1]


# ----------------------------------------------------------------- batches

def test_pad_batch_marks_only_answer_targets() -> None:
    inputs, targets, mask = pad_batch([[5, 6, 7, 8], [5, 6, 9]], [2, 2])
    assert inputs.tolist() == [[5, 6, 7], [5, 6, 1]]
    assert targets.tolist() == [[6, 7, 8], [6, 9, 1]]
    assert mask.tolist() == [[False, True, True], [False, True, False]]


def test_token_logprobs_match_log_softmax() -> None:
    torch.manual_seed(0)
    model = GLMLadderModel(TINY).eval()
    inputs = torch.randint(0, 64, (2, 7))
    targets = torch.randint(0, 64, (2, 7))
    logp, _ = token_logprobs(model, inputs, targets)
    expected = torch.log_softmax(model(inputs).logits, -1).gather(-1, targets[..., None]).squeeze(-1)
    torch.testing.assert_close(logp, expected)


# ----------------------------------------------------------------- evaluation helpers

def test_summarize_counts_and_bootstrap() -> None:
    assert summarize_counts([0, 10, 5], n=10, ks=(1,))["pass@1"] == pytest.approx(0.5)
    assert summarize_counts([1], n=4, ks=(1, 4)) == {"pass@1": 0.25, "pass@4": pass_at_k(4, 1, 4)}
    result = paired_bootstrap([0.1] * 50, [0.3] * 50)
    assert result["mean"] == pytest.approx(0.2) and result["low"] == pytest.approx(0.2)


# ----------------------------------------------------------------- RL actually learns (toy reward)

def test_rl_raises_a_toy_reward() -> None:
    # Reward 1 when the first generated token is below 16 (about a quarter of the vocabulary at
    # the start). A few RLOO steps must move the policy towards it.
    torch.manual_seed(0)
    model = GLMLadderModel(TINY)
    tasks = [CodeTask(f"t{i}", "x", "", ("assert True",), "") for i in range(4)]
    prompts = {task.name: [2 + i, 9, 11] for i, task in enumerate(tasks)}
    cfg = RLConfig(run_name="toy", steps=25, prompts_per_step=4, group=4, max_new_tokens=2, lr=3e-3,
                   warmup_steps=1, kl_coef=0.0, micro_batch=8, balance_coef=0.0)
    trainer = RLTrainer(cfg, model, tokenizer=None, tasks=tasks, device=CPU, log=lambda _: None,
                        prompts_fn=lambda ts: [prompts[t.name] for t in ts], stop=lambda out: False,
                        reward_fn=lambda ts, answers: [float(a[0] < 16) for a in answers])
    history = trainer.run()
    first = sum(row["reward"] for row in history[:5]) / 5
    last = sum(row["reward"] for row in history[-5:]) / 5
    assert last > first + 0.3, (first, last)
    assert all(math.isfinite(row["grad_norm"]) for row in history)


def test_kl_is_measured_against_a_frozen_reference() -> None:
    # With a KL penalty the trainer keeps a frozen copy of the starting model; after the
    # policy has moved, the measured KL is positive. Without the penalty there is no copy.
    def run(kl_coef):
        torch.manual_seed(0)
        model = GLMLadderModel(TINY)
        tasks = [CodeTask(f"t{i}", "x", "", ("assert True",), "") for i in range(4)]
        cfg = RLConfig(run_name="kl", steps=8, prompts_per_step=4, group=4, max_new_tokens=2, lr=3e-3,
                       warmup_steps=1, kl_coef=kl_coef, micro_batch=8, balance_coef=0.0)
        trainer = RLTrainer(cfg, model, tokenizer=None, tasks=tasks, device=CPU, log=lambda _: None,
                            prompts_fn=lambda ts: [[2, 9, 11] for _ in ts], stop=lambda out: False,
                            reward_fn=lambda ts, answers: [float(a[0] < 16) for a in answers])
        history = trainer.run()
        if trainer.reference is not None:
            assert all(not p.requires_grad for p in trainer.reference.parameters())
        return trainer.reference is not None, history[-1]["kl"]

    assert run(0.5)[0] and run(0.5)[1] > 0
    assert run(0.0) == (False, 0.0)


# ----------------------------------------------------------------- resuming

def toy_trainer(steps, model, reference=None, kl_coef=0.5):
    tasks = [CodeTask(f"t{i}", "x", "", ("assert True",), "") for i in range(5)]
    cfg = RLConfig(run_name="resume", steps=steps, prompts_per_step=4, group=4, max_new_tokens=2, lr=3e-3,
                   warmup_steps=1, kl_coef=kl_coef, micro_batch=8, balance_coef=0.0)
    return RLTrainer(cfg, model, tokenizer=None, tasks=tasks, device=CPU, log=lambda _: None,
                     prompts_fn=lambda ts: [[2 + int(t.name[1:]), 9, 11] for t in ts], stop=lambda out: False,
                     reward_fn=lambda ts, answers: [float(a[0] < 16) for a in answers], reference=reference)


def fresh_model():
    torch.manual_seed(0)
    return GLMLadderModel(TINY)


def test_state_dict_is_json_and_round_trips() -> None:
    import json

    trainer = toy_trainer(3, fresh_model())
    trainer.run()
    state = json.loads(json.dumps(trainer.state_dict()))            # what a checkpoint stores
    other = toy_trainer(3, fresh_model())
    other.load_state_dict(state)
    assert other.history == trainer.history and other.order == trainer.order
    assert other.rng.random() == trainer.rng.random()                 # same random stream from here on


def test_resume_refuses_different_settings() -> None:
    trainer = toy_trainer(3, fresh_model())
    state = trainer.state_dict()
    with pytest.raises(ValueError):
        toy_trainer(4, fresh_model()).load_state_dict(state)          # steps differ


def test_interrupted_rl_run_resumes_exactly(tmp_path) -> None:
    import copy

    start = fresh_model()
    straight = toy_trainer(6, copy.deepcopy(start), reference=copy.deepcopy(start))
    straight.run()

    first = toy_trainer(6, copy.deepcopy(start), reference=copy.deepcopy(start))
    for step in range(3):                                             # session dies after step 3 ...
        first.step(step)
    first.save(tmp_path / "step-000003")
    resumed = toy_trainer(6, fresh_model(), reference=copy.deepcopy(start))   # ... new session, new model object
    resumed.restore(tmp_path / "step-000003")
    resumed.run()

    assert [r["reward"] for r in resumed.history] == [r["reward"] for r in straight.history]
    for p, q in zip(straight.model.parameters(), resumed.model.parameters()):
        assert torch.equal(p, q)


def test_checkpoint_callback_cadence() -> None:
    calls = []
    toy_trainer(7, fresh_model(), kl_coef=0.0).run(on_checkpoint=calls.append, checkpoint_every=3)
    assert calls == [3, 6, 7]
