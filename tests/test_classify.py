import math
from pathlib import Path

import numpy as np
import pytest
import torch

from glm_ladder import GLMLadderModel, ModelConfig
from glm_ladder.chat import END_TOKEN, encode_conversation, special_id
from glm_ladder.classify import (
    conversation, label_log_probs, label_token_ids, majority_baseline, metrics, review_prompts, score,
)
from glm_ladder.tokenizer import load_tokenizer

TOKENIZER = load_tokenizer(Path(__file__).resolve().parents[1] / "tokenizer" / "glm-ladder-16k.json")
TINY = ModelConfig(vocab_size=64, dim=32, layers=4, heads=4, expert_hidden=48, experts=4,
                   sparse_window=8, sparse_stride=8, max_sequence_length=64, linear_chunk_size=8)


# ----------------------------------------------------------------- prompts and labels

def test_labels_are_single_tokens_and_only_the_answer_is_learned() -> None:
    labels = label_token_ids(TOKENIZER)
    assert len(set(labels)) == 5
    ids, learn = encode_conversation(TOKENIZER, conversation("Kargo çok hızlıydı.", "tr", stars=5))
    learned = [token for token, flag in zip(ids, learn) if flag]
    assert learned == [labels[4], special_id(TOKENIZER, END_TOKEN)]     # "5" and <|end|>, nothing else


def test_prompt_is_the_training_example_without_its_answer() -> None:
    ids, _ = encode_conversation(TOKENIZER, conversation("Great food.", "en", stars=4))
    prompt = review_prompts(TOKENIZER, ["Great food."], "en")[0]
    assert ids[:len(prompt)] == prompt                  # training and scoring see the same tokens
    assert ids[len(prompt)] == label_token_ids(TOKENIZER)[3]


def test_conversation_rejects_bad_stars() -> None:
    with pytest.raises(ValueError):
        conversation("ok", "en", stars=6)


# ----------------------------------------------------------------- scoring

def test_label_log_probs_by_hand() -> None:
    logits = torch.tensor([[0.0, math.log(2), math.log(3), 0.0, math.log(4)]])   # probabilities 1:2:3:1:4 of 11
    log_probs, mass = label_log_probs(logits, [1, 2, 4])
    torch.testing.assert_close(log_probs.exp(), torch.tensor([[2 / 9, 3 / 9, 4 / 9]]))
    torch.testing.assert_close(mass, torch.tensor([9 / 11]))


def test_label_log_probs_shape_and_normalisation() -> None:
    logits = torch.randn(7, 64, generator=torch.Generator().manual_seed(0)) * 5
    log_probs, mass = label_log_probs(logits, [10, 11, 12, 13, 14])
    assert log_probs.shape == (7, 5) and mass.shape == (7,)
    torch.testing.assert_close(log_probs.exp().sum(-1), torch.ones(7))
    assert ((mass > 0) & (mass < 1)).all()


def test_scores_do_not_depend_on_batching_or_padding() -> None:
    # Prompts of different lengths share a batch and are right-padded. The real model must
    # give the same answer distribution as when each prompt is scored alone.
    torch.manual_seed(0)
    model = GLMLadderModel(TINY)
    rng = np.random.default_rng(1)
    prompts = [rng.integers(2, 64, size=length).tolist() for length in (5, 17, 9, 30, 12, 3)]
    labels = [10, 11, 12, 13, 14]
    alone, mass_alone = score(model, prompts, labels, batch_size=1)
    batched, mass_batched = score(model, prompts, labels, batch_size=4)
    np.testing.assert_allclose(batched, alone, atol=1e-5)
    np.testing.assert_allclose(mass_batched, mass_alone, atol=1e-6)


# ----------------------------------------------------------------- metrics

def test_metrics_by_hand() -> None:
    true = [1, 1, 2, 5, 5, 5]
    predicted = [1, 2, 2, 5, 4, 5]
    result = metrics(predicted, true)
    assert result["accuracy"] == pytest.approx(4 / 6)
    assert result["mae_stars"] == pytest.approx(2 / 6)
    # F1 per star: 1 -> 2/3 (p 1, r 1/2), 2 -> 2/3 (p 1/2, r 1), 3 -> 0, 4 -> 0 (p 0), 5 -> 4/5 (p 1, r 2/3)
    assert result["macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 0 + 0 + 0.8) / 5)
    assert result["confusion"][4] == [0, 0, 0, 1, 2]


def test_majority_baseline() -> None:
    result = majority_baseline([5, 5, 4, 1], [5, 4, 3, 5])
    assert result["predicts"] == 5 and result["accuracy"] == pytest.approx(0.5)
