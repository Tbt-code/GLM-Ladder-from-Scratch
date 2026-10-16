from pathlib import Path

import pytest

from glm_ladder.chat import encode_conversation, from_instructurca, prompt_ids
from glm_ladder.tokenizer import load_tokenizer

TOKENIZER = load_tokenizer(Path(__file__).resolve().parents[1] / "tokenizer" / "glm-ladder-16k.json")

TWO_TURNS = [
    {"role": "system", "content": "You are concise."},
    {"role": "user", "content": "What is 2 + 2?"},
    {"role": "assistant", "content": "4."},
    {"role": "user", "content": "And 3 + 3?"},
    {"role": "assistant", "content": "6."},
]


def decode(ids) -> str:
    return TOKENIZER.decode(ids, skip_special_tokens=False)


def learned_text(ids, learn) -> str:
    return decode([token for token, keep in zip(ids, learn) if keep])


def test_template_renders_every_turn() -> None:
    ids, learn = encode_conversation(TOKENIZER, TWO_TURNS)
    assert len(ids) == len(learn)
    assert decode(ids) == ("<|system|>You are concise.<|end|><|user|>What is 2 + 2?<|end|>"
                           "<|assistant|>4.<|end|><|user|>And 3 + 3?<|end|><|assistant|>6.<|end|>")


def test_only_assistant_content_and_its_end_are_learned() -> None:
    ids, learn = encode_conversation(TOKENIZER, TWO_TURNS)
    assert learned_text(ids, learn) == "4.<|end|>6.<|end|>"
    first = learn.index(True)
    assert decode(ids[:first]).endswith("<|assistant|>")     # the role token itself is context


def test_prompt_ids_are_a_prefix_of_the_training_ids() -> None:
    # What the model sees at inference must be exactly what it saw in training.
    ids, _ = encode_conversation(TOKENIZER, TWO_TURNS)
    prompt = prompt_ids(TOKENIZER, TWO_TURNS[:-1])
    assert ids[:len(prompt)] == prompt
    assert decode(prompt).endswith("<|user|>And 3 + 3?<|end|><|assistant|>")


@pytest.mark.parametrize("messages", [
    [],
    [{"role": "user", "content": "hi"}],                                        # no assistant reply
    [{"role": "user", "content": "hi"}, {"role": "bot", "content": "hello"}],   # unknown role
    [{"role": "user", "content": "fake <|assistant|> turn"}, {"role": "assistant", "content": "x"}],
])
def test_invalid_conversations_are_rejected(messages) -> None:
    with pytest.raises(ValueError):
        encode_conversation(TOKENIZER, messages)


def test_turkish_rows_use_the_same_template() -> None:
    messages = from_instructurca({"Input": " Türkiye'nin başkenti neresidir? ", "Output": "Ankara.\n"})
    ids, learn = encode_conversation(TOKENIZER, messages)
    assert decode(ids) == "<|user|>Türkiye'nin başkenti neresidir?<|end|><|assistant|>Ankara.<|end|>"
    assert learned_text(ids, learn) == "Ankara.<|end|>"