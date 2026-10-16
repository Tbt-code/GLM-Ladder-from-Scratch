"""Chat template and loss mask for supervised fine-tuning.

A conversation is a list of {"role", "content"} messages, rendered as

    <|system|>...<|end|><|user|>...<|end|><|assistant|>...<|end|><|user|>...<|end|><|assistant|>...<|end|>

The model learns only what the assistant says: each assistant message's content and the
<|end|> that closes it. Role tokens, system prompts, and user messages are context; their
loss mask is False. Every segment is tokenized on its own, so the token ids at training
time and at inference time (see `prompt_ids`) are exactly the same.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from .shards import contains_special_token

ROLE_TOKENS = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>"}
END_TOKEN = "<|end|>"


def special_id(tokenizer, token: str) -> int:
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(f"{token} is not in the tokenizer")
    return token_id


def check_messages(messages: Sequence[Mapping[str, str]]) -> None:
    """Reject conversations the template cannot represent safely."""
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("a training conversation must end with an assistant message")
    for message in messages:
        if message["role"] not in ROLE_TOKENS:
            raise ValueError(f"unknown role {message['role']!r}")
        # A literal "<|user|>" inside a message would be tokenized as the real control token
        # and could fake a turn boundary (prompt injection through the data).
        if contains_special_token(message["content"]):
            raise ValueError("message content contains a special token")


def encode_conversation(tokenizer, messages: Sequence[Mapping[str, str]]) -> tuple[list[int], list[bool]]:
    """Return (ids, learn): learn[i] is True when the model should be trained to produce ids[i]."""
    check_messages(messages)
    end_id = special_id(tokenizer, END_TOKEN)
    ids: list[int] = []
    learn: list[bool] = []
    
    for message in messages:
        role = message["role"]
        content = message["content"]
        is_assistant = (role == "assistant")

        # 1. append its role token id -> learn False
        role_id = special_id(tokenizer,ROLE_TOKENS[role])
        ids.append(role_id)
        learn.append(False)

        # 2. Add tokens inside -> for assistant True, others False
        content_ids = tokenizer.encode(content).ids
        ids.extend(content_ids)
        learn.extend([is_assistant] * len(content_ids))

        # 3. Add end token (<|end|>) -> for assistant True , others False
        ids.append(end_id)
        learn.append(is_assistant)



    return ids, learn


def prompt_ids(tokenizer, messages: Sequence[Mapping[str, str]]) -> list[int]:
    """Ids that ask the model for the next assistant turn: the history, then <|assistant|>."""
    end_id = special_id(tokenizer, END_TOKEN)
    ids: list[int] = []
    for message in messages:
        if contains_special_token(message["content"]):
            raise ValueError("message content contains a special token")
        ids.append(special_id(tokenizer, ROLE_TOKENS[message["role"]]))
        ids.extend(tokenizer.encode(message["content"]).ids)
        ids.append(end_id)
    ids.append(special_id(tokenizer, ROLE_TOKENS["assistant"]))
    return ids


def from_instructurca(row: Mapping[str, str]) -> list[dict[str, str]]:
    """InstrucTurca rows have two text columns, Input and Output."""
    return [{"role": "user", "content": row["Input"].strip()},
            {"role": "assistant", "content": row["Output"].strip()}]