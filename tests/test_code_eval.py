from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from glm_ladder.code_eval import (
    CodeTask, extract_from_chat, extract_from_completion, is_safe, load_tasks, pass_at_k, run_tests,
    sample_many, summarize,
)

TASKS = load_tasks(Path(__file__).resolve().parents[1] / "evals" / "code_tasks.jsonl")
EVEN = next(task for task in TASKS if task.name == "is_even")


# ----------------------------------------------------------------- the task set itself

def test_task_set_is_well_formed() -> None:
    assert len(TASKS) == 40
    assert len({task.name for task in TASKS}) == 40
    assert all(len(task.tests) >= 2 for task in TASKS)


def test_every_reference_solution_passes_its_tests() -> None:
    # If this fails, the task is wrong, not the model.
    failing = [task.name for task in TASKS
               if not run_tests(f"def {task.name}({task.args}):\n{task.reference}", task.tests)]
    assert failing == []


def test_a_trivial_solution_fails_every_task() -> None:
    # Tests that `return None` passes would measure nothing.
    passing = [task.name for task in TASKS
               if run_tests(f"def {task.name}({task.args}):\n    return None\n", task.tests)]
    assert passing == []


# ----------------------------------------------------------------- running code

def test_wrong_code_and_syntax_errors_fail() -> None:
    assert not run_tests("def is_even(n):\n    return n % 2 == 1\n", EVEN.tests)
    assert not run_tests("def is_even(n)\n    return True\n", EVEN.tests)


def test_infinite_loop_times_out() -> None:
    assert not run_tests("def is_even(n):\n    while True:\n        pass\n", EVEN.tests, timeout=1.0)


def test_unsafe_code_is_never_run() -> None:
    code = "import os\ndef is_even(n):\n    return n % 2 == 0\n"
    assert not is_safe(code)
    assert not run_tests(code, EVEN.tests)                     # correct, but rejected


# ----------------------------------------------------------------- extracting code

def test_completion_is_cut_where_the_function_ends() -> None:
    prompt = EVEN.completion_prompt()
    continuation = "\n    return n % 2 == 0\n\n\nprint(is_even(3))\ndef other():\n    pass\n"
    code = extract_from_completion(prompt, continuation)
    assert code == prompt + "\n    return n % 2 == 0\n"
    assert run_tests(code, EVEN.tests)


def test_completion_without_a_body_fails() -> None:
    code = extract_from_completion(EVEN.completion_prompt(), "\nprint('no body')")
    assert not run_tests(code, EVEN.tests)


def test_chat_answer_with_a_fenced_block() -> None:
    answer = "Here you go:\n\n```python\ndef is_even(n):\n    return n % 2 == 0\n```\n\nUse it like this: ..."
    assert run_tests(extract_from_chat(answer, "is_even"), EVEN.tests)


def test_chat_answer_without_fences_or_cut_off() -> None:
    plain = "Sure. def is_even(n):\n    return n % 2 == 0\n"
    cut = "```python\ndef is_even(n):\n    return n % 2 == 0\n"          # no closing fence
    assert run_tests(extract_from_chat(plain, "is_even"), EVEN.tests)
    assert run_tests(extract_from_chat(cut, "is_even"), EVEN.tests)


def test_chat_answer_with_the_wrong_name_is_rejected() -> None:
    answer = "```python\ndef even_odd(n):\n    return n % 2 == 0\n```"
    assert extract_from_chat(answer, "is_even") is None


def test_prompts() -> None:
    assert EVEN.completion_prompt() == 'def is_even(n):\n    """Return True if n is even, otherwise False."""'
    assert "`is_even(n)`" in EVEN.chat_prompt()


# ----------------------------------------------------------------- pass@k

@pytest.mark.parametrize("n, c, k, expected", [
    (10, 0, 1, 0.0), (10, 10, 1, 1.0), (10, 3, 1, 0.3), (5, 2, 2, 0.7), (10, 3, 8, 1.0), (1, 1, 1, 1.0),
])
def test_pass_at_k(n, c, k, expected) -> None:
    assert pass_at_k(n, c, k) == pytest.approx(expected)


def test_summarize_averages_over_tasks() -> None:
    results = [[True] * 4, [False] * 4]
    assert summarize(results, [1, 4]) == {"pass@1": 0.5, "pass@4": 0.5}
    assert "pass@10" not in summarize(results, [10])          # only 4 samples per task


# ----------------------------------------------------------------- batched sampling

class Counter(nn.Module):
    """Fake model that always predicts (last token + 1) % vocab."""

    def __init__(self, vocab: int = 32, window: int = 8) -> None:
        super().__init__()
        self.config = SimpleNamespace(max_sequence_length=window)
        self.vocab = vocab
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, ids):
        logits = torch.full((*ids.shape, self.vocab), -1e9)
        logits.scatter_(-1, ((ids + 1) % self.vocab)[..., None], 0.0)
        return SimpleNamespace(logits=logits)


def test_sample_many_stops_each_sequence_on_its_own() -> None:
    outputs = sample_many(Counter(), [3], n=3, max_new_tokens=10, temperature=0.0, top_k=None, seed=0,
                          stop=lambda ids: 6 in ids)
    assert outputs == [[4, 5, 6]] * 3


def test_sample_many_respects_max_tokens_and_is_reproducible() -> None:
    model = Counter()
    a = sample_many(model, [1, 2], n=4, max_new_tokens=5, temperature=1.0, top_k=5, seed=7, stop=lambda ids: False)
    b = sample_many(model, [1, 2], n=4, max_new_tokens=5, temperature=1.0, top_k=5, seed=7, stop=lambda ids: False)
    assert a == b and all(len(out) == 5 for out in a)
    assert a[0] == [3, 4, 5, 6, 7]                    # one token has all the probability, even when sampling