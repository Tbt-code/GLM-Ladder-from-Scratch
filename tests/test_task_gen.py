import json
import os
import random
from pathlib import Path

import pytest

from glm_ladder.code_eval import is_safe, load_tasks, run_tests, run_tests_many
from glm_ladder.task_gen import OOD_FAMILIES, all_variants, build_task, count_greater_than, generate_splits

ROOT = Path(__file__).resolve().parents[1]
SPLITS = generate_splits()
HAND_WRITTEN = load_tasks(ROOT / "evals" / "code_tasks.jsonl")


def run_in_process(task, body: str) -> bool:
    namespace: dict = {}
    exec(f"def {task.name}({task.args}):\n{body}", namespace)
    try:
        for line in task.tests:
            exec(line, namespace)
    except AssertionError:
        return False
    return True


# ----------------------------------------------------------------- your family

@pytest.mark.parametrize("k, name", [(3, "count_greater_than_3"), (-2, "count_greater_than_minus_2")])
def test_count_greater_than_family(k, name) -> None:
    variant = count_greater_than(k)
    assert variant.family == "count_greater_than_k" and variant.name == name and variant.args == "xs"
    assert str(k) in variant.doc and "greater than" in variant.doc
    namespace: dict = {}
    exec(f"def f({variant.args}):\n{variant.body}", namespace)
    assert namespace["f"]([k - 1, k, k + 1, k + 5]) == 2
    assert namespace["f"]([]) == 0
    assert isinstance(variant.sample(random.Random(0))[0], list)


# ----------------------------------------------------------------- the task set

def test_splits_have_a_useful_size() -> None:
    assert len(SPLITS["train"]) >= 250
    assert len(SPLITS["test_id"]) >= 50
    assert len(SPLITS["test_ood"]) >= 50


def test_generation_is_deterministic() -> None:
    assert generate_splits() == SPLITS


def test_names_are_unique_and_never_hand_written() -> None:
    names = [task.name for tasks in SPLITS.values() for task in tasks]
    assert len(names) == len(set(names))
    assert not set(names) & {task.name for task in HAND_WRITTEN}


def test_held_out_families_appear_only_in_test_ood() -> None:
    family = {variant.name: variant.family for variant in all_variants()}
    assert {family[t.name] for t in SPLITS["test_ood"]} <= set(OOD_FAMILIES)
    assert not {family[t.name] for t in SPLITS["train"] + SPLITS["test_id"]} & set(OOD_FAMILIES)


def test_every_reference_passes_and_trivial_answers_fail() -> None:
    for task in (t for tasks in SPLITS.values() for t in tasks):
        assert run_in_process(task, task.reference), task.name
        assert not run_in_process(task, "    return None\n"), task.name
        assert not run_in_process(task, "    return 0\n"), task.name


def test_tests_run_in_a_subprocess_too() -> None:
    sample = random.Random(1).sample([t for tasks in SPLITS.values() for t in tasks], 12)
    jobs = [(f"def {t.name}({t.args}):\n{t.reference}", t.tests) for t in sample]
    assert run_tests_many(jobs, workers=4) == [True] * 12


def test_build_task_rejects_a_variant_without_discriminating_tests() -> None:
    from glm_ladder.task_gen import Variant, one_int
    constant = Variant("x", "always_seven", "x", "Return 7.", "    return 7\n", one_int(0, 9))
    assert build_task(constant, random.Random(0)) is None


# ----------------------------------------------------------------- running generated code

def test_run_tests_many_keeps_order_and_counts_missing_code_as_failure() -> None:
    good = ("def f(x):\n    return x + 1\n", ["assert f(1) == 2"])
    bad = ("def f(x):\n    return x\n", ["assert f(1) == 2"])
    assert run_tests_many([good, bad, (None, ["assert True"]), good], workers=2) == [True, False, False, True]


def test_new_unsafe_patterns_are_rejected() -> None:
    for code in ("import resource\n", "import ctypes\n", "from threading import Thread\n", "import importlib\n"):
        assert not is_safe(code)


@pytest.mark.skipif(os.name != "posix", reason="resource limits are set on Linux (Colab) only")
def test_memory_hog_fails_under_the_limit() -> None:
    assert not run_tests("def f():\n    x = [0] * (10 ** 9)\n    return 1\n", ["assert f() == 1"])


@pytest.mark.skipif(os.name != "posix" or not hasattr(os, "geteuid") or os.geteuid() != 0,
                    reason="the privilege drop happens only when tests run as root (Colab)")
def test_test_programs_run_as_nobody_and_cannot_write() -> None:
    # The model's code may not import os (deny-list), but the tests may: check the uid there.
    assert run_tests("def f():\n    return 1\n", ["import os", "assert os.getuid() == 65534"])
    assert not run_tests("def f():\n    return 1\n", ["import io", "io.open('x.txt', 'w').write('a')"])
