"""Generated coding tasks for RL: small Python functions with a reference and assert tests.

RL learns from tasks the model can sometimes solve, and the 40 hand-written tasks in
evals/code_tasks.jsonl are too few to train on and must stay untouched for evaluation. This
module generates several hundred tasks of the same style from families: a family is one
kind of function ("count the numbers greater than k") with a parameter (k = 3, 7, ...).

Tests are not written by hand: random inputs are run through the reference solution, so
the expected values are always right. A task is kept only if its tests tell the reference
apart from trivial answers (`return None`, a constant), so they really check something.

Splits (fixed by the seed):
    train     variants of the training families, minus a held-out share
    test_id   held-out variants of the training families (new parameters, same kind of task)
    test_ood  every variant of the held-out families (kinds of task never trained on)
No generated task is the same function as one of the 40 hand-written tasks; some are close
relatives (a fixed-parameter clamp vs. clamp(x, low, high)), which is why the hand-written
set is reported as "near transfer".
"""
from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

from .code_eval import CodeTask

SEED = 2026
TESTS_PER_TASK = 5
ID_SHARE = 0.25                    # share of each training family's variants held out for test_id
OOD_FAMILIES = (                   # families used only for test_ood
    "distance_to_k", "round_down_to_multiple", "keep_less_than_k", "sum_greater_than_k",
    "ends_with", "last_k_chars", "count_spaces", "last_word", "adjacent_differences",
)

Sampler = Callable[[random.Random], tuple]


@dataclass(frozen=True)
class Variant:
    family: str
    name: str
    args: str
    doc: str
    body: str                        # reference body, indented by four spaces
    sample: Sampler                  # one random input (a tuple of arguments)
    edge_cases: tuple = field(default_factory=tuple)


# ----------------------------------------------------------------- input samplers

def one_int(low: int, high: int) -> Sampler:
    return lambda rng: (rng.randint(low, high),)


def two_ints(low: int, high: int) -> Sampler:
    return lambda rng: (rng.randint(low, high), rng.randint(low, high))


def int_list(low: int = -15, high: int = 15, max_length: int = 8, min_length: int = 0) -> Sampler:
    return lambda rng: ([rng.randint(low, high) for _ in range(rng.randint(min_length, max_length))],)


WORDS = ("apple", "banana", "cherry", "rain", "river", "unit", "under", "redo", "reader", "preview",
         "intake", "export", "coder", "debug", "dance", "singing", "jumped", "quickly", "tables", "nation",
         "final", "boxes", "fixed", "player", "softly", "station", "global", "matches", "aaa", "abc")


def text(max_words: int = 4) -> Sampler:
    return lambda rng: (" ".join(rng.choice(WORDS) for _ in range(rng.randint(1, max_words))),)


def word() -> Sampler:
    return lambda rng: (rng.choice(WORDS),)


# ----------------------------------------------------------------- families

def number_families() -> list[Variant]:
    v: list[Variant] = []
    for k in range(1, 41):
        v.append(Variant("add_k", f"add_{k}", "x", f"Return x plus {k}.", f"    return x + {k}\n", one_int(-50, 50)))
        v.append(Variant("subtract_k", f"subtract_{k}", "x", f"Return x minus {k}.", f"    return x - {k}\n",
                         one_int(-50, 50)))
        v.append(Variant("distance_to_k", f"distance_to_{k}", "x", f"Return the absolute difference between x and {k}.",
                         f"    return abs(x - {k})\n", one_int(-50, 60)))
    for k in range(2, 21):
        v.append(Variant("multiply_by_k", f"multiply_by_{k}", "x", f"Return x multiplied by {k}.",
                         f"    return x * {k}\n", one_int(-30, 30)))
        v.append(Variant("remainder_by_k", f"remainder_by_{k}", "n",
                         f"Return the remainder of the non-negative integer n divided by {k}.",
                         f"    return n % {k}\n", one_int(0, 300)))
        v.append(Variant("round_down_to_multiple", f"round_down_to_multiple_of_{k}", "n",
                         f"Return the largest multiple of {k} that is not greater than the non-negative integer n.",
                         f"    return n - n % {k}\n", one_int(0, 300)))
    for k in range(3, 21):         # k = 2 would be is_even, a hand-written task
        v.append(Variant("is_divisible_by_k", f"is_divisible_by_{k}", "n",
                         f"Return True if n is divisible by {k}, otherwise False.", f"    return n % {k} == 0\n",
                         lambda rng, k=k: (rng.choice([rng.randint(-200, 200), k * rng.randint(-20, 20)]),)))
    for k in range(-10, 31):
        v.append(Variant("is_greater_than_k", f"is_greater_than_{k}".replace("-", "minus_"), "x",
                         f"Return True if x is greater than {k}, otherwise False.", f"    return x > {k}\n",
                         one_int(k - 15, k + 15), edge_cases=((k,), (k + 1,))))
    for low in (-10, -5, 0, 3, 10):
        for width in (5, 10, 20):
            high = low + width
            tag = f"{low}_{high}".replace("-", "minus_")
            v.append(Variant("is_between", f"is_between_{tag}", "x",
                             f"Return True if x is between {low} and {high} inclusive, otherwise False.",
                             f"    return {low} <= x <= {high}\n", one_int(low - 10, high + 10),
                             edge_cases=((low,), (high,), (high + 1,))))
            v.append(Variant("clamp_fixed", f"clamp_{tag}", "x",
                             f"Return x limited to the range {low} to {high}: smaller values become {low}, "
                             f"larger values become {high}.", f"    return max({low}, min({high}, x))\n",
                             one_int(low - 15, high + 15)))
    for k in (3, 4, 5):            # k = 2 would be square, a hand-written task
        v.append(Variant("power_k", f"power_{k}", "x", f"Return x raised to the power {k}.", f"    return x ** {k}\n",
                         one_int(-9, 9)))
    v += [
        Variant("single", "count_digits", "n", "Return the number of decimal digits of the non-negative integer n.",
                "    return len(str(n))\n", one_int(0, 10**6), edge_cases=((0,), (10,))),
        Variant("single", "last_digit", "n", "Return the last decimal digit of the non-negative integer n.",
                "    return n % 10\n", one_int(0, 10**5)),
        Variant("single", "sign", "x", "Return -1 if x is negative, 0 if x is zero, and 1 if x is positive.",
                "    return (x > 0) - (x < 0)\n", one_int(-20, 20), edge_cases=((0,),)),
        Variant("single", "sum_up_to", "n", "Return 1 + 2 + ... + n for a non-negative integer n (0 when n is 0).",
                "    return n * (n + 1) // 2\n", one_int(0, 200), edge_cases=((0,),)),
        Variant("single", "is_odd", "n", "Return True if n is odd, otherwise False.", "    return n % 2 == 1\n",
                one_int(-100, 100)),
        Variant("single", "min_of_two", "a, b", "Return the smaller of a and b.", "    return min(a, b)\n",
                two_ints(-50, 50)),
    ]
    return v


def list_families() -> list[Variant]:
    v: list[Variant] = []
    for k in range(-5, 11):
        tag = str(k).replace("-", "minus_")
        if k != 0:                 # k = 0 would be count_positive, a hand-written task
            v.append(count_greater_than(k))
        v.append(Variant("count_less_than_k", f"count_less_than_{tag}", "xs",
                         f"Return how many numbers in the list xs are less than {k}.",
                         f"    return sum(1 for x in xs if x < {k})\n", int_list()))
        v.append(Variant("keep_greater_than_k", f"keep_greater_than_{tag}", "xs",
                         f"Return a list of the numbers in xs that are greater than {k}, in their original order.",
                         f"    return [x for x in xs if x > {k}]\n", int_list()))
        v.append(Variant("keep_less_than_k", f"keep_less_than_{tag}", "xs",
                         f"Return a list of the numbers in xs that are less than {k}, in their original order.",
                         f"    return [x for x in xs if x < {k}]\n", int_list()))
        v.append(Variant("sum_greater_than_k", f"sum_greater_than_{tag}", "xs",
                         f"Return the sum of the numbers in the list xs that are greater than {k}.",
                         f"    return sum(x for x in xs if x > {k})\n", int_list()))
        v.append(Variant("remove_all_k", f"remove_all_{tag}", "xs",
                         f"Return a new list with every occurrence of {k} removed from xs.",
                         f"    return [x for x in xs if x != {k}]\n",
                         lambda rng, k=k: ([rng.choice([k, k, rng.randint(-8, 12)]) for _ in range(rng.randint(0, 8))],)))
    for k in range(1, 16):
        v.append(Variant("add_k_to_each", f"add_{k}_to_each", "xs", f"Return a new list with {k} added to every number in xs.",
                         f"    return [x + {k} for x in xs]\n", int_list()))
    for k in range(3, 11):         # k = 2 would be double_list, a hand-written task
        v.append(Variant("multiply_each_by_k", f"multiply_each_by_{k}", "xs",
                         f"Return a new list with every number in xs multiplied by {k}.",
                         f"    return [x * {k} for x in xs]\n", int_list()))
    for low, high in ((0, 5), (-3, 3), (1, 10), (5, 15), (-10, 0), (2, 4)):
        tag = f"{low}_{high}".replace("-", "minus_")
        v.append(Variant("count_in_range", f"count_in_range_{tag}", "xs",
                         f"Return how many numbers in xs are between {low} and {high} inclusive.",
                         f"    return sum(1 for x in xs if {low} <= x <= {high})\n", int_list()))
    v += [
        Variant("single", "count_even", "xs", "Return how many numbers in the list xs are even.",
                "    return sum(1 for x in xs if x % 2 == 0)\n", int_list()),
        Variant("single", "count_odd", "xs", "Return how many numbers in the list xs are odd.",
                "    return sum(1 for x in xs if x % 2 == 1)\n", int_list()),
        Variant("single", "keep_odd", "xs", "Return a list of the odd numbers in xs, in their original order.",
                "    return [x for x in xs if x % 2 == 1]\n", int_list()),
        Variant("single", "sum_of_squares", "xs", "Return the sum of the squares of the numbers in xs.",
                "    return sum(x * x for x in xs)\n", int_list()),
        Variant("single", "spread", "xs", "Return the largest number in the non-empty list xs minus the smallest.",
                "    return max(xs) - min(xs)\n", int_list(min_length=1)),
        Variant("single", "reverse_list", "xs", "Return a new list with the elements of xs in reverse order.",
                "    return xs[::-1]\n", int_list()),
        Variant("single", "sort_ascending", "xs", "Return a new list with the numbers of xs sorted from smallest to largest.",
                "    return sorted(xs)\n", int_list()),
        Variant("single", "sort_descending", "xs", "Return a new list with the numbers of xs sorted from largest to smallest.",
                "    return sorted(xs, reverse=True)\n", int_list()),
        Variant("single", "all_positive", "xs", "Return True if every number in xs is greater than zero, otherwise False.",
                "    return all(x > 0 for x in xs)\n", int_list(-3, 15)),
        Variant("single", "any_negative", "xs", "Return True if any number in xs is less than zero, otherwise False.",
                "    return any(x < 0 for x in xs)\n", int_list(-3, 15)),
        Variant("single", "first_or_none", "xs", "Return the first element of xs, or None if xs is empty.",
                "    return xs[0] if xs else None\n", int_list(), edge_cases=(([],),)),
        Variant("single", "last_or_none", "xs", "Return the last element of xs, or None if xs is empty.",
                "    return xs[-1] if xs else None\n", int_list(), edge_cases=(([],),)),
        Variant("adjacent_differences", "adjacent_differences", "xs",
                "Return the list of differences between neighbours: [xs[1] - xs[0], xs[2] - xs[1], ...].",
                "    return [b - a for a, b in zip(xs, xs[1:])]\n", int_list()),
    ]
    return v


def count_greater_than(k: int) -> Variant:
    """The family `count_greater_than_k`: how many numbers in a list are greater than k.

    Name: count_greater_than_{k} (a negative k is written minus_3, because '-' cannot be in a
    Python name), argument xs, doc "Return how many numbers in the list xs are greater than {k}.",
    sampler int_list().
    """
    tag = str(k).replace("-", "minus_")
    return Variant(
        "count_greater_than_k",
        f"count_greater_than_{tag}",
        "xs",
        f"Return how many numbers in the list xs are greater than {k}.",
        f"    return sum(1 for x in xs if x > {k})\n",
        int_list()
    )


def string_families() -> list[Variant]:
    v: list[Variant] = []
    for c in "abcdeilnorst":
        v.append(Variant("count_char", f"count_char_{c}", "s", f"Return how many times the letter '{c}' appears in s.",
                         f"    return s.count('{c}')\n", text()))
        v.append(Variant("remove_char", f"remove_char_{c}", "s", f"Return s with every '{c}' removed.",
                         f"    return s.replace('{c}', '')\n", text()))
    for a, b in (("a", "o"), ("e", "i"), ("r", "l"), ("s", "z"), ("n", "m"), ("t", "d"), ("o", "u"), ("i", "y")):
        v.append(Variant("replace_char", f"replace_{a}_with_{b}", "s", f"Return s with every '{a}' replaced by '{b}'.",
                         f"    return s.replace('{a}', '{b}')\n", text()))
    for prefix in ("re", "un", "in", "ex", "de", "pre", "co", "ab"):
        v.append(Variant("starts_with", f"starts_with_{prefix}", "s",
                         f"Return True if s starts with '{prefix}', otherwise False.",
                         f"    return s.startswith('{prefix}')\n", word()))
    for suffix in ("ing", "ed", "er", "ly", "es", "tion", "al", "s"):
        v.append(Variant("ends_with", f"ends_with_{suffix}", "s", f"Return True if s ends with '{suffix}', otherwise False.",
                         f"    return s.endswith('{suffix}')\n", word()))
        v.append(Variant("add_suffix", f"add_suffix_{suffix}", "s", f"Return s followed by '{suffix}'.",
                         f"    return s + '{suffix}'\n", word()))
    for k in range(2, 7):
        v.append(Variant("repeat_k", f"repeat_{k}_times", "s", f"Return s repeated {k} times, with nothing in between.",
                         f"    return s * {k}\n", word()))
    for k in range(1, 6):
        v.append(Variant("first_k_chars", f"first_{k}_chars", "s",
                         f"Return the first {k} characters of s, or all of s if it is shorter.", f"    return s[:{k}]\n", word()))
        v.append(Variant("last_k_chars", f"last_{k}_chars", "s",
                         f"Return the last {k} characters of s, or all of s if it is shorter.", f"    return s[-{k}:]\n", word()))
    v += [
        Variant("single", "to_lowercase", "s", "Return s converted to lower case.", "    return s.lower()\n",
                lambda rng: (rng.choice(WORDS).upper() + " " + rng.choice(WORDS).capitalize(),)),
        Variant("single", "remove_spaces", "s", "Return s with every space removed.", "    return s.replace(' ', '')\n", text()),
        Variant("count_spaces", "count_spaces", "s", "Return how many spaces s contains.", "    return s.count(' ')\n", text()),
        Variant("single", "first_word", "s", "Return the first word of s, where words are separated by spaces.",
                "    return s.split()[0]\n", text()),
        Variant("last_word", "last_word", "s", "Return the last word of s, where words are separated by spaces.",
                "    return s.split()[-1]\n", text()),
        Variant("single", "double_letters", "s", "Return s with every character written twice, e.g. 'ab' becomes 'aabb'.",
                "    return ''.join(c * 2 for c in s)\n", word()),
        Variant("single", "is_all_digits", "s", "Return True if s is non-empty and every character is a digit, otherwise False.",
                "    return s.isdigit()\n",
                lambda rng: (rng.choice([str(rng.randint(0, 99999)), rng.choice(WORDS), "12a4", ""]),)),
    ]
    return v


def all_variants() -> list[Variant]:
    return number_families() + list_families() + string_families()


# ----------------------------------------------------------------- tasks and splits

def reference_function(variant: Variant) -> Callable:
    namespace: dict = {}
    exec(f"def {variant.name}({variant.args}):\n{variant.body}", namespace)   # our own reference code
    return namespace[variant.name]


def build_task(variant: Variant, rng: random.Random, tests: int = TESTS_PER_TASK) -> CodeTask | None:
    """A CodeTask with tests computed by the reference, or None if no discriminating test set is found."""
    function = reference_function(variant)
    for _ in range(20):
        inputs = list(variant.edge_cases)
        seen = {repr(case) for case in inputs}
        for _ in range(200):
            if len(inputs) >= max(tests, len(variant.edge_cases)):
                break
            case = variant.sample(rng)
            if repr(case) not in seen:
                seen.add(repr(case))
                inputs.append(case)
        outputs = [function(*[_copy(a) for a in case]) for case in inputs]
        # At least two different answers: then `return None` or any constant fails some test.
        if len({repr(o) for o in outputs}) >= 2:
            lines = [f"assert {variant.name}({', '.join(repr(a) for a in case)}) == {out!r}"
                     for case, out in zip(inputs, outputs)]
            return CodeTask(variant.name, variant.args, variant.doc, tuple(lines), variant.body)
    return None


def _copy(value):
    return list(value) if isinstance(value, list) else value


def generate_splits(seed: int = SEED) -> dict[str, list[CodeTask]]:
    rng = random.Random(seed)
    splits: dict[str, list[CodeTask]] = {"train": [], "test_id": [], "test_ood": []}
    names: set[str] = set()
    for variant in all_variants():
        if variant.name in names:
            raise ValueError(f"duplicate task name {variant.name}")
        names.add(variant.name)
        task = build_task(variant, rng)
        if task is None:
            continue
        if variant.family in OOD_FAMILIES:
            splits["test_ood"].append(task)
        elif variant.family != "single" and rng.random() < ID_SHARE:
            splits["test_id"].append(task)
        else:
            splits["train"].append(task)
    return splits
