"""Math / coding slices mapped onto ForkServe's branch verbs.

These are *serving* workloads, not accuracy leaderboards. Each item has a
long shared trunk and short residuals so CoW + speculative prefill are
visible against vLLM recompute / APC.

* ``gsm8k`` — grade-school word problems, Tree-of-Thoughts / self-consistency
  fan-out (same stem, B strategy prefixes). Protocol of Cobbe et al. 2021.
* ``svamp`` / ``gsmhard`` — same ToT contract on grade-school variants
  (structure-perturbed / large-number GSM).
* ``math500`` / ``aime`` / ``amc23`` — contest math, ToT fan-out, boxed gold.
* ``game24`` — the ToT paper's math puzzle (Yao et al. 2023): four numbers
  to make 24, high branching on a tiny trunk.
* ``humaneval`` — function-completion + pytest tool-idle. Same ToT contract:
  idle fans out wrap **and** recovery; peak is the live tree at fan-out
  (shared trunk vs cloned trunks). Observation residual is the TTFT tail.
"""

from __future__ import annotations

import json
import os
from csv import DictReader
from dataclasses import dataclass
from pathlib import Path

from forkserve.adapters.templates import ToolWrappers


def benchmarks_dir() -> Path:
    """Prefer ``~/benchmarks``, then ``FORKSERVE_BENCHMARKS``, then the repo."""
    env = os.environ.get("FORKSERVE_BENCHMARKS") or os.environ.get("BENCHMARKS_DIR")
    if env:
        return Path(env).expanduser()
    home = Path.home() / "benchmarks"
    if home.is_dir():
        return home
    return Path(__file__).resolve().parents[1] / "benchmarks"


@dataclass(frozen=True)
class MathItem:
    item_id: str
    question: str
    answer: str
    n_steps: int = 0
    rank: int = 0
    solved_rate: float = -1.0


@dataclass(frozen=True)
class CodeItem:
    item_id: str
    prompt: str
    entry_point: str
    tests: str


# Original GSM8K-protocol items (short shared stem, long enough for CoW).
GSM8K_SLICE: tuple[MathItem, ...] = (
    MathItem(
        "gsm8k-01",
        "A shop sold 48 clips in April and half as many in May. "
        "How many clips were sold in April and May combined?",
        "72",
    ),
    MathItem(
        "gsm8k-02",
        "Weng earns $12 an hour. Yesterday she worked 4 hours and today she "
        "worked 2 hours more than yesterday. How much did she earn in total?",
        "120",
    ),
    MathItem(
        "gsm8k-03",
        "A school has 6 buses. Each bus holds 32 students. 18 students walk. "
        "How many students are there if every bus is full and the walkers attend?",
        "210",
    ),
    MathItem(
        "gsm8k-04",
        "Sam has 3 boxes of 24 pencils. He gives 15 pencils to each of 4 friends. "
        "How many pencils does he have left?",
        "12",
    ),
    MathItem(
        "gsm8k-05",
        "A recipe needs 3/4 cup of sugar. Maya makes 8 batches and already used "
        "2 cups. How many cups of sugar does she still need?",
        "4",
    ),
    MathItem(
        "gsm8k-06",
        "A train travels 90 km in 1.5 hours, then 60 km in 45 minutes. "
        "What is its average speed in km/h for the whole trip?",
        "60",
    ),
)

# Classic 24-game boards used in Tree-of-Thoughts evaluations.
GAME24_SLICE: tuple[MathItem, ...] = (
    MathItem("24-4-9-10-13", "Use 4, 9, 10, 13 each once with + - * / to make 24.", "24"),
    MathItem("24-1-1-1-8", "Use 1, 1, 1, 8 each once with + - * / to make 24.", "24"),
    MathItem("24-2-3-8-9", "Use 2, 3, 8, 9 each once with + - * / to make 24.", "24"),
    MathItem("24-8-8-3-3", "Use 8, 8, 3, 3 each once with + - * / to make 24.", "24"),
    MathItem("24-5-5-5-1", "Use 5, 5, 5, 1 each once with + - * / to make 24.", "24"),
    MathItem("24-6-6-6-6", "Use 6, 6, 6, 6 each once with + - * / to make 24.", "24"),
)

# HumanEval-protocol: signature + docstring trunk, hidden tests as pytest obs.
HUMANEVAL_SLICE: tuple[CodeItem, ...] = (
    CodeItem(
        "he-close",
        "from typing import List\n\n"
        "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
        '    """Return True if any two values differ by less than threshold."""\n',
        "has_close_elements",
        "assert has_close_elements([1.0, 2.0, 3.0], 0.5) is False\n"
        "assert has_close_elements([1.0, 2.8, 3.0], 0.3) is True\n",
    ),
    CodeItem(
        "he-paren",
        "from typing import List\n\n"
        "def separate_paren_groups(paren_string: str) -> List[str]:\n"
        '    """Split a string of nested () groups into balanced pieces."""\n',
        "separate_paren_groups",
        "assert separate_paren_groups('( ) (( ))') == ['()', '(())']\n",
    ),
    CodeItem(
        "he-strlen",
        "def strlen(string: str) -> int:\n"
        '    """Return the length of the string."""\n',
        "strlen",
        "assert strlen('') == 0\nassert strlen('abc') == 3\n",
    ),
    CodeItem(
        "he-max",
        "from typing import List\n\n"
        "def max_element(xs: List[int]) -> int:\n"
        '    """Return the maximum element of the list."""\n',
        "max_element",
        "assert max_element([1, 2, 3]) == 3\nassert max_element([-1, 0]) == 0\n",
    ),
    CodeItem(
        "he-filter",
        "from typing import List\n\n"
        "def filter_integers(values: List[object]) -> List[int]:\n"
        '    """Keep only the integers from a mixed list."""\n',
        "filter_integers",
        "assert filter_integers(['a', 3.1, 1, True]) == [1, True]\n",
    ),
    CodeItem(
        "he-sort",
        "from typing import List, Tuple\n\n"
        "def sort_tuple(xs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:\n"
        '    """Sort pairs by the second coordinate, then the first."""\n',
        "sort_tuple",
        "assert sort_tuple([(1, 3), (2, 1)]) == [(2, 1), (1, 3)]\n",
    ),
)

GSM8K_STRATEGIES = (
    "Translate the story into equations, then solve for the unknown.",
    "Work backwards from the asked quantity to the given numbers.",
    "Name each intermediate quantity in order and add them last.",
    "Build a small table of given facts, then apply one operation at a time.",
    "Estimate the magnitude first, then do exact arithmetic to confirm.",
    "Convert all units, cancel common factors, then compute.",
)

GAME24_STRATEGIES = (
    "Look for a pair that multiplies to 24, then fold the other two into 1.",
    "Factor 24 as 8*3 or 6*4 and hunt those subproducts.",
    "Try (a*b)-(c-d) and (a+b)*c/d systematically.",
    "Put the largest number in the denominator or as a multiplier first.",
    "Search for 24 = 3*8 = 4*6 = 2*12 and match leftover cards.",
    "Keep a running total near 24 and adjust with +1 / -1 pairs.",
)

HUMANEVAL_SKETCHES = (
    "Write a straightforward loop and early-return on the first hit.",
    "Use a builtin (sorted, any, max) and keep the body to one expression.",
    "Maintain an explicit stack / two pointers over the input.",
    "Hash seen values so each lookup is O(1), then reconstruct the answer.",
    "Recurse on a smaller suffix and combine at the return.",
    "Filter, then map; avoid mutating the caller's list.",
)


def chat_style() -> str:
    """Qwen chat by default; DeepSeek-R1 distill uses its own special tokens."""
    env = (os.environ.get("FORKSERVE_CHAT_STYLE") or "").strip()
    if env:
        return env
    model = (os.environ.get("FORKSERVE_MODEL") or "").lower()
    if "deepseek" in model or "r1-distill" in model:
        return "deepseek_r1"
    if "mistral" in model:
        return "mistral"
    if "llama" in model:
        return "llama"
    return "qwen"


def _think_tail() -> str:
    # Qwen3 honors /no_think. R1-distill always opens <think> in the template.
    # Llama / Mistral treat that token as user text and should not see it.
    if chat_style() in ("deepseek_r1", "mistral", "llama"):
        return ""
    return "\n/no_think"


def _wrappers() -> ToolWrappers:
    return ToolWrappers(style=chat_style())


def _chat(system: str, user: str) -> str:
    return ToolWrappers(style=chat_style()).chat(system, user)


def gsm8k_trunk(item: MathItem) -> str:
    system = (
        "You are a careful math tutor. Use short arithmetic. "
        "The last line of your reply must be exactly: #### <number>"
    )
    user = (
        f"{item.question}\n\n"
        "Do not list alternate plans. Finish with #### <number>."
        f"{_think_tail()}"
    )
    return _chat(system, user)


def gsm8k_thoughts(branching: int) -> list[str]:
    w = _wrappers()
    out: list[str] = []
    for i in range(branching):
        out.append(w.thought_prefix(i) + GSM8K_STRATEGIES[i % len(GSM8K_STRATEGIES)])
    return out


def game24_trunk(item: MathItem) -> str:
    system = (
        "You solve the 24 game. Use each of the four numbers exactly once "
        "with + - * / and parentheses. "
        "Finish with one line of the form: Equation: <expression> = 24"
    )
    user = (
        f"{item.question}\n"
        "Write the equation. Do not stop after the plan."
        f"{_think_tail()}"
    )
    return _chat(system, user)


def game24_thoughts(branching: int) -> list[str]:
    w = _wrappers()
    out: list[str] = []
    for i in range(branching):
        out.append(
            w.thought_prefix(i)
            + GAME24_STRATEGIES[i % len(GAME24_STRATEGIES)]
            + "\nEquation: "
        )
    return out


def humaneval_trunk(item: CodeItem) -> str:
    system = (
        "You are a coding agent. Complete the function, then run its tests "
        "with bash. Do not skip the test command."
    )
    user = (
        f"Implement `{item.entry_point}`.\n\n```python\n{item.prompt}```\n"
        "After the implementation, call bash to run pytest on the hidden tests."
    )
    return _chat(system, user)


def humaneval_sketches(branching: int) -> list[str]:
    w = _wrappers()
    out: list[str] = []
    for i in range(branching):
        out.append(w.thought_prefix(i) + HUMANEVAL_SKETCHES[i % len(HUMANEVAL_SKETCHES)])
    return out


def humaneval_react_strings(item: CodeItem) -> tuple[str, str, str]:
    w = _wrappers()
    wrap = w.observation("bash")
    recov = w.recovery("bash")
    obs = (
        f"stdout of bash(pytest -q test_{item.entry_point}.py)\n"
        f"{item.tests}"
        "1 passed in 0.02s\n"
        + w.close_observation()
    )
    return wrap, recov, obs


def unlimited(limit: int | None) -> bool:
    """``--limit 0`` / negative means the whole jsonl/csv, not a 1-item slice."""
    return limit is None or int(limit) <= 0


def take(items: tuple | list, limit: int | None) -> list:
    xs = list(items)
    if unlimited(limit):
        return xs
    return xs[: max(1, int(limit))]


def _hit_cap(n: int, limit: int | None) -> bool:
    return not unlimited(limit) and n >= int(limit)


def _gsm8k_final_answer(answer: str) -> str:
    if "####" in answer:
        return answer.rsplit("####", 1)[-1].strip().replace(",", "")
    return answer.strip()


def load_gsm8k(limit: int, split: str = "test") -> list[MathItem]:
    path = benchmarks_dir() / "gsm8k" / f"{split}.jsonl"
    if not path.is_file():
        return take(GSM8K_SLICE, limit)
    items: list[MathItem] = []
    with path.open() as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            raw = str(row["answer"])
            items.append(
                MathItem(
                    item_id=f"gsm8k-{split}-{i}",
                    question=str(row["question"]).strip(),
                    answer=_gsm8k_final_answer(raw),
                    n_steps=raw.count("<<"),
                )
            )
            if _hit_cap(len(items), limit):
                break
    return items or take(GSM8K_SLICE, limit)


def _parse_pct(raw: str) -> float:
    s = (raw or "").strip().replace("%", "")
    try:
        return float(s)
    except ValueError:
        return -1.0


def load_game24(limit: int) -> list[MathItem]:
    path = benchmarks_dir() / "game24" / "24.csv"
    if not path.is_file():
        return take(GAME24_SLICE, limit)
    items: list[MathItem] = []
    with path.open() as fh:
        for row in DictReader(fh):
            puzzle = (row.get("Puzzles") or row.get("puzzle") or "").strip()
            if not puzzle:
                continue
            nums = puzzle.replace(",", " ")
            rank_s = (row.get("Rank") or str(len(items) + 1)).strip()
            try:
                rank = int(float(rank_s))
            except ValueError:
                rank = len(items) + 1
            items.append(
                MathItem(
                    item_id=f"24-{rank}-{nums.replace(' ', '-')}",
                    question=f"Use {', '.join(nums.split())} each once with + - * / to make 24.",
                    answer="24",
                    rank=rank,
                    solved_rate=_parse_pct(str(row.get("Solved rate") or "")),
                )
            )
            if _hit_cap(len(items), limit):
                break
    return items or take(GAME24_SLICE, limit)


def load_humaneval(limit: int) -> list[CodeItem]:
    path = benchmarks_dir() / "humaneval" / "HumanEval.jsonl"
    if not path.is_file():
        gz = path.with_suffix(path.suffix + ".gz")
        if gz.is_file():
            import gzip

            text = gzip.open(gz, "rt").read()
            lines = text.splitlines()
        else:
            return take(HUMANEVAL_SLICE, limit)
    else:
        lines = path.read_text().splitlines()
    items: list[CodeItem] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        items.append(
            CodeItem(
                item_id=str(row.get("task_id", f"HumanEval/{len(items)}")),
                prompt=str(row["prompt"]),
                entry_point=str(row["entry_point"]),
                tests=str(row.get("test") or row.get("canonical_solution") or ""),
            )
        )
        if _hit_cap(len(items), limit):
            break
    return items or take(HUMANEVAL_SLICE, limit)


CONTEST_STRATEGIES = (
    "Rewrite the given conditions algebraically, then solve.",
    "Look for symmetry, an invariant, or a substitution that collapses the problem.",
    "Try a small case or count, then generalize to the asked quantity.",
    "Work backwards from the requested quantity to the given numbers.",
    "Factor, complete a square, or clear denominators before combining terms.",
    "Convert to a standard contest form (AM-GM, roots of unity, similar triangles).",
)


def math_data_dir() -> Path:
    env = os.environ.get("MATH_DATA") or os.environ.get("MATH_REASONING_DATA")
    if env:
        return Path(env).expanduser()
    home = Path.home() / "math-reasoning-datasets" / "data"
    if home.is_dir():
        return home
    return benchmarks_dir()


def numeric_math_trunk(item: MathItem) -> str:
    return gsm8k_trunk(item)


def contest_math_trunk(item: MathItem) -> str:
    system = (
        "You are a contest mathematician. Reason carefully, then give the answer. "
        "The last line must contain the final answer in \\boxed{}."
    )
    user = (
        f"{item.question}\n\n"
        "Work through the solution, then put the final answer in \\boxed{}."
        f"{_think_tail()}"
    )
    return _chat(system, user)


def contest_thoughts(branching: int) -> list[str]:
    w = _wrappers()
    out: list[str] = []
    for i in range(branching):
        out.append(w.thought_prefix(i) + CONTEST_STRATEGIES[i % len(CONTEST_STRATEGIES)])
    return out


def _read_json_or_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    text = path.read_text()
    if path.suffix == ".json":
        data = json.loads(text)
        return list(data) if isinstance(data, list) else []
    rows: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def load_svamp(limit: int) -> list[MathItem]:
    path = benchmarks_dir() / "svamp" / "test.json"
    if not path.is_file():
        alt = math_data_dir() / "SVAMP" / "test.json"
        path = alt if alt.is_file() else path
    items: list[MathItem] = []
    for i, row in enumerate(_read_json_or_jsonl(path)):
        body = str(row.get("Body") or "").strip()
        q = str(row.get("Question") or "").strip()
        if not q:
            continue
        ans = row.get("Answer")
        items.append(
            MathItem(
                item_id=str(row.get("ID") or f"svamp-{i}"),
                question=f"{body} {q}".strip(),
                answer=str(ans).strip(),
            )
        )
        if _hit_cap(len(items), limit):
            break
    return items or take(GSM8K_SLICE, limit)


def load_gsmhard(limit: int) -> list[MathItem]:
    path = benchmarks_dir() / "gsmhard" / "gsmhardv2.jsonl"
    if not path.is_file():
        alt = math_data_dir() / "gsm-hard" / "gsmhardv2.jsonl"
        path = alt if alt.is_file() else path
    items: list[MathItem] = []
    for i, row in enumerate(_read_json_or_jsonl(path)):
        q = str(row.get("input") or row.get("question") or "").strip()
        if not q:
            continue
        ans = row.get("target", row.get("answer", ""))
        items.append(MathItem(item_id=f"gsmhard-{i}", question=q, answer=str(ans).strip()))
        if _hit_cap(len(items), limit):
            break
    return items or take(GSM8K_SLICE, limit)


def load_math500(limit: int) -> list[MathItem]:
    path = benchmarks_dir() / "math500" / "test.jsonl"
    if not path.is_file():
        alt = math_data_dir() / "MATH-500" / "test.jsonl"
        path = alt if alt.is_file() else path
    items: list[MathItem] = []
    for i, row in enumerate(_read_json_or_jsonl(path)):
        q = str(row.get("problem") or row.get("question") or "").strip()
        if not q:
            continue
        level_s = str(row.get("level") or "0")
        try:
            level = int(float(level_s))
        except ValueError:
            level = 0
        items.append(
            MathItem(
                item_id=str(row.get("unique_id") or f"math500-{i}"),
                question=q,
                answer=str(row.get("answer") or "").strip(),
                n_steps=level,
            )
        )
        if _hit_cap(len(items), limit):
            break
    return items or take(GSM8K_SLICE, limit)


def _aime_item(row: dict, fallback_id: str) -> MathItem | None:
    q = str(row.get("problem") or row.get("Problem") or row.get("question") or "").strip()
    if not q:
        return None
    ans = str(row.get("answer") or row.get("Answer") or "").strip()
    iid = str(row.get("id") or row.get("ID") or fallback_id)
    return MathItem(item_id=f"aime-{iid}", question=q, answer=ans)


def load_aime(limit: int) -> list[MathItem]:
    root = benchmarks_dir() / "aime"
    paths = [
        root / "aime2024_hf.jsonl",
        root / "aime2024_problems.jsonl",
        root / "aime2025.jsonl",
        math_data_dir() / "aime_2025_opencompass" / "aime2025-I.jsonl",
        math_data_dir() / "aime_2025_opencompass" / "aime2025-II.jsonl",
    ]
    seen: set[str] = set()
    items: list[MathItem] = []
    n = 0
    for path in paths:
        for row in _read_json_or_jsonl(path):
            item = _aime_item(row, f"{path.stem}-{n}")
            if item is None:
                continue
            key = " ".join(item.question.split())
            if key in seen:
                continue
            seen.add(key)
            items.append(item)
            n += 1
            if _hit_cap(len(items), limit):
                return items
    return items or take(GSM8K_SLICE, limit)


def load_amc23(limit: int) -> list[MathItem]:
    path = benchmarks_dir() / "amc23" / "test.jsonl"
    items: list[MathItem] = []
    for i, row in enumerate(_read_json_or_jsonl(path)):
        q = str(row.get("question") or row.get("problem") or "").strip()
        if not q:
            continue
        ans = str(row.get("answer") or "").strip()
        if ans.endswith(".0"):
            try:
                ans = str(int(float(ans)))
            except ValueError:
                pass
        items.append(
            MathItem(
                item_id=f"amc23-{row.get('id', i)}",
                question=q,
                answer=ans,
            )
        )
        if _hit_cap(len(items), limit):
            break
    return items or take(GSM8K_SLICE, limit)

