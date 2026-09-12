"""Task quality for the three serving workloads — not token overlap.

Standard metrics (what papers report):

* ``gsm8k`` — **accuracy**: extract the final number (``####`` / last numeral)
  and exact-match the gold (Cobbe et al. 2021 / lm-eval).
* ``game24`` — **success rate**: a ``+ - * /`` expression that uses each
  puzzle number once and evaluates to 24 (Yao et al. 2023 ToT).
* ``humaneval`` — **pass@1**: extracted Python plus the official prompt
  passes the hidden unit tests (Chen et al. 2021).

Serving changes (CoW, APC) must not move these scores. Token/LCP match is
the wrong instrument: two correct answers can differ in wording; two
identical wrong traces look perfect.
"""

from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

_NUM = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
_CARD = re.compile(r"\d+(?:\.\d+)?")
_FENCE = re.compile(r"```(?:python)?\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_CHAT_MARK = re.compile(r"<\|[^|]*\|>")
_DEF = re.compile(r"def\s+(\w+)\s*\(")


@dataclass(frozen=True)
class TaskScore:
    metric: str
    n: int
    score: float
    correct: list[bool] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_metric": self.metric,
            "task_n": self.n,
            "task_score": self.score,
            "task_correct": self.correct,
        }


def metric_for(workload: str) -> str:
    if workload == "gsm8k":
        return "accuracy"
    if workload == "game24":
        return "success_rate"
    if workload == "humaneval":
        return "pass_at_1"
    return "task_score"


def normalize_num(text: str) -> str:
    s = text.strip().replace(",", "").replace("$", "")
    if s.endswith("."):
        s = s[:-1]
    return s


def extract_gsm8k_answer(text: str) -> str:
    """lm-eval / Cobbe: prefer ``#### <n>``, else the last number in the text."""
    if "####" in text:
        tail = text.rsplit("####", 1)[-1]
        m = _NUM.search(tail.replace("\n", " "))
        if m:
            return normalize_num(m.group(0))
    found = _NUM.findall(text.replace("\n", " "))
    return normalize_num(found[-1]) if found else ""


def gsm8k_correct(text: str, gold: str) -> bool:
    pred = extract_gsm8k_answer(text)
    g = normalize_num(gold)
    if not pred or not g:
        return False
    try:
        return abs(float(pred) - float(g)) < 1e-6
    except ValueError:
        return pred == g


def extract_game24_nums(gold: str) -> list[int]:
    nums = [int(float(x)) for x in _NUM.findall(gold)]
    return nums[:4] if len(nums) >= 4 else nums


def extract_game24_expr(text: str) -> str:
    lines = [ln.strip() for ln in text.replace("\\n", "\n").splitlines() if ln.strip()]
    candidates = [ln for ln in lines if any(op in ln for op in "+-*/")]
    pool = candidates or lines
    best = ""
    for ln in pool:
        left = ln.split("=")[0]
        cleaned = re.sub(r"[^0-9+\-*/().\s]", "", left)
        if cleaned.count("(") != cleaned.count(")"):
            continue
        if len(cleaned) > len(best):
            best = cleaned
    return best.strip()


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_ast(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    raise ValueError("disallowed")


def game24_correct(text: str, gold: str) -> bool:
    nums = extract_game24_nums(gold)
    expr = extract_game24_expr(text)
    if not expr or not nums:
        return False
    used = [int(float(x)) for x in _CARD.findall(expr)]
    if sorted(used) != sorted(nums):
        return False
    try:
        val = _eval_ast(ast.parse(expr, mode="eval"))
    except Exception:
        return False
    return abs(val - 24.0) < 1e-6


def extract_python(text: str) -> str:
    """Pull a function body / fenced snippet out of chat or think residue."""
    raw = (text or "").replace("\\n", "\n")
    if "</think>" in raw:
        raw = raw.split("</think>")[-1]
    raw = _CHAT_MARK.sub("", raw)
    fences = _FENCE.findall(raw)
    if fences:
        return max(fences, key=len).strip("\n")
    return raw.strip("\n")


def humaneval_entry(prompt: str) -> str:
    m = _DEF.search(prompt or "")
    return m.group(1) if m else ""


def humaneval_pass(completion: str, tests: str, prompt: str = "") -> bool:
    """pass@1: official HumanEval is prompt + body + tests + check(entry)."""
    body = extract_python(completion)
    entry = humaneval_entry(prompt)
    defined = _DEF.search(body)
    if entry and defined and defined.group(1) == entry:
        src = f"{body}\n{tests}"
    else:
        src = f"{prompt}\n{body}\n{tests}"
    if entry and re.search(r"def\s+check\s*\(", tests):
        src += f"\ncheck({entry})\n"
    ns: dict[str, Any] = {"__builtins__": __builtins__}
    try:
        exec(src, ns, ns)  # noqa: S102 — HumanEval hidden-test exec
    except Exception:
        return False
    return True


def score_task(
    workload: str,
    texts: Sequence[str],
    golds: Sequence[str],
    prompts: Sequence[str] | None = None,
) -> TaskScore:
    metric = metric_for(workload)
    n = min(len(texts), len(golds))
    if n == 0:
        return TaskScore(metric, 0, 0.0, [])
    prompts = list(prompts or [])
    flags: list[bool] = []
    for i in range(n):
        text = texts[i] or ""
        gold = golds[i] or ""
        if workload == "gsm8k":
            flags.append(gsm8k_correct(text, gold))
        elif workload == "game24":
            flags.append(game24_correct(text, gold))
        elif workload == "humaneval":
            prompt = prompts[i] if i < len(prompts) else ""
            flags.append(humaneval_pass(text, gold, prompt))
        else:
            flags.append(False)
    return TaskScore(metric, n, sum(flags) / n, flags)


def annotate_quality(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score every system against gold, then record ForkServe − APC delta."""
    for r in rows:
        texts = r.get("decode_texts") or []
        golds = r.get("golds") or []
        if not texts or not golds:
            continue
        scored = score_task(
            str(r.get("workload") or ""),
            texts,
            golds,
            r.get("gold_prompts") or [],
        )
        r.update(scored.to_dict())
    idx: dict[tuple[str, int, str], dict[str, Any]] = {}
    for r in rows:
        idx[(str(r.get("system")), int(r.get("tp") or 0), str(r.get("workload")))] = r
    keys = {
        (int(r.get("tp") or 0), str(r.get("workload")))
        for r in rows
        if r.get("system") == "forkserve"
    }
    for tp, wl in keys:
        fs = idx.get(("forkserve", tp, wl))
        if fs is None or float(fs.get("task_n") or 0) <= 0:
            continue
        ref_name = "vllm_apc" if ("vllm_apc", tp, wl) in idx else "vllm_recompute"
        ref = idx.get((ref_name, tp, wl))
        if ref is None or float(ref.get("task_n") or 0) <= 0:
            continue
        ref_s = float(ref.get("task_score") or 0.0)
        fs["quality_vs"] = ref_name
        fs["quality_ref_score"] = ref_s
        fs["quality_delta"] = float(fs.get("task_score") or 0.0) - ref_s
    return rows
