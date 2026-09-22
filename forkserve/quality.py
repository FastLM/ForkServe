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
import textwrap
from dataclasses import dataclass, field
from typing import Any, Sequence

_NUM = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
_CARD = re.compile(r"\d+(?:\.\d+)?")
_FENCE = re.compile(r"```(?:python)?\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_CHAT_MARK = re.compile(
    r"<\|[^|]*\|>|"
    r"</?s>|"
    r"\[/?INST\]|"
    r"\[/?AVAILABLE_TOOLS\]|"
    r"\[/?TOOL_RESULTS\]|"
    r"\[TOOL_CALLS\]"
)
_DEF = re.compile(r"def\s+(\w+)\s*\(")
_TOPLEVEL_STOP = ("def ", "class ", "if __name__")


def strip_think(text: str) -> str:
    """R1-style traces: grade the answer after ``</think>`` when present."""
    raw = text or ""
    if "</think>" in raw:
        return raw.rsplit("</think>", 1)[-1]
    return raw


@dataclass(frozen=True)
class TaskScore:
    metric: str
    n: int
    score: float
    correct: list[bool] = field(default_factory=list)
    preds: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_metric": self.metric,
            "task_n": self.n,
            "task_score": self.score,
            "task_correct": self.correct,
            "item_preds": self.preds,
        }


def metric_for(workload: str) -> str:
    if workload in ("gsm8k", "svamp", "gsmhard", "aime", "amc23", "math500"):
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
    text = strip_think(text)
    if "####" in text:
        tail = text.rsplit("####", 1)[-1]
        m = _NUM.search(tail.replace("\n", " "))
        if m:
            return normalize_num(m.group(0))
    boxed = extract_boxed(text)
    if boxed:
        m = _NUM.search(boxed.replace("\n", " "))
        if m:
            return normalize_num(m.group(0))
        return boxed.strip()
    found = _NUM.findall(text.replace("\n", " "))
    return normalize_num(found[-1]) if found else ""


def gsm8k_correct(text: str, gold: str) -> bool:
    pred = extract_gsm8k_answer(text)
    return _answers_match(pred, gold)


def extract_boxed(text: str) -> str:
    """Last ``\\boxed{...}`` with nested braces; empty if none."""
    raw = strip_think(text)
    key = r"\boxed"
    start = raw.rfind(key)
    if start < 0:
        return ""
    i = start + len(key)
    while i < len(raw) and raw[i].isspace():
        i += 1
    if i >= len(raw):
        return ""
    if raw[i] != "{":
        rest = raw[i:].split("\n", 1)[0]
        return rest.strip().strip("$")
    depth = 0
    out: list[str] = []
    for ch in raw[i:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
        if depth:
            out.append(ch)
    return "".join(out).strip()


def _frac_to_slash(s: str) -> str:
    pat = re.compile(r"\\(?:dfrac|tfrac|frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    prev = None
    cur = s
    while prev != cur:
        prev = cur
        cur = pat.sub(r"(\1)/(\2)", cur)
    return cur


def normalize_math_ans(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1]
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "").replace("\\!", "").replace("\\;", "").replace("\\:", "")
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", s)
    s = _frac_to_slash(s)
    s = s.replace("\\pi", "pi").replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("{", "").replace("}", "").replace("$", "")
    s = s.replace("\\", "")
    s = s.replace(" ", "").replace(",", "")
    if s.endswith("."):
        s = s[:-1]
    return s.lower()


def _answers_match(pred: str, gold: str) -> bool:
    p = normalize_math_ans(pred)
    g = normalize_math_ans(gold)
    if not p or not g:
        return False
    if p == g:
        return True
    try:
        return abs(float(normalize_num(p)) - float(normalize_num(g))) < 1e-6
    except ValueError:
        return False


def math_correct(text: str, gold: str) -> bool:
    """Contest / grade-school: boxed, ####, or last number vs gold."""
    boxed = extract_boxed(text)
    if boxed and _answers_match(boxed, gold):
        return True
    return gsm8k_correct(text, gold)


def extract_game24_nums(gold: str) -> list[int]:
    nums = [int(float(x)) for x in _NUM.findall(gold)]
    return nums[:4] if len(nums) >= 4 else nums


_EXPR_CHUNK = re.compile(r"[\d.]+(?:\s*[+\-*/]\s*[\d.()]+)+")


def _clean_game24_expr(raw: str) -> str:
    left = raw.split("=")[0]
    cleaned = re.sub(r"[^0-9+\-*/().\s]", "", left)
    if cleaned.count("(") != cleaned.count(")"):
        return ""
    return cleaned.strip()


def iter_game24_exprs(text: str) -> list[str]:
    """First-to-last candidates — do not prefer the longest (noisy) line."""
    raw = strip_think(text or "").replace("\\n", "\n")
    seen: list[str] = []
    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln or not any(op in ln for op in "+-*/"):
            continue
        cleaned = _clean_game24_expr(ln)
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
        for m in _EXPR_CHUNK.finditer(ln):
            cleaned = _clean_game24_expr(m.group(0))
            if cleaned and cleaned not in seen:
                seen.append(cleaned)
    for m in _EXPR_CHUNK.finditer(raw):
        cleaned = _clean_game24_expr(m.group(0))
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return seen


def extract_game24_expr(text: str) -> str:
    exprs = iter_game24_exprs(text)
    return exprs[0] if exprs else ""


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


def _game24_ok(expr: str, nums: list[int]) -> bool:
    used = [int(float(x)) for x in _CARD.findall(expr)]
    if sorted(used) != sorted(nums):
        return False
    try:
        val = _eval_ast(ast.parse(expr, mode="eval"))
    except Exception:
        return False
    return abs(val - 24.0) < 1e-6


def game24_correct(text: str, gold: str) -> bool:
    nums = extract_game24_nums(gold)
    if not nums:
        return False
    return any(_game24_ok(expr, nums) for expr in iter_game24_exprs(text))


def quality_collapsed(workload: str, texts: Sequence[str], golds: Sequence[str]) -> bool:
    """True when every item decoded the same payload (session mixup)."""
    n = min(len(texts), len(golds))
    if n < 3:
        return False
    if workload in ("gsm8k", "svamp", "gsmhard", "aime", "amc23", "math500"):
        preds = [extract_gsm8k_answer(t or "") for t in texts[:n]]
        return len(set(normalize_num(g) for g in golds[:n])) >= 3 and len(set(preds)) == 1
    if workload == "game24":
        exprs = [extract_game24_expr(t or "") for t in texts[:n]]
        return len(set(golds[:n])) >= 3 and len(set(exprs)) == 1
    if workload == "humaneval":
        heads = [(extract_python(t or "")[:96]).strip() for t in texts[:n]]
        return len({h for h in heads if h}) == 1
    return False


def extract_python(text: str, entry: str = "") -> str:
    """Pull a function body / fenced snippet out of chat or think residue."""
    raw = (text or "").replace("\\n", "\n")
    if "</think>" in raw:
        raw = raw.split("</think>")[-1]
    raw = _CHAT_MARK.sub("", raw)
    fences = _FENCE.findall(raw)
    if fences:
        raw = fences[0]
    named = _extract_named_def(raw, entry)
    if named:
        return named
    return _trim_humaneval_body(_drop_leading_prose(raw))


def _extract_named_def(raw: str, entry: str) -> str:
    """If the model rewrote ``def <entry>``, keep that function and drop helpers."""
    if not entry:
        return ""
    m = re.search(rf"(?m)^def\s+{re.escape(entry)}\s*\(", raw or "")
    if not m:
        return ""
    lines = (raw[m.start():]).splitlines()
    kept = [lines[0]]
    for ln in lines[1:]:
        s = ln.strip()
        if s.startswith(_TOPLEVEL_STOP) and not ln[:1].isspace():
            break
        if s.startswith("print(") and not ln[:1].isspace():
            break
        kept.append(ln)
    return "\n".join(kept).strip("\n")


_CODE_HEAD = (
    "def ",
    "class ",
    "return ",
    "import ",
    "from ",
    "#",
    "if ",
    "for ",
    "while ",
    "try:",
    "with ",
    "elif ",
    "else:",
    "assert ",
    "raise ",
    "pass",
    "@",
)


def _drop_leading_prose(body: str) -> str:
    """Skip chat leftovers such as ``write code`` after ``[/INST]`` is stripped."""
    lines = (body or "").splitlines()
    start = 0
    found = False
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s:
            continue
        indented = (len(ln) - len(ln.lstrip(" "))) >= 2
        if indented or s.startswith(_CODE_HEAD):
            start = i
            found = True
            break
    return "\n".join(lines[start:] if found else lines)


def _trim_humaneval_body(body: str) -> str:
    """Keep the function body; drop print-and-chat tails that break exec."""
    lines: list[str] = []
    for line in (body or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            break
        if lines and stripped.startswith("print(") and not line[:1].isspace():
            break
        # Instruct models (especially Mistral) keep writing extra helpers
        # after the first solution; those extra ``def``s used to be kept and
        # then ``humaneval_pass`` concatenated them onto the official prompt.
        if lines and stripped.startswith(_TOPLEVEL_STOP) and not line[:1].isspace():
            break
        if (
            lines
            and stripped
            and not line[:1].isspace()
            and not stripped.startswith(("#", "import ", "from ", "return "))
            and any(x.strip().startswith("return ") or x.strip().startswith("    return ") for x in lines)
        ):
            break
        lines.append(line)
    return _normalize_humaneval_indent("\n".join(lines).strip("\n"))


def _normalize_humaneval_indent(body: str, indent: str = "    ") -> str:
    """HumanEval prompts are 4-space; Mistral often emits 0- or 3-space bodies.

    The tokenizer sets ``add_prefix_space``, so the first generated line
    frequently loses one space and ``prompt + body`` becomes IndentationError.
    """
    if not body or _DEF.search(body):
        return body
    lines = body.splitlines()
    nonempty = [(i, ln) for i, ln in enumerate(lines) if ln.strip()]
    if len(nonempty) >= 2:
        i0, ln0 = nonempty[0]
        _i1, ln1 = nonempty[1]
        ind0 = len(ln0) - len(ln0.lstrip(" "))
        ind1 = len(ln1) - len(ln1.lstrip(" "))
        if 0 <= ind0 < ind1 <= 8 and not ln0.lstrip().startswith(("def ", "class ")):
            lines[i0] = (" " * (ind1 - ind0)) + ln0
    return textwrap.indent(textwrap.dedent("\n".join(lines)), indent).strip("\n")


def humaneval_entry(prompt: str) -> str:
    m = _DEF.search(prompt or "")
    return m.group(1) if m else ""


def humaneval_pass(completion: str, tests: str, prompt: str = "") -> bool:
    """pass@1: official HumanEval is prompt + body + tests + check(entry)."""
    entry = humaneval_entry(prompt)
    body = extract_python(completion, entry)
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


def pred_for(workload: str, text: str, gold: str = "", prompt: str = "") -> str:
    if workload in ("gsm8k", "svamp", "gsmhard"):
        return extract_gsm8k_answer(text)
    if workload in ("math500", "aime", "amc23"):
        return extract_boxed(text) or extract_gsm8k_answer(text)
    if workload == "game24":
        return extract_game24_expr(text)
    if workload == "humaneval":
        body = extract_python(text, humaneval_entry(prompt))
        head = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
        return head[:96]
    return ""


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
    preds: list[str] = []
    for i in range(n):
        text = texts[i] or ""
        gold = golds[i] or ""
        prompt = prompts[i] if i < len(prompts) else ""
        if workload in ("gsm8k", "svamp", "gsmhard"):
            flags.append(gsm8k_correct(text, gold))
        elif workload in ("math500", "aime", "amc23"):
            flags.append(math_correct(text, gold))
        elif workload == "game24":
            flags.append(game24_correct(text, gold))
        elif workload == "humaneval":
            flags.append(humaneval_pass(text, gold, prompt))
        else:
            flags.append(False)
        preds.append(pred_for(workload, text, gold, prompt))
    return TaskScore(metric, n, sum(flags) / n, flags, preds)


_FAIL_TEXT = 2000


def compact_scored_row(row: dict[str, Any], *, keep_texts_n: int = 16) -> dict[str, Any]:
    """Drop bulky traces after scoring; keep short preds + failed tails."""
    texts = list(row.get("decode_texts") or [])
    golds = list(row.get("golds") or [])
    ids = list(row.get("item_ids") or [])
    correct = list(row.get("task_correct") or [])
    preds = list(row.get("item_preds") or [])
    n = min(len(texts), len(golds), len(correct) or len(texts))
    fails: list[dict[str, Any]] = []
    for i in range(n):
        if i < len(correct) and correct[i]:
            continue
        fails.append(
            {
                "i": i,
                "item_id": ids[i] if i < len(ids) else f"{row.get('workload')}-{i}",
                "pred": preds[i] if i < len(preds) else "",
                "gold": (golds[i] or "")[:240],
                "text": (texts[i] or "")[-_FAIL_TEXT:],
            }
        )
    row["failures"] = fails
    if int(row.get("task_n") or n) > keep_texts_n:
        row["decode_texts"] = []
        row["decode_ids"] = []
        if str(row.get("workload")) == "humaneval":
            row["golds"] = []
            row["gold_prompts"] = []
    return row


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
        r["quality_collapsed"] = quality_collapsed(
            str(r.get("workload") or ""),
            texts,
            golds,
        )
        compact_scored_row(r)
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
