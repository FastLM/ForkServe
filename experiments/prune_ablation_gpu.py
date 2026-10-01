"""GPU ablation: prefill admission stacked on decoding-time pruning.

Same fan-out as ``experiments/prefill_prune_bench.py``, measured with vLLM
instead of the 12 µs/token model.

Each GSM8K question is one session. The trunk is the chat prompt. The k
thoughts are the residuals: with ``mix=hopeless`` the first half are live
strategies and the second half are loop / illegal strings a text draft can
see; ``mix=clean`` is live strategies only. The winner is always thought 0.

Decoding pruners cut only after a generated prefix:

* ESC — window 5 never closes at k≤4, so every admitted child runs to the
  budget. ``esc_window=2`` is the setting where a repeated ``####`` answer
  can stop later children.
* Speculative Rejection — lowest ``floor(α k)`` non-winners (draft score,
  never the winner) stop at τ. Everyone else runs to the budget.
* DPTS — non-winners whose draft score is below 0.45 stop after the
  mini-step. A clean strategy scores ~1 and is not cut.

Prefill policies run first:

* ``base`` admits every child, so the decoder still pays its prefix.
* ``draft`` is the text heuristic (illegal / loop never start).
* ``app`` admission is selected by ``admit_mode`` (default ``score``):
  score bar, optional ``winner``, ``top_m``, or ``alpha``.

A child dropped before prefill does not pay that residual or the decoder's
decision prefix. A child both sides keep still decodes to the budget.
Temperature is 0 and EOS is ignored, so a finished child is exactly
``budget`` tokens. The trunk is warmed once per job with prefix caching on,
then the cache is reset before the next job so rows do not inherit hits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

from forkserve.adapters.templates import ToolWrappers
from forkserve.config import ForkServeConfig
from forkserve.prefill_prune import PrefillPruner
from forkserve.prune import BranchPruner, apply_admit_mode, plus_config
from forkserve.quality import extract_boxed, extract_gsm8k_answer, gsm8k_correct, math_correct

# Published checkpoints. See experiments/prefill_prune_bench.py.
DPTS_SCORE_BAR = 0.45
METHODS = ("esc", "specrej", "dpts")
POLICIES = ("base", "draft", "app")

LIVE = (
    "Translate the story into equations, then solve for the missing value.",
    "Work backwards from the asked quantity to the given numbers.",
    "Name each intermediate quantity in order and add them last.",
    "Build a small table of given facts, then apply one operation at a time.",
    "Estimate the magnitude first, then do exact arithmetic to confirm.",
    "Convert all units, cancel common factors, then compute.",
)
DEAD = (
    "****loop****loop****loop****loop****loop",
    "undefined nan junk residual that cannot be a proof",
)


def build_thoughts(k: int, mix: str, family: str = "grade") -> list[str]:
    """k residuals. Winner is index 0, always a live strategy.

    ``family=contest`` uses the contest strategy pack for the live half.
    The dead half is still the loop string alternating with illegal text,
    so a text draft can see both.
    """
    if k < 1:
        raise ValueError("k must be positive")
    if family == "contest":
        from forkserve.bench_tasks import CONTEST_STRATEGIES

        live = CONTEST_STRATEGIES
    elif family == "grade":
        live = LIVE
    else:
        raise ValueError(f"unknown family {family}")
    if mix == "clean":
        bodies = [live[i % len(live)] for i in range(k)]
    elif mix == "hopeless":
        n_dead = k // 2
        n_live = k - n_dead
        bodies = [live[i % len(live)] for i in range(n_live)]
        bodies += [DEAD[i % len(DEAD)] for i in range(n_dead)]
    else:
        raise ValueError(f"unknown mix {mix}")
    prefix = ToolWrappers().thought_prefix
    return [prefix(i) + body for i, body in enumerate(bodies)]


def _policy_config(
    policy: str,
    threshold: float,
    *,
    admit_mode: str = "score",
    keep_m: int = 0,
    admit_alpha: float = 0.5,
) -> ForkServeConfig:
    cfg = plus_config(ForkServeConfig(page_size=16))
    cfg.prune_threshold = float(threshold)
    cfg.prune_enabled = True
    if policy == "draft":
        apply_admit_mode(cfg, "score", threshold=threshold)
    elif policy == "app":
        apply_admit_mode(
            cfg,
            admit_mode,
            keep_m=keep_m,
            alpha=admit_alpha,
            threshold=threshold,
        )
    elif policy != "base":
        raise ValueError(f"unknown policy {policy}")
    return cfg


def admitted_indices(
    policy: str,
    thoughts: Sequence[str],
    threshold: float,
    *,
    admit_mode: str = "score",
    keep_m: int = 0,
    admit_alpha: float = 0.5,
) -> list[int]:
    if policy == "base":
        return list(range(len(thoughts)))
    cfg = _policy_config(
        policy,
        threshold,
        admit_mode=admit_mode,
        keep_m=keep_m,
        admit_alpha=admit_alpha,
    )
    plan = PrefillPruner(cfg, winner=0).plan(list(thoughts))
    return [d.index for d in plan.decisions if d.keep]


def _ranked_losers(thoughts: Sequence[str]) -> list[int]:
    """Non-winners, lowest draft score first. Winner is never a loser."""
    cfg = ForkServeConfig()
    cfg.prune_threshold = DPTS_SCORE_BAR
    rows = BranchPruner(cfg, winner=0).rank(list(thoughts), threshold=DPTS_SCORE_BAR)
    rows = [row for row in rows if row.index != 0]
    rows.sort(key=lambda row: (row.score, row.index))
    return [row.index for row in rows]


def drop_indices(
    method: str,
    thoughts: Sequence[str],
    *,
    alpha: float,
) -> list[int]:
    """Children the decoder cuts after its checkpoint. Winner stays."""
    ranked = _ranked_losers(thoughts)
    if method == "esc":
        return []
    if method == "specrej":
        n_drop = min(len(ranked), int(alpha * len(thoughts)))
        return ranked[:n_drop]
    if method == "dpts":
        cfg = ForkServeConfig()
        rows = BranchPruner(cfg, winner=0).rank(list(thoughts))
        return [row.index for row in rows if row.index != 0 and row.score < DPTS_SCORE_BAR]
    raise ValueError(f"unknown method {method}")


def decision_step(method: str, *, budget: int, tau: int, dpts_step: int) -> int:
    if method == "esc":
        return budget
    if method == "specrej":
        if tau > budget:
            raise ValueError(f"tau {tau} exceeds budget {budget}")
        return tau
    if method == "dpts":
        if dpts_step > budget:
            raise ValueError(f"mini-step {dpts_step} exceeds budget {budget}")
        return dpts_step
    raise ValueError(f"unknown method {method}")


def loop_indices(thoughts: Sequence[str]) -> list[int]:
    """Non-winners that are repeated loops. Illegal text that is not a loop stays.

    The GSM8K run showed the loop solved nothing, while the other low-score
    string still produced exact matches. Rank is index order, winner excluded.
    """
    return [i for i, text in enumerate(thoughts) if i != 0 and "****" in text]


def assign_mild(
    thoughts: Sequence[str],
    *,
    prefill_drop: int,
    decode_drop: int,
    budget: int,
    step: int,
    target: str = "low_score",
) -> dict[str, Any]:
    """Drop a short prefix of the loser list, one phase at a time.

    ``target=low_score`` is every non-winner under 0.45 (loops and illegal
    text). ``target=loop`` is only the repeated-loop thoughts, so the other
    low-score branch still runs to the budget. Live strategies are never
    cut. Prefill removes the first ``prefill_drop`` of that list before any
    GPU work. Decoding then stops the next ``decode_drop`` after ``step``
    tokens. Everyone else runs to ``budget``.
    """
    if step > budget:
        raise ValueError(f"mild step {step} exceeds budget {budget}")
    if target == "loop":
        losers = loop_indices(thoughts)
    elif target == "low_score":
        losers = drop_indices("dpts", thoughts, alpha=0.5)
    else:
        raise ValueError(f"unknown mild target {target}")
    pre_n = max(0, min(int(prefill_drop), len(losers)))
    skipped = losers[:pre_n]
    rest = losers[pre_n:]
    dec_n = max(0, min(int(decode_drop), len(rest)))
    shortened = rest[:dec_n]
    skip_set = set(skipped)
    short_set = set(shortened)
    admitted = [i for i in range(len(thoughts)) if i not in skip_set]
    assigned = {i: (step if i in short_set else budget) for i in admitted}
    avoided = 0
    for i in range(len(thoughts)):
        got = assigned.get(i, 0)
        avoided += budget - got
    return {
        "admitted": admitted,
        "drop": list(shortened),
        "assigned": assigned,
        "avoided_decode": avoided,
        "step": step if shortened else budget,
        "kept": sum(1 for n in assigned.values() if n == budget),
        "prefill_skipped": list(skipped),
    }


def assign_tokens(
    method: str,
    thoughts: Sequence[str],
    *,
    policy: str,
    threshold: float,
    budget: int,
    tau: int,
    dpts_step: int,
    alpha: float,
    prefill_drop: int = 0,
    decode_drop: int = 0,
    mild_step: int = 64,
    mild_target: str = "low_score",
    admit_mode: str = "score",
    keep_m: int = 0,
    admit_alpha: float = 0.5,
) -> dict[str, Any]:
    """Per-child decode length after prefill admission."""
    if method == "mild":
        return assign_mild(
            thoughts,
            prefill_drop=prefill_drop,
            decode_drop=decode_drop,
            budget=budget,
            step=mild_step,
            target=mild_target,
        )
    step = decision_step(method, budget=budget, tau=tau, dpts_step=dpts_step)
    entered = admitted_indices(
        policy,
        thoughts,
        threshold,
        admit_mode=admit_mode,
        keep_m=keep_m,
        admit_alpha=admit_alpha,
    )
    drop = drop_indices(method, thoughts, alpha=alpha)
    drop_set = set(drop)
    assigned = {
        i: (step if i in drop_set else budget)
        for i in entered
    }
    avoided = 0
    for i in range(len(thoughts)):
        if i in assigned:
            continue
        avoided += step if i in drop_set else budget
    return {
        "admitted": entered,
        "drop": drop,
        "assigned": assigned,
        "avoided_decode": avoided,
        "step": step,
        "kept": sum(1 for n in assigned.values() if n == budget),
    }


def esc_should_stop(answers: Sequence[str], window: int) -> bool:
    """True when the last ``window`` finished answers are the same number."""
    if window <= 1 or len(answers) < window:
        return False
    last = list(answers[-window:])
    return all(last) and len(set(last)) == 1


def marker_answer(text: str) -> str:
    if "####" not in (text or ""):
        return ""
    return extract_gsm8k_answer(text)


def work_key(job: dict[str, Any]) -> tuple[Any, ...]:
    """Identity of the GPU work. Unused knobs do not split a row."""
    method = job["method"]
    parts: list[Any] = [
        method,
        job["policy"],
        job["mix"],
        int(job["n"]),
        int(job["k"]),
        int(job["budget"]),
        float(job["threshold"]),
        int(job["esc_window"]) if method == "esc" else 0,
    ]
    if method == "specrej":
        parts.extend((int(job["tau"]), float(job["alpha"])))
    if method == "dpts":
        parts.append(int(job["dpts_step"]))
    if method == "mild":
        parts.extend(
            (int(job.get("prefill_drop", 0)), int(job.get("decode_drop", 0)), int(job.get("mild_step", 64)))
        )
        if job.get("mild_target", "low_score") != "low_score":
            parts.append(str(job["mild_target"]))
    if job.get("workload", "gsm8k") != "gsm8k":
        parts.append(str(job["workload"]))
    if job.get("family", "grade") != "grade":
        parts.append(str(job["family"]))
    if job.get("policy") == "app":
        mode = str(job.get("admit_mode", "score"))
        parts.append(mode)
        if mode == "top_m":
            parts.append(int(job.get("keep_m", 0)))
        if mode == "alpha":
            parts.append(float(job.get("admit_alpha", 0.5)))
    return tuple(parts)


def _job(**kwargs: Any) -> dict[str, Any]:
    base = dict(
        tag="paper",
        method="esc",
        policy="base",
        mix="hopeless",
        n=16,
        k=4,
        budget=512,
        tau=256,
        dpts_step=100,
        alpha=0.5,
        threshold=0.45,
        esc_window=0,
        admit_mode="score",
        keep_m=0,
        admit_alpha=0.5,
    )
    base.update(kwargs)
    return base


def _finalize_jobs(specs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop exact-duplicate work. Non-mild checkpoints must sit in the budget."""
    seen: set[tuple[Any, ...]] = set()
    jobs: list[dict[str, Any]] = []
    for spec in specs:
        if spec["method"] != "mild":
            decision_step(
                spec["method"],
                budget=int(spec["budget"]),
                tau=int(spec["tau"]),
                dpts_step=int(spec["dpts_step"]),
            )
        key = work_key(spec)
        if key in seen:
            continue
        seen.add(key)
        jobs.append(dict(spec))
    return jobs


def build_jobs() -> list[dict[str, Any]]:
    """Paper stack plus one-knob sweeps. Exact duplicate work is dropped."""
    specs: list[dict[str, Any]] = []
    for method in METHODS:
        for policy in POLICIES:
            specs.append(_job(tag="paper", method=method, policy=policy))
    for k in (2, 4, 8):
        for method in METHODS:
            for policy in POLICIES:
                specs.append(
                    _job(
                        tag="branching",
                        method=method,
                        policy=policy,
                        n=8,
                        k=k,
                        budget=256,
                        tau=128,
                        dpts_step=64,
                    )
                )
    for budget in (128, 256, 512):
        for method in METHODS:
            for policy in POLICIES:
                specs.append(
                    _job(
                        tag="budget",
                        method=method,
                        policy=policy,
                        n=8,
                        k=4,
                        budget=budget,
                        tau=64,
                        dpts_step=32,
                    )
                )
    for tau in (64, 128, 256):
        for policy in POLICIES:
            specs.append(
                _job(
                    tag="tau",
                    method="specrej",
                    policy=policy,
                    n=8,
                    k=4,
                    budget=512,
                    tau=tau,
                    dpts_step=64,
                )
            )
    for step in (32, 64, 100):
        for policy in POLICIES:
            specs.append(
                _job(
                    tag="dpts_step",
                    method="dpts",
                    policy=policy,
                    n=8,
                    k=4,
                    budget=512,
                    tau=128,
                    dpts_step=step,
                )
            )
    for alpha in (0.25, 0.5):
        for policy in POLICIES:
            specs.append(
                _job(
                    tag="alpha",
                    method="specrej",
                    policy=policy,
                    n=8,
                    k=8,
                    budget=256,
                    tau=128,
                    dpts_step=64,
                    alpha=alpha,
                )
            )
    for method in METHODS:
        for policy in POLICIES:
            specs.append(
                _job(
                    tag="clean",
                    method=method,
                    policy=policy,
                    mix="clean",
                    n=8,
                    k=4,
                    budget=256,
                    tau=128,
                    dpts_step=64,
                )
            )
    for mix in ("hopeless", "clean"):
        for policy in ("base", "draft"):
            specs.append(
                _job(
                    tag="esc_window",
                    method="esc",
                    policy=policy,
                    mix=mix,
                    n=8,
                    k=4,
                    budget=256,
                    tau=128,
                    dpts_step=64,
                    esc_window=2,
                )
            )
    return _finalize_jobs(specs)


def parse_ks(text: str) -> tuple[int, ...]:
    ks = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not ks or any(k < 1 for k in ks):
        raise ValueError(f"k must be a positive comma list, got {text!r}")
    return ks


def build_wide_jobs(ks: Sequence[int] = (4, 8, 16)) -> list[dict[str, Any]]:
    """Paper stack at larger fan-out. Same n / budget / checkpoints as paper.

    k=4 is the published setting. k=8 and k=16 are the same hopeless mix
    (half live strategies, half loop / illegal) so draft and DPTS cut more
    losers as the tree widens. APP keeps every residual at or above
    the score bar, so the live half still starts.
    """
    specs: list[dict[str, Any]] = []
    for k in ks:
        for method in METHODS:
            for policy in POLICIES:
                specs.append(
                    _job(
                        tag="wide",
                        method=method,
                        policy=policy,
                        n=16,
                        k=int(k),
                        budget=512,
                        tau=256,
                        dpts_step=100,
                    )
                )
    return _finalize_jobs(specs)


def label_of(job: dict[str, Any]) -> str:
    named = job.get("mild_label")
    if named:
        return str(named)
    method = str(job["method"])
    policy = str(job["policy"])
    return method if policy == "base" else f"{method}+{policy}"


def build_mild_jobs(ks: Sequence[int] = (4,)) -> list[dict[str, Any]]:
    """Each phase drops at most one hopeless thought, instead of half the fan-out.

    ``full`` decodes every child. ``prefill-1`` skips the worst before GPU.
    ``decode-1`` stops the worst after a short prefix. ``both-1`` lets each
    phase take one, so two hopeless thoughts are cut and the live ones still
    finish. ``prefill-2`` is the old draft (both hopeless thoughts skipped)
    kept as the aggressive anchor. At k>4 the same absolute cuts stay mild
    because half the tree is still hopeless.
    """
    specs: list[dict[str, Any]] = []

    def add(
        label: str,
        *,
        k: int,
        mix: str,
        prefill_drop: int,
        decode_drop: int,
        mild_step: int,
    ) -> None:
        specs.append(
            _job(
                tag="mild",
                method="mild",
                policy="base",
                mix=mix,
                n=16,
                k=int(k),
                budget=512,
                mild_label=label,
                prefill_drop=prefill_drop,
                decode_drop=decode_drop,
                mild_step=mild_step,
            )
        )

    for k in ks:
        for mix, prefix in (("hopeless", ""), ("clean", "clean/")):
            add(f"{prefix}full", k=k, mix=mix, prefill_drop=0, decode_drop=0, mild_step=64)
            add(f"{prefix}prefill-1", k=k, mix=mix, prefill_drop=1, decode_drop=0, mild_step=64)
            add(f"{prefix}decode-1@64", k=k, mix=mix, prefill_drop=0, decode_drop=1, mild_step=64)
            add(f"{prefix}decode-1@128", k=k, mix=mix, prefill_drop=0, decode_drop=1, mild_step=128)
            add(f"{prefix}both-1@64", k=k, mix=mix, prefill_drop=1, decode_drop=1, mild_step=64)
            add(f"{prefix}prefill-2", k=k, mix=mix, prefill_drop=2, decode_drop=0, mild_step=64)
    return _finalize_jobs(specs)


def build_scale_jobs() -> list[dict[str, Any]]:
    """Hard MATH-500, wider fan-out, and cuts that touch loops only.

    32 level-4 and level-5 problems, budget 1024. Half the children are
    contest strategies and half are dead text (loop, then illegal,
    alternating). ``prefill-loops`` and ``decode-loops`` touch only the
    loops, so the illegal strings still finish. ``both-loops`` splits those
    loops across the two phases. ``prefill-all-dead`` skips every draft
    score below 0.45 before prefill, including the illegal strings.
    """
    specs: list[dict[str, Any]] = []

    def add(k: int, label: str, prefill_drop: int, decode_drop: int, target: str) -> None:
        specs.append(
            _job(
                tag="scale",
                method="mild",
                policy="base",
                mix="hopeless",
                n=32,
                k=k,
                budget=1024,
                mild_label=f"k{k}/{label}",
                prefill_drop=prefill_drop,
                decode_drop=decode_drop,
                mild_step=128,
                mild_target=target,
                workload="math500",
                family="contest",
            )
        )

    for k in (4, 8, 16):
        n_loop = k // 4
        n_dead = k // 2
        add(k, "full", 0, 0, "loop")
        add(k, "prefill-loops", n_loop, 0, "loop")
        add(k, "decode-loops@128", 0, n_loop, "loop")
        if n_loop >= 2:
            half = n_loop // 2
            add(k, "both-loops@128", half, half, "loop")
        add(k, "prefill-all-dead", n_dead, 0, "low_score")
    return _finalize_jobs(specs)


def load_job_problems(jobs: Sequence[dict[str, Any]]) -> list[Any]:
    """GSM8K, or the first ``n`` MATH-500 items at level 4 or 5."""
    from forkserve.bench_tasks import load_gsm8k, load_math500

    n = max(int(job["n"]) for job in jobs)
    workloads = {str(job.get("workload", "gsm8k")) for job in jobs}
    if workloads == {"gsm8k"}:
        return load_gsm8k(n)
    if workloads != {"math500"}:
        raise RuntimeError(f"mixed workloads {sorted(workloads)}")
    hard = [item for item in load_math500(0) if int(item.n_steps or 0) >= 4]
    if len(hard) < n:
        raise RuntimeError(f"MATH-500 level>=4 has {len(hard)} items, need {n}")
    return hard[:n]


def _load_done(path: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    done: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in payload.get("rows", []):
        if row.get("ok"):
            done[work_key(row)] = row
    return done


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def _encode(tok: Any, text: str) -> list[int]:
    ids = tok.encode(text, add_special_tokens=False)
    return [int(i) for i in ids]


def _peak_tokens(
    trunk: int,
    residual: Sequence[int],
    plan: dict[str, Any],
) -> int:
    admitted: list[int] = plan["admitted"]
    assigned: dict[int, int] = plan["assigned"]
    drop = set(plan["drop"])
    step = int(plan["step"])
    at_cut = trunk + sum(residual[i] + min(step, assigned[i]) for i in admitted)
    after = trunk + sum(residual[i] + assigned[i] for i in admitted if i not in drop)
    return max(at_cut, after)


def _generate(
    llm: Any,
    sampling_cls: Any,
    prompt_cls: Any,
    seqs: Sequence[Sequence[int]],
    lengths: Sequence[int],
) -> tuple[list[str], list[int], list[int], float]:
    prompts = [prompt_cls(prompt_token_ids=list(seq)) for seq in seqs]
    params = [
        sampling_cls(
            max_tokens=int(n),
            min_tokens=int(n),
            temperature=0.0,
            ignore_eos=True,
        )
        for n in lengths
    ]
    t0 = time.perf_counter()
    outs = llm.generate(prompts, params, use_tqdm=False)
    elapsed = (time.perf_counter() - t0) * 1000.0
    texts: list[str] = []
    n_out: list[int] = []
    cached: list[int] = []
    for out in outs:
        comp = out.outputs[0]
        texts.append(comp.text or "")
        n_out.append(len(comp.token_ids))
        cached.append(int(out.num_cached_tokens or 0))
    return texts, n_out, cached, elapsed


def answer_correct(text: str, gold: str, workload: str) -> bool:
    """GSM8K needs a #### number. Contest math accepts \\boxed{} or that number."""
    if workload in ("math500", "aime", "amc23"):
        return math_correct(text, gold)
    return bool(marker_answer(text)) and gsm8k_correct(text, gold)


def has_answer_marker(text: str, workload: str) -> bool:
    if workload in ("math500", "aime", "amc23"):
        return bool(extract_boxed(text)) or "####" in (text or "")
    return "####" in (text or "")


def _score_finished(
    texts: Sequence[str],
    gold: str,
    finished: Sequence[bool],
    workload: str = "gsm8k",
) -> tuple[bool, bool]:
    """(winner match, any finished branch match).

    ``texts[0]`` is thought 0 when it was generated. Callers pass only the
    branches they actually decoded, with index 0 first if it is present.
    """
    winner_hit = False
    any_hit = False
    for text, is_done in zip(texts, finished, strict=True):
        if not is_done:
            continue
        ok = answer_correct(text, gold, workload)
        any_hit = any_hit or ok
    if texts and finished[0]:
        winner_hit = answer_correct(texts[0], gold, workload)
    return winner_hit, any_hit


def run_job(
    llm: Any,
    sampling_cls: Any,
    prompt_cls: Any,
    tok: Any,
    problems: Sequence[Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    from forkserve.bench_tasks import contest_math_trunk, gsm8k_trunk

    n = int(job["n"])
    k = int(job["k"])
    budget = int(job["budget"])
    workload = str(job.get("workload", "gsm8k"))
    thoughts = build_thoughts(k, str(job["mix"]), family=str(job.get("family", "grade")))
    plan = assign_tokens(
        str(job["method"]),
        thoughts,
        policy=str(job["policy"]),
        threshold=float(job["threshold"]),
        budget=budget,
        tau=int(job["tau"]),
        dpts_step=int(job["dpts_step"]),
        alpha=float(job["alpha"]),
        prefill_drop=int(job.get("prefill_drop", 0)),
        decode_drop=int(job.get("decode_drop", 0)),
        admit_mode=str(job.get("admit_mode", "score")),
        keep_m=int(job.get("keep_m", 0)),
        admit_alpha=float(job.get("admit_alpha", 0.5)),
        mild_step=int(job.get("mild_step", 64)),
        mild_target=str(job.get("mild_target", "low_score")),
    )
    items = list(problems)[:n]
    trunk_fn = contest_math_trunk if workload == "math500" else gsm8k_trunk
    trunks = [trunk_fn(item) for item in items]
    trunk_ids = [_encode(tok, text) for text in trunks]
    child_ids: list[list[list[int]]] = []
    prefix_ok = True
    for trunk_text, trunk in zip(trunks, trunk_ids, strict=True):
        row = []
        for thought in thoughts:
            full = _encode(tok, trunk_text + thought)
            if full[: len(trunk)] == trunk:
                row.append(full)
            else:
                # Boundary merge: keep an explicit trunk prefix so the warm
                # request still hits, and flag the row.
                prefix_ok = False
                row.append(trunk + _encode(tok, thought))
        child_ids.append(row)

    llm.reset_prefix_cache()
    warm_texts, warm_n, warm_cached, warm_ms = _generate(
        llm, sampling_cls, prompt_cls, trunk_ids, [1] * len(trunk_ids)
    )
    del warm_texts, warm_cached

    def _residuals(pi: int) -> list[int]:
        trunk_n = len(trunk_ids[pi])
        return [len(child_ids[pi][b]) - trunk_n for b in range(k)]

    peak = sum(
        _peak_tokens(len(trunk_ids[pi]), _residuals(pi), plan) for pi in range(len(items))
    )

    window = int(job["esc_window"]) if job["method"] == "esc" else 0
    decode_tokens = 0
    prompt_tokens = 0
    cached_tokens = 0
    fanout_ms = 0.0
    winner_hits = 0
    survivor_hits = 0
    branch_hits = [0] * k
    only_hits = [0] * k
    finished_branches = 0
    marker_branches = 0
    winner_hash = hashlib.sha256()
    # Branches actually launched. Windowed ESC may skip the tail.
    launched = 0

    if window > 1:
        # One branch in flight per problem, so the peak is that round, not k-wide.
        first = plan["admitted"][0]
        peak = sum(
            len(trunk_ids[pi]) + _residuals(pi)[first] + budget for pi in range(len(items))
        )
        # Classic ESC: one more child only if the window has not closed.
        # Problems stay batched; branches are rounds.
        active = set(range(len(items)))
        answers: list[list[str]] = [[] for _ in items]
        generated: list[list[tuple[str, bool]]] = [[] for _ in items]
        for branch in plan["admitted"]:
            todo = sorted(pi for pi in active)
            if not todo:
                break
            seqs = [child_ids[pi][branch] for pi in todo]
            texts, n_out, cached, elapsed = _generate(
                llm, sampling_cls, prompt_cls, seqs, [budget] * len(todo)
            )
            fanout_ms += elapsed
            for pi, text, ntok, hit in zip(todo, texts, n_out, cached, strict=True):
                decode_tokens += ntok
                prompt_tokens += len(seqs[todo.index(pi)])
                cached_tokens += hit
                answers[pi].append(marker_answer(text))
                generated[pi].append((text, True))
                finished_branches += 1
                if has_answer_marker(text, workload):
                    marker_branches += 1
                if esc_should_stop(answers[pi], window):
                    active.discard(pi)
        for pi, item in enumerate(items):
            texts = [text for text, _ in generated[pi]]
            flags = [flag for _, flag in generated[pi]]
            # Thought 0 is first only when it was admitted, which it is.
            w_hit, s_hit = _score_finished(texts, item.answer, flags, workload)
            winner_hits += int(w_hit)
            survivor_hits += int(s_hit)
            if texts:
                winner_hash.update(texts[0].encode())
    else:
        seqs: list[list[int]] = []
        lengths: list[int] = []
        owners: list[tuple[int, int]] = []
        for pi in range(len(items)):
            for branch, ntok in plan["assigned"].items():
                seqs.append(child_ids[pi][branch])
                lengths.append(int(ntok))
                owners.append((pi, branch))
        by_problem: list[list[tuple[int, str, bool]]] = [[] for _ in items]
        if seqs:
            texts, n_out, cached, elapsed = _generate(
                llm, sampling_cls, prompt_cls, seqs, lengths
            )
            fanout_ms += elapsed
            for (pi, branch), text, ntok, hit, seq, n_req in zip(
                owners, texts, n_out, cached, seqs, lengths, strict=True
            ):
                decode_tokens += ntok
                prompt_tokens += len(seq)
                cached_tokens += hit
                done = int(n_req) == budget
                by_problem[pi].append((branch, text, done))
                if done:
                    finished_branches += 1
                    if has_answer_marker(text, workload):
                        marker_branches += 1
        for pi, item in enumerate(items):
            rows = sorted(by_problem[pi], key=lambda row: row[0])
            texts = [text for _, text, _ in rows]
            flags = [flag for _, _, flag in rows]
            # Winner text is branch 0. Move it to index 0 for the scorer
            # only when it was the first generated branch, which it is
            # because admitted indices come back in plan order starting at 0.
            w_hit, s_hit = _score_finished(texts, item.answer, flags, workload)
            winner_hits += int(w_hit)
            survivor_hits += int(s_hit)
            hit_branches: list[int] = []
            for branch, text, done in rows:
                if branch == 0:
                    winner_hash.update(text.encode())
                ok = bool(done) and answer_correct(text, item.answer, workload)
                if ok:
                    branch_hits[branch] += 1
                    hit_branches.append(branch)
            if len(hit_branches) == 1:
                only_hits[hit_branches[0]] += 1

    prefill_avoided = len(items) * int(plan["avoided_decode"])
    if window > 1:
        # A round that never starts saves a full budget, not just a prefix.
        skipped = len(items) * len(plan["admitted"]) - finished_branches
        avoided = prefill_avoided + skipped * budget
        issued = int(decode_tokens)
    else:
        issued = len(items) * sum(plan["assigned"].values())
        avoided = prefill_avoided
    row = dict(job)
    row.update(
        {
            "ok": True,
            "label": label_of(job),
            "admitted": len(plan["admitted"]),
            "kept": int(plan["kept"]),
            "dropped_at_cut": len(set(plan["drop"]) & set(plan["admitted"])),
            "step": int(plan["step"]),
            "decode_tokens": int(decode_tokens),
            "issued_decode_tokens": int(issued),
            "avoided_decode": int(avoided),
            "warm_decode_tokens": int(sum(warm_n)),
            "prompt_tokens": int(prompt_tokens),
            "cached_tokens": int(cached_tokens),
            "computed_prefill_tokens": int(max(0, prompt_tokens - cached_tokens)),
            "peak_kv_tokens": int(peak),
            "trunk_warm_ms": round(warm_ms, 2),
            "fanout_ms": round(fanout_ms, 2),
            "e2e_ms": round(warm_ms + fanout_ms, 2),
            "winner_correct": int(winner_hits),
            "survivor_correct": int(survivor_hits),
            "branch_correct": branch_hits,
            "branch_unique": only_hits,
            "finished_branches": int(finished_branches),
            "marker_branches": int(marker_branches),
            "prefix_cache_aligned": bool(prefix_ok),
            "winner_hash": winner_hash.hexdigest()[:12],
        }
    )
    return row


def _shard_jobs(jobs: Sequence[dict[str, Any]], shard: int, shards: int) -> list[dict[str, Any]]:
    return [job for i, job in enumerate(jobs) if i % shards == shard]


def run_gpu(args: argparse.Namespace) -> int:
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("FORKSERVE_MODEL", args.model)
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    jobs = select_jobs(args)
    if args.smoke:
        if args.scale:
            picked = list(jobs)
        else:
            picked = [job for job in jobs if job["tag"] in ("paper", "wide")][:3]
            if not picked:
                picked = list(jobs)[:3]
            extra = next(
                (job for job in jobs if job["tag"] == "esc_window" and job["mix"] == "hopeless"),
                None,
            )
            if extra:
                picked.append(extra)
        for job in picked:
            job["n"] = 1
            job["budget"] = 32
            job["tau"] = min(int(job["tau"]), 16)
            job["dpts_step"] = min(int(job["dpts_step"]), 8)
            job["mild_step"] = min(int(job.get("mild_step", 8)), 8)
            job["tag"] = "smoke"
        jobs = picked
    else:
        jobs = _shard_jobs(jobs, args.shard, args.shards)
    out = Path(args.out)
    done = _load_done(out)
    payload: dict[str, Any] = {
        "model": args.model,
        "tp": args.tp,
        "shard": args.shard,
        "shards": 1 if args.smoke else args.shards,
        "smoke": bool(args.smoke),
        "rows": list(done.values()),
    }
    pending = [job for job in jobs if work_key(job) not in done]
    print(
        f"jobs {len(jobs)} pending {len(pending)} already {len(done)} -> {out}",
        flush=True,
    )
    if not pending:
        _write(out, payload)
        return 0

    problems = load_job_problems(jobs)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_util,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=args.max_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
    )
    tok = llm.get_tokenizer()
    # One short batch so the first measured job is not the compile step.
    _generate(llm, SamplingParams, TokensPrompt, [[1, 2, 3, 4] * 16], [8])
    llm.reset_prefix_cache()

    for index, job in enumerate(pending, start=1):
        t0 = time.perf_counter()
        try:
            row = run_job(llm, SamplingParams, TokensPrompt, tok, problems, job)
        except Exception as exc:  # noqa: BLE001 — keep the shard going
            row = dict(job)
            row.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            print(f"FAIL {label_of(job)} {job['tag']}: {row['error']}", flush=True)
        payload["rows"] = [existing for existing in payload["rows"] if work_key(existing) != work_key(row)]
        payload["rows"].append(row)
        _write(out, payload)
        if row.get("ok"):
            print(
                f"[{index}/{len(pending)}] {row['label']:<16} {row['tag']:<12} "
                f"k={row['k']} D={row['budget']} e2e={row['e2e_ms']:.0f}ms "
                f"decode={row['decode_tokens']} avoid={row['avoided_decode']} "
                f"W={row['winner_correct']}/{row['n']} any={row['survivor_correct']}/{row['n']} "
                f"({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )
    print(f"wrote {out}", flush=True)
    return 0


def load_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for path in paths:
        if path.is_dir():
            files = sorted(path.glob("*.json"))
        else:
            files = [path]
        for file in files:
            if file.name.endswith(".tmp"):
                continue
            try:
                payload = json.loads(file.read_text())
            except json.JSONDecodeError:
                continue
            for row in payload.get("rows", []):
                if not row.get("ok"):
                    continue
                key = work_key(row)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
    return rows


def _fmt_table(headers: Sequence[str], data: Sequence[Sequence[Any]]) -> str:
    cols = [list(headers)] + [[("" if cell is None else str(cell)) for cell in line] for line in data]
    widths = [max(len(row[i]) for row in cols) for i in range(len(headers))]
    lines = []
    for r, row in enumerate(cols):
        lines.append("  ".join(cell.rjust(widths[i]) for i, cell in enumerate(row)))
        if r == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


def _pick(rows: Sequence[dict[str, Any]], **want: Any) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if all(row.get(key) == value for key, value in want.items()):
            out.append(row)
    return out


def render_report(rows: Sequence[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append(f"rows {len(rows)}")
    if not rows:
        return "\n".join(lines)

    def emit(title: str, group: Sequence[dict[str, Any]], cols: Sequence[str]) -> None:
        lines.append("")
        lines.append(title)
        ordered = sorted(group, key=lambda row: (str(row.get("tag")), label_of(row), int(row.get("k", 0)), int(row.get("budget", 0))))
        data = []
        for row in ordered:
            cells: list[Any] = []
            for col in cols:
                if col == "label":
                    cells.append(label_of(row))
                elif col == "winner_acc":
                    n = int(row["n"])
                    cells.append(f"{int(row['winner_correct'])}/{n}")
                elif col == "surv_acc":
                    n = int(row["n"])
                    cells.append(f"{int(row['survivor_correct'])}/{n}")
                elif col in ("e2e_ms", "fanout_ms", "trunk_warm_ms"):
                    cells.append(f"{float(row[col]):.0f}")
                else:
                    cells.append(row.get(col, ""))
            data.append(cells)
        lines.append(_fmt_table(cols, data))

    paper = _pick(rows, tag="paper") or [
        row
        for row in rows
        if row.get("mix") == "hopeless"
        and int(row.get("n", 0)) == 16
        and int(row.get("k", 0)) == 4
        and int(row.get("budget", 0)) == 512
        and int(row.get("esc_window") or 0) == 0
        and (
            (row["method"] == "esc")
            or (row["method"] == "specrej" and int(row["tau"]) == 256)
            or (row["method"] == "dpts" and int(row["dpts_step"]) == 100)
        )
    ]
    if paper:
        emit(
        "paper  n=16 k=4 budget=512  tau=256  dpts=100  hopeless",
        paper,
        (
            "label",
            "admitted",
            "kept",
            "decode_tokens",
            "avoided_decode",
            "computed_prefill_tokens",
            "cached_tokens",
            "peak_kv_tokens",
            "fanout_ms",
            "e2e_ms",
            "winner_acc",
            "surv_acc",
        ),
    )
    wide_cols = (
        "label",
        "k",
        "admitted",
        "kept",
        "decode_tokens",
        "avoided_decode",
        "computed_prefill_tokens",
        "peak_kv_tokens",
        "e2e_ms",
        "winner_acc",
        "surv_acc",
    )
    wide = _pick(rows, tag="wide")
    if wide:
        for k in sorted({int(row["k"]) for row in wide}):
            emit(
                f"wide  n=16 k={k} budget=512  tau=256  dpts=100  hopeless",
                [row for row in wide if int(row["k"]) == k],
                wide_cols,
            )
    scale = _pick(rows, tag="scale")
    if scale:
        emit(
            "scale  MATH-500 level>=4  n=32  budget=1024  loops only, except prefill-all-dead",
            scale,
            (
                "label",
                "k",
                "admitted",
                "kept",
                "decode_tokens",
                "avoided_decode",
                "peak_kv_tokens",
                "e2e_ms",
                "winner_acc",
                "surv_acc",
            ),
        )
    mild = _pick(rows, tag="mild")
    if mild:
        for k in sorted({int(row["k"]) for row in mild}):
            emit(
                f"mild  n=16 k={k} budget=512  drop at most one hopeless thought per phase",
                [row for row in mild if int(row["k"]) == k],
                (
                    "label",
                    "k",
                    "admitted",
                    "kept",
                    "decode_tokens",
                    "avoided_decode",
                    "peak_kv_tokens",
                    "e2e_ms",
                    "winner_acc",
                    "surv_acc",
                ),
            )
    return "\n".join(lines)


def _stamp_admit(jobs: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    mode = str(getattr(args, "admit_mode", "score") or "score")
    keep_m = int(getattr(args, "keep_m", 0) or 0)
    alpha = float(getattr(args, "admit_alpha", 0.5) or 0.5)
    for job in jobs:
        if job.get("policy") != "app":
            continue
        job["admit_mode"] = mode
        job["keep_m"] = keep_m
        job["admit_alpha"] = alpha
    return jobs


def select_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    ks = parse_ks(args.ks) if args.ks else ((4, 8, 16) if args.wide else (4,))
    if args.scale:
        jobs = build_scale_jobs()
    elif args.mild:
        jobs = build_mild_jobs(ks=ks if (args.wide or args.ks) else (4,))
    elif args.wide or args.ks:
        jobs = build_wide_jobs(ks=ks)
    else:
        jobs = build_jobs()
    return _stamp_admit(jobs, args)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPU ablation of prefill vs decoding prune")
    p.add_argument("--model", default=os.environ.get("FORKSERVE_MODEL", "/home/dliu/models/Qwen3-4B"))
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--out", default="logs/prune_ablation/shard0.json")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--mild", action="store_true", help="drop at most one hopeless thought per phase")
    p.add_argument(
        "--scale",
        action="store_true",
        help="MATH-500 level 4-5, k=4/8/16, loop-only cuts on Qwen-scale runs",
    )
    p.add_argument(
        "--wide",
        action="store_true",
        help="paper stack at k=4,8,16 (override with --ks)",
    )
    p.add_argument(
        "--ks",
        default="",
        help="comma-separated fan-out, used with --wide or --mild",
    )
    p.add_argument(
        "--admit-mode",
        choices=("score", "winner", "top_m", "alpha"),
        default=os.environ.get("FORKSERVE_ADMIT_MODE", "score"),
        help="APP admission: score bar, winner-only, top-m, or αk",
    )
    p.add_argument("--keep-m", type=int, default=int(os.environ.get("FORKSERVE_KEEP_M", "0") or 0))
    p.add_argument(
        "--admit-alpha",
        type=float,
        default=float(os.environ.get("FORKSERVE_ADMIT_ALPHA", "0.5") or 0.5),
    )
    p.add_argument("--report", default="", help="directory or json to print, no GPU")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-batched-tokens", type=int, default=8192)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--gpu-util", type=float, default=0.90)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--list", action="store_true", help="print the job grid and exit")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        jobs = select_jobs(args)
        print(f"{len(jobs)} jobs")
        for job in jobs:
            print(
                f"{job['tag']:<12} {label_of(job):<16} mix={job['mix']:<8} "
                f"n={job['n']:<3} k={job['k']:<2} D={job['budget']:<4} "
                f"tau={job['tau']:<4} step={job['dpts_step']:<4} "
                f"a={job['alpha']:<4} w={job['esc_window']}"
                + (
                    f" admit={job.get('admit_mode', 'score')}"
                    if job.get("policy") == "app"
                    else ""
                )
            )
        return 0
    if args.report:
        path = Path(args.report)
        files = sorted(path.glob("*.json")) if path.is_dir() else [path]
        rows = load_rows(files)
        print(render_report(rows))
        return 0
    return run_gpu(args)


if __name__ == "__main__":
    raise SystemExit(main())
