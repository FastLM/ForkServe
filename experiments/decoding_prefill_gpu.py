"""GPU check: decoding-time pruning, then the same run with prefill admission.

ESC, Speculative Rejection, and DPTS are the baselines. Each child is a
GSM8K strategy residual on a shared question. The decoder generates a
prefix, scores it by mean token logprob, and only then drops children.
Prefill admission runs first, with the same pruner as the cost-model bench:

* ``base`` — every strategy is prefilled and decoded.
* ``draft`` — text heuristic only. On the published GSM8K strategies this
  keeps all four (the winner is forced, and the others score as ordinary
  prose), so the GPU trace matches ``base``.
* ``app`` — ForkServe+ admission. Non-winners are not submitted, so the
  decoder only sees strategy 0.

Greedy decoding. Survivors continue to the 512-token budget. A child the
prefill stage rejected contributes no prompt and no decode tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from forkserve.bench_tasks import gsm8k_thoughts, gsm8k_trunk, load_gsm8k
from forkserve.config import ForkServeConfig
from forkserve.prefill_prune import PrefillPruner
from forkserve.prune import plus_config
from forkserve.quality import extract_gsm8k_answer, gsm8k_correct

from experiments.prefill_prune_bench import (
    DECODE_BUDGET,
    DPTS_MINI_STEP,
    ESC_WINDOW,
    SR_ALPHA,
    SR_TAU,
)

METHODS = ("esc", "specrej", "dpts")
ADMISSIONS = ("base", "draft", "app")
STEP = {"esc": DECODE_BUDGET, "specrej": SR_TAU, "dpts": DPTS_MINI_STEP}


def admit(thoughts: Sequence[str], policy: str) -> tuple[int, ...]:
    """Branch indices that are allowed to reach the GPU."""
    if policy == "base":
        return tuple(range(len(thoughts)))
    if policy not in ("draft", "app"):
        raise ValueError(policy)
    cfg = plus_config(ForkServeConfig(page_size=16))
    if policy == "draft":
        cfg.skip_known_losers = False
        cfg.gc_admit = False
        cfg.prefill_keep_m = 0
    plan = PrefillPruner(cfg, winner=0).plan(
        list(thoughts),
        token_counts=[32] * len(thoughts),
    )
    kept = tuple(d.index for d in plan.decisions if d.keep)
    return kept or (0,)


def cut_indices(method: str, branch_ids: Sequence[int], scores: dict[int, float]) -> tuple[int, ...]:
    """Children that continue after the decoding-time checkpoint.

    Higher score is better. ESC never fills a window of 5 at this fan-out,
    so it cuts nobody. Speculative Rejection drops the lower half of whoever
    was admitted. DPTS drops the two lowest, and always leaves one.
    """
    ids = list(branch_ids)
    if not ids:
        return ()
    if method == "esc":
        n_drop = 0
    elif method == "specrej":
        n_drop = int(SR_ALPHA * len(ids))
    elif method == "dpts":
        n_drop = min(2, len(ids) - 1)
    else:
        raise ValueError(method)
    n_drop = min(n_drop, len(ids) - 1)
    worst = sorted(ids, key=lambda i: (scores.get(i, 0.0), -i))
    dropped = set(worst[:n_drop])
    return tuple(i for i in ids if i not in dropped)


def vote(texts: Sequence[str], gold: str) -> bool:
    """Majority of extracted answers. Ties break toward the earlier survivor."""
    preds = [extract_gsm8k_answer(t) for t in texts]
    preds = [p for p in preds if p]
    if not preds:
        return False
    counts = Counter(preds)
    best = max(counts.values())
    for pred in preds:
        if counts[pred] == best:
            return gsm8k_correct(f"#### {pred}", gold)
    return False


def _blank() -> dict[str, float]:
    return {
        "n": 0.0,
        "correct": 0.0,
        "prefill_tokens": 0.0,
        "decode_tokens": 0.0,
        "admitted": 0.0,
        "kept": 0.0,
        "wall_ms": 0.0,
        "peak_mib": 0.0,
    }


def _gpu_mib() -> int:
    import subprocess

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return 0
    vals = []
    for line in out.splitlines():
        line = line.strip()
        if line.isdigit():
            vals.append(int(line))
    return max(vals) if vals else 0


def _mean_logprob(out: Any) -> float:
    comp = out.outputs[0]
    ids = comp.token_ids
    lp = getattr(comp, "cumulative_logprob", None)
    if not ids or lp is None:
        return -1.0e9
    return float(lp) / len(ids)


def _generate(llm: Any, SamplingParams: Any, seqs: Sequence[Sequence[int]], n: int, *, scored: bool) -> list[Any]:
    from vllm.inputs import TokensPrompt

    prompts = [TokensPrompt(prompt_token_ids=list(s)) for s in seqs]
    kwargs: dict[str, Any] = {"max_tokens": int(n), "temperature": 0.0}
    if scored:
        kwargs["logprobs"] = 1
    return list(llm.generate(prompts, SamplingParams(**kwargs), use_tqdm=False))


def _run_method(
    llm: Any,
    SamplingParams: Any,
    tok: Any,
    items: Sequence[Any],
    thoughts: Sequence[str],
    thought_ids: Sequence[Sequence[int]],
    method: str,
    admitted: Sequence[int],
    *,
    chunk: int,
    budget: int,
) -> dict[str, float]:
    step = min(STEP[method], budget)
    scored = method != "esc"
    row = _blank()
    row["n"] = float(len(items))
    t0 = time.perf_counter()
    for start in range(0, len(items), chunk):
        batch = items[start : start + chunk]
        trunks = [tok(gsm8k_trunk(item)) for item in batch]
        prompts: list[list[int]] = []
        owners: list[tuple[int, int]] = []
        for i, trunk in enumerate(trunks):
            for b in admitted:
                prompts.append(list(trunk) + list(thought_ids[b]))
                owners.append((i, b))
        row["prefill_tokens"] += float(sum(len(p) for p in prompts))
        row["admitted"] += float(len(batch) * len(admitted))
        first = _generate(llm, SamplingParams, prompts, step, scored=scored)
        partial: dict[tuple[int, int], list[int]] = {}
        scores: dict[int, dict[int, float]] = {i: {} for i in range(len(batch))}
        for (i, b), out in zip(owners, first, strict=True):
            ids = [int(t) for t in out.outputs[0].token_ids]
            partial[(i, b)] = ids
            scores[i][b] = _mean_logprob(out)
            row["decode_tokens"] += float(len(ids))
        cont_prompts: list[list[int]] = []
        cont_owners: list[tuple[int, int]] = []
        kept_text_ids: dict[int, list[list[int]]] = {i: [] for i in range(len(batch))}
        for i in range(len(batch)):
            alive = cut_indices(method, admitted, scores[i])
            row["kept"] += float(len(alive))
            for b in alive:
                got = partial[(i, b)]
                if len(got) >= step and step < budget:
                    cont_prompts.append(prompts[owners.index((i, b))] + got)
                    cont_owners.append((i, b))
                else:
                    kept_text_ids[i].append(got)
        if cont_prompts:
            rest = budget - step
            second = _generate(llm, SamplingParams, cont_prompts, rest, scored=False)
            for (i, b), out in zip(cont_owners, second, strict=True):
                extra = [int(t) for t in out.outputs[0].token_ids]
                row["decode_tokens"] += float(len(extra))
                kept_text_ids[i].append(partial[(i, b)] + extra)
        for i, item in enumerate(batch):
            texts = [tok.decode(ids) for ids in kept_text_ids[i]]
            if vote(texts, item.answer):
                row["correct"] += 1.0
        row["peak_mib"] = float(max(row["peak_mib"], _gpu_mib()))
        done = start + len(batch)
        print(
            f"  {method} {done}/{len(items)} decode={int(row['decode_tokens'])} "
            f"correct={int(row['correct'])}",
            flush=True,
        )
    row["wall_ms"] = (time.perf_counter() - t0) * 1000.0
    row["peak_mib"] = float(max(row["peak_mib"], _gpu_mib()))
    return row


def run_shard(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    items = load_gsm8k(args.limit)
    items = [it for i, it in enumerate(items) if i % args.shards == args.shard]
    thoughts = gsm8k_thoughts(args.branching)
    admitted = {policy: admit(thoughts, policy) for policy in ADMISSIONS}
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_util,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=args.max_batched_tokens,
        max_num_seqs=args.max_seqs,
        enforce_eager=args.enforce_eager,
    )
    tokenizer = llm.get_tokenizer()

    def tok(text: str) -> list[int]:
        return [int(i) for i in tokenizer.encode(text, add_special_tokens=False)]

    def detok(ids: Sequence[int]) -> str:
        return tokenizer.decode(list(ids), skip_special_tokens=True)

    tok.decode = detok  # type: ignore[attr-defined]
    thought_ids = [tok(th) for th in thoughts]
    policies: dict[str, dict[str, float]] = {}
    base_rows: dict[str, dict[str, float]] = {}
    for method in METHODS:
        for policy in ADMISSIONS:
            label = method if policy == "base" else f"{method}+{policy}"
            if policy == "draft" and admitted["draft"] == admitted["base"]:
                copied = dict(base_rows[method])
                copied["aliased"] = 1.0
                policies[label] = copied
                print(f"{label}: draft kept every branch; GPU work matches {method}", flush=True)
                continue
            print(
                f"{label}: admitted={list(admitted[policy])} step={STEP[method]} "
                f"window={ESC_WINDOW} n={len(items)}",
                flush=True,
            )
            row = _run_method(
                llm,
                SamplingParams,
                tok,
                items,
                thoughts,
                thought_ids,
                method,
                admitted[policy],
                chunk=args.chunk,
                budget=args.budget,
            )
            if policy == "app":
                row["aliased"] = 0.0
            policies[label] = row
            if policy == "base":
                base_rows[method] = row
            _dump(args.out, _payload(args, items, thoughts, admitted, policies))
    report = _payload(args, items, thoughts, admitted, policies)
    _dump(args.out, report)
    return report


def _payload(
    args: argparse.Namespace,
    items: Sequence[Any],
    thoughts: Sequence[str],
    admitted: dict[str, tuple[int, ...]],
    policies: dict[str, dict[str, float]],
) -> dict[str, Any]:
    return {
        "model": args.model,
        "shard": args.shard,
        "shards": args.shards,
        "n": len(items),
        "branching": len(thoughts),
        "budget": args.budget,
        "admitted": {k: list(v) for k, v in admitted.items()},
        "thoughts": list(thoughts),
        "policies": policies,
    }


def _dump(path: str, payload: dict[str, Any]) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2))


def merge_shards(directory: Path) -> dict[str, Any]:
    files = sorted(directory.glob("shard*.json"))
    if not files:
        raise FileNotFoundError(directory)
    shards = [json.loads(p.read_text()) for p in files]
    labels = list(shards[0]["policies"])
    merged: dict[str, dict[str, float]] = {}
    for label in labels:
        acc = _blank()
        aliased = 0.0
        for shard in shards:
            row = shard["policies"][label]
            for key in ("n", "correct", "prefill_tokens", "decode_tokens", "admitted", "kept"):
                acc[key] += float(row[key])
            acc["wall_ms"] = max(acc["wall_ms"], float(row["wall_ms"]))
            acc["peak_mib"] = max(acc["peak_mib"], float(row["peak_mib"]))
            aliased = max(aliased, float(row.get("aliased", 0.0)))
        acc["accuracy"] = acc["correct"] / acc["n"] if acc["n"] else 0.0
        acc["aliased"] = aliased
        base_name = label.split("+", 1)[0]
        base = None
        if base_name in merged:
            base = merged[base_name]
        elif base_name == label:
            base = acc
        if base is not None and label != base_name and base["wall_ms"]:
            acc["e2e_cut_vs_base"] = 1.0 - acc["wall_ms"] / base["wall_ms"]
            acc["decode_cut_vs_base"] = (
                1.0 - acc["decode_tokens"] / base["decode_tokens"] if base["decode_tokens"] else 0.0
            )
        else:
            acc["e2e_cut_vs_base"] = 0.0
            acc["decode_cut_vs_base"] = 0.0
        merged[label] = acc
    # Base rows were filled before their +draft/+app siblings, so cuts are set.
    # Recompute cuts now that every base exists even if a shard order differed.
    for label, acc in merged.items():
        base_name = label.split("+", 1)[0]
        if label == base_name:
            continue
        base = merged[base_name]
        acc["e2e_cut_vs_base"] = 1.0 - acc["wall_ms"] / base["wall_ms"] if base["wall_ms"] else 0.0
        acc["decode_cut_vs_base"] = (
            1.0 - acc["decode_tokens"] / base["decode_tokens"] if base["decode_tokens"] else 0.0
        )
    report = {
        "model": shards[0]["model"],
        "n": int(sum(s["n"] for s in shards)),
        "shards": len(shards),
        "admitted": shards[0]["admitted"],
        "budget": shards[0]["budget"],
        "policies": merged,
    }
    dest = directory / "merged.json"
    dest.write_text(json.dumps(report, indent=2))
    report["wrote"] = str(dest)
    return report


def _accuracy(row: dict[str, float]) -> float:
    if "accuracy" in row:
        return float(row["accuracy"])
    return float(row["correct"]) / float(row["n"]) if row.get("n") else 0.0


def _cut(report: dict[str, Any], label: str, row: dict[str, float]) -> float:
    if "e2e_cut_vs_base" in row and "+" in label:
        return float(row["e2e_cut_vs_base"])
    base_name = label.split("+", 1)[0]
    if label == base_name:
        return 0.0
    base = report["policies"].get(base_name)
    if not base or not base.get("wall_ms"):
        return 0.0
    return 1.0 - float(row["wall_ms"]) / float(base["wall_ms"])


def format_report(report: dict[str, Any]) -> str:
    lines = [
        f"model={report['model']} n={report['n']} budget={report['budget']}",
        f"admitted={report['admitted']}",
        f"{'method':<16} {'admit':>8} {'kept':>8} {'prefill':>10} {'decode':>10} "
        f"{'acc':>7} {'wall_s':>8} {'vs_base':>8}",
    ]
    for label, row in report["policies"].items():
        lines.append(
            f"{label:<16} {int(row['admitted']):8d} {int(row['kept']):8d} "
            f"{int(row['prefill_tokens']):10d} {int(row['decode_tokens']):10d} "
            f"{100 * _accuracy(row):6.1f}% {row['wall_ms'] / 1000.0:8.1f} "
            f"{100 * _cut(report, label, row):7.1f}%"
        )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPU decoding prune ± prefill admission")
    p.add_argument("--model", default=os.environ.get("FORKSERVE_MODEL", "/home/dliu/models/Qwen3-8B"))
    p.add_argument("--limit", type=int, default=0, help="0 = full GSM8K test")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--branching", type=int, default=4)
    p.add_argument("--budget", type=int, default=DECODE_BUDGET)
    p.add_argument("--chunk", type=int, default=4)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-batched-tokens", type=int, default=8192)
    p.add_argument("--max-seqs", type=int, default=32)
    p.add_argument("--gpu-util", type=float, default=0.90)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--out", default="logs/decoding_prefill_gpu/shard0.json")
    p.add_argument("--merge", default="", help="directory of shard*.json to combine")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.merge:
        report = merge_shards(Path(args.merge))
        print(format_report(report), flush=True)
        print(f"wrote {report['wrote']}", flush=True)
        return 0
    report = run_shard(args)
    print(format_report(report), flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
