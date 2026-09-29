"""Control-plane multiturn ToT: 4 agents × D turns on one problem.

No GPU. Prefill ms = tokens × prefill_us_per_token. Each session is one
problem. Every turn fans out k=4 residuals on a growing winner spine:

* ``apc`` — hash after tokens exist. Turn t pays k × spine unless the
  spine was published last turn; residuals always miss.
* ``forkserve`` — CoW alias; every residual prefills; losers abort.
* ``app`` — CoW + draft prune (keep winner + 1) + slack-fill the next
  winner thought into the budget the losers just freed.

The gap widens with depth: APC's live KV is k × spine, ForkServe's is
spine + residuals, APP skips the hopeless siblings and hits the next
turn's known suffix.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from forkserve.config import ForkServeConfig
from forkserve.prefill_prune import PrefillHashIndex, PrefillPruner
from forkserve.prune import plus_config
from forkserve.slack_fill import freed_tokens, miss_tokens, refill

OUT = Path(__file__).with_suffix(".json")
THOUGHTS = (
    (
        "Thought 1: translate the story into equations.",
        "Thought 2: ****loop****loop****loop****loop****loop",
        "Thought 3: undefined nan junk residual that cannot be a proof",
        "Thought 4: name each intermediate quantity.",
    ),
    (
        "Thought 1: check the last step and continue.",
        "Thought 2: ****loop****loop****loop****loop****loop",
        "Thought 3: undefined nan junk residual that cannot be a proof",
        "Thought 4: try an alternative identity.",
    ),
    (
        "Thought 1: write the remaining algebra and box the answer.",
        "Thought 2: ****loop****loop****loop****loop****loop",
        "Thought 3: undefined nan junk residual that cannot be a proof",
        "Thought 4: audit units and give the number.",
    ),
)


@dataclass
class TurnRow:
    method: str
    turn: int
    sessions: int
    branching: int
    spine_tokens: int
    prefill_tokens: int
    skipped_tokens: int
    peak_kv_tokens: int
    clone_kv_tokens: int
    kv_saving: float
    pinned_tokens: int
    next_miss_tokens: int
    prefill_ms: float
    notes: str = ""


def _cfg(*, app: bool) -> ForkServeConfig:
    cfg = ForkServeConfig(
        page_size=16,
        bytes_per_token=1.0,
        prefill_us_per_token=12.0,
    )
    return plus_config(cfg) if app else cfg


def _residual(n: int, seed: int) -> tuple[int, ...]:
    return tuple(range(20_000 + seed * 80, 20_000 + seed * 80 + n))


def run_method(
    method: str,
    *,
    sessions: int = 8,
    branching: int = 4,
    turns: int = 3,
    trunk_len: int = 256,
    residual: int = 32,
    step: int = 16,
) -> list[TurnRow]:
    app = method == "app"
    cfg = _cfg(app=app)
    us = cfg.prefill_us_per_token
    rows: list[TurnRow] = []
    for s in range(sessions):
        spine = tuple(range(s * 100_000, s * 100_000 + trunk_len))
        index = PrefillHashIndex(cfg.page_size)
        if method == "apc":
            index.publish(spine)
        for turn in range(turns):
            texts = THOUGHTS[turn % len(THOUGHTS)][:branching]
            res = [_residual(residual, 100 * turn + j) for j in range(branching)]
            fulls = [spine + r for r in res]
            clone = branching * len(spine) + branching * residual
            nxt = _residual(residual, 100 * (turn + 1))
            grown = spine + res[0] + tuple(range(1, step + 1))
            next_prompt = grown + nxt
            if method == "apc":
                # Hash hits the published spine; k residuals still miss.
                prefill = sum(miss_tokens(index, p) for p in fulls)
                skipped = branching * len(spine) + branching * residual - prefill
                peak = len(spine) + branching * residual
                pinned = 0
                next_miss = 0
                index.publish(spine + res[0])
            elif method == "forkserve":
                prefill = branching * residual
                skipped = 0
                peak = len(spine) + residual
                pinned = 0
                next_miss = residual
                index.publish(spine + res[0])
            else:
                plan = PrefillPruner(
                    cfg, winner=0, hash_index=index
                ).plan(list(texts), full_prompts=fulls, token_counts=[residual] * branching)
                prefill = plan.prefill_tokens
                skipped = plan.skipped_tokens
                peak = len(spine) + residual
                pinned = 0
                next_miss = residual
                if turn + 1 < turns:
                    filled = refill(
                        index,
                        grown,
                        [nxt],
                        freed=freed_tokens(plan.decisions),
                    )
                    pinned = filled.pinned_tokens
                    next_miss = miss_tokens(index, next_prompt)
            saving = 0.0 if clone <= 0 else 1.0 - (peak / clone)
            rows.append(
                TurnRow(
                    method=method,
                    turn=turn,
                    sessions=1,
                    branching=branching,
                    spine_tokens=len(spine),
                    prefill_tokens=prefill,
                    skipped_tokens=skipped,
                    peak_kv_tokens=peak,
                    clone_kv_tokens=clone,
                    kv_saving=saving,
                    pinned_tokens=pinned,
                    next_miss_tokens=next_miss,
                    prefill_ms=prefill * us / 1000.0,
                    notes=f"session {s} turn {turn}",
                )
            )
            spine = grown
    return rows


def summarize(rows: list[TurnRow]) -> dict:
    by: dict[str, list[TurnRow]] = {}
    for r in rows:
        by.setdefault(r.method, []).append(r)
    out: dict[str, dict] = {}
    for method, xs in by.items():
        n = max(len(xs), 1)
        last_turn = max(r.turn for r in xs)
        deep = [r for r in xs if r.turn == last_turn]
        out[method] = {
            "prefill_tokens": sum(r.prefill_tokens for r in xs),
            "prefill_ms": sum(r.prefill_ms for r in xs),
            "peak_kv_last": max((r.peak_kv_tokens for r in deep), default=0),
            "clone_kv_last": max((r.clone_kv_tokens for r in deep), default=0),
            "kv_saving_last": sum(r.kv_saving for r in deep) / max(len(deep), 1),
            "pinned_tokens": sum(r.pinned_tokens for r in xs),
            "next_miss_tokens": sum(r.next_miss_tokens for r in xs),
            "turns": last_turn + 1,
            "rows": n,
        }
    apc = out.get("apc") or {}
    for other in ("forkserve", "app"):
        if other in out and apc.get("prefill_tokens"):
            out[other]["prefill_cut_vs_apc"] = 1.0 - (
                out[other]["prefill_tokens"] / apc["prefill_tokens"]
            )
            if apc.get("peak_kv_last"):
                out[other]["kv_cut_vs_apc"] = 1.0 - (
                    out[other]["peak_kv_last"] / apc["peak_kv_last"]
                )
    return out


def main() -> None:
    sessions = 8
    branching = 4
    turns = 3
    rows: list[TurnRow] = []
    for method in ("apc", "forkserve", "app"):
        rows.extend(
            run_method(
                method,
                sessions=sessions,
                branching=branching,
                turns=turns,
            )
        )
    summary = summarize(rows)
    payload = {
        "sessions": sessions,
        "branching": branching,
        "turns": turns,
        "agents_per_problem": branching,
        "summary": summary,
        "rows": [asdict(r) for r in rows],
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(
        f"sessions={sessions} k={branching} turns={turns} "
        f"(one problem, {branching} agents / batch)\n"
    )
    print(f"{'method':<12} prefill_tok  prefill_ms  peak_kv  kv_save  pinned  next_miss  vs_apc")
    print("-" * 88)
    for method in ("apc", "forkserve", "app"):
        s = summary[method]
        cut = s.get("prefill_cut_vs_apc")
        cut_s = f"{cut:.1%}" if cut is not None else "  —"
        print(
            f"{method:<12} {s['prefill_tokens']:10d}  {s['prefill_ms']:9.2f}  "
            f"{s['peak_kv_last']:7d}  {s['kv_saving_last']:6.2f}  "
            f"{s['pinned_tokens']:6d}  {s['next_miss_tokens']:8d}  {cut_s}"
        )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
