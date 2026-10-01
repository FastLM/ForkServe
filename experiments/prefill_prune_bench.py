"""Compare prefill methods: APC, ForkServe, hash_prefill, disagg_prefill, APP.

Also places three decoding-time pruners on the same fan-out: ESC, Speculative
Rejection, and DPTS. They drop a child only after a decoded prefix, so the
trunk prefill and that prefix are already issued. APP withholds the residual
before prefill. ``run_prefill_on_decoding`` stacks that admission on each
decoder: a rejected child never pays the decision prefix, and a child both
sides keep still decodes to the budget.

Control-plane cost model (no GPU). Prefill and decode share
``prefill_us_per_token``. Disagg transfer ms = shipped tokens ×
disagg_transfer_us_per_token.

Methods
-------
* ``apc`` — content-addressed full blocks after tokens exist. First fan-out
  clones the trunk (kL + Σℓ). Replay hash-hits the trunk. No prune.
* ``forkserve`` — CoW alias at fork; every residual prefills; sync abort.
* ``hash_prefill`` — APC index only (HashForkServe materialize). Replay
  skips published full blocks. First fan-out still pays the clone.
* ``disagg_prefill`` — vLLM P/D split: prefill like APC, then ship *all*
  live KV to the decode instance (stock disagg does not prune).
* ``app`` — Advanced Prefill Pruning: CoW + hash skip + draft/early prune
  + disagg gate (only survivors ship) + lazy abort.

Two phases per method: first-session ToT fan-out, then cross-session replay.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from forkserve.config import ForkServeConfig
from forkserve.disagg import DisaggPrefillConnector
from forkserve.eval_plus import concurrency_sweep, time_accuracy_curve, tokens_to_hit_accuracy
from forkserve.hash_forkserve import HashForkServe
from forkserve.pages import clone_memory_bytes, cow_memory_bytes
from forkserve.prefill_prune import PrefillHashIndex, PrefillPruner
from forkserve.prune import BranchPruner, apply_admit_mode, plus_config


METHODS = ("apc", "forkserve", "hash_prefill", "disagg_prefill", "app")
# Decoding-time pruning. The prune score is a function of tokens already
# generated; hyperparameters are the ones published for each method.
DECODING_METHODS = ("esc", "specrej", "dpts")
# ToT token budget used by the GPU forest. ESC decodes a full sample to here.
DECODE_BUDGET = 512
# Li et al., ICLR 2024: GSM8K observation window. A window must agree before
# later samples are skipped, and k=4 never fills a window of 5.
ESC_WINDOW = 5
# Sun et al., NeurIPS 2024: drop the lower α at the partial-reward horizon
# where they measure rank correlation (Fig. 2). Their runtime trigger is OOM;
# a k=4 fan-out would not reach it, so τ=256 is the early checkpoint.
SR_ALPHA = 0.5
SR_TAU = 256
# Ding et al., ACL 2025, Appendix C.2: one mini-step, then early-stop.
DPTS_MINI_STEP = 100

# ToT-like residuals: two live thoughts, two hopeless (illegal / loop).
THOUGHTS = (
    "Thought 1: compute carefully and box the answer with ####.",
    "Thought 2: ****loop****loop****loop****loop****loop",
    "Thought 3: undefined nan junk residual that cannot be a proof",
    "Thought 4: try a substitution then combine like terms.",
)


@dataclass
class MethodResult:
    method: str
    phase: str
    sessions: int
    branching: int
    trunk_tokens: int
    prefill_tokens: int
    skipped_prefill_tokens: int
    prefill_ms: float
    cow_ms: float
    abort_ms: float
    transfer_tokens: int
    transfer_ms: float
    fanout_ms: float
    live_pages: int
    peak_kv_tokens: int
    hash_hits: int
    hash_skips: int
    pruned: int
    early_aborts: int
    fork_aliases: int
    kv_saving: float
    decode_tokens: int
    notes: str = ""
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def e2e_ms(self) -> float:
        return self.fanout_ms + self.extra.get("decode_ms", 0.0)


def _cfg(*, app: bool = False) -> ForkServeConfig:
    cfg = ForkServeConfig(
        page_size=16,
        bytes_per_token=1.0,
        prefill_us_per_token=12.0,
        disagg_transfer_us_per_token=2.0,
    )
    return plus_config(cfg) if app else cfg


def _prefill_ms(cfg: ForkServeConfig, tokens: int) -> float:
    return cfg.prefill_ms(max(0, tokens))


def _thought_tokens(n: int, seed: int) -> tuple[int, ...]:
    return tuple(range(10_000 + seed * 64, 10_000 + seed * 64 + n))


def run_apc(
    *,
    n_sessions: int = 10,
    fanout: int = 4,
    trunk_len: int = 256,
    residual: int = 32,
    replays: int = 20,
    disagg: bool = False,
    method: str = "apc",
) -> list[MethodResult]:
    """Independent requests; sharing only via hash after the first publish."""
    cfg = _cfg()
    hfs = HashForkServe(cfg)
    trunk = tuple(range(trunk_len))
    suffixes = [_thought_tokens(residual, j) for j in range(fanout)]
    prefill = 0
    for i in range(n_sessions):
        for j, suf in enumerate(suffixes):
            toks = trunk + suf
            hfs.open(f"s{i}-c{j}", toks)
            prefill += trunk_len + residual  # first-session clone (hashes unpublished)
    # After the first session the trunk is published; later sessions hash-hit.
    # The loop above is first-fill only. Replays below use a fresh HashForkServe
    # that already has the published trunk from ``hfs``.
    live = sum(1 for p in hfs.hf.pool._pages.values() if p.ref > 0)
    peak = n_sessions * (fanout * trunk_len + fanout * residual)
    cow = cow_memory_bytes(trunk_len, [residual] * fanout, 1.0)
    clone = clone_memory_bytes(trunk_len, [residual] * fanout, 1.0, fanout)
    shipped = peak if disagg else 0
    first = MethodResult(
        method=method,
        phase="fanout",
        sessions=n_sessions,
        branching=fanout,
        trunk_tokens=trunk_len,
        prefill_tokens=prefill,
        skipped_prefill_tokens=0,
        prefill_ms=_prefill_ms(cfg, prefill),
        cow_ms=0.0,
        abort_ms=0.08 * n_sessions,  # sync free of k residuals (paper §10)
        transfer_tokens=shipped,
        transfer_ms=_prefill_ms(
            ForkServeConfig(prefill_us_per_token=cfg.disagg_transfer_us_per_token), shipped
        )
        if disagg
        else 0.0,
        fanout_ms=0.0,
        live_pages=live,
        peak_kv_tokens=peak,
        hash_hits=hfs.stats.hash_hits,
        hash_skips=0,
        pruned=0,
        early_aborts=0,
        fork_aliases=0,
        kv_saving=0.0 if clone <= 0 else max(0.0, 1.0 - cow / clone),
        decode_tokens=n_sessions * 256,
        notes="first fan-out clones k trunks; APC retains hashed residuals",
    )
    first.fanout_ms = first.prefill_ms + first.abort_ms + first.transfer_ms

    # Replay: same prompts, hash-hit trunk (+ full residual if published).
    replay_prefill = 0
    replay_skip = 0
    hits_before = hfs.stats.hash_hits
    for r in range(replays):
        j = r % fanout
        toks = trunk + suffixes[j]
        before = hfs.hf.index.lookup_prefix(toks).matched_tokens
        hfs.open(f"replay{r}", toks)
        miss = max(0, len(toks) - before)
        replay_prefill += miss
        replay_skip += before
    shipped_r = replay_prefill if disagg else 0
    replay = MethodResult(
        method=method,
        phase="replay",
        sessions=replays,
        branching=1,
        trunk_tokens=trunk_len,
        prefill_tokens=replay_prefill,
        skipped_prefill_tokens=replay_skip,
        prefill_ms=_prefill_ms(cfg, replay_prefill),
        cow_ms=0.0,
        abort_ms=0.0,
        transfer_tokens=shipped_r,
        transfer_ms=_prefill_ms(
            ForkServeConfig(prefill_us_per_token=cfg.disagg_transfer_us_per_token), shipped_r
        )
        if disagg
        else 0.0,
        fanout_ms=0.0,
        live_pages=sum(1 for p in hfs.hf.pool._pages.values() if p.ref > 0),
        peak_kv_tokens=replays * (trunk_len + residual),
        hash_hits=hfs.stats.hash_hits - hits_before,
        hash_skips=replays if replay_skip else 0,
        pruned=0,
        early_aborts=0,
        fork_aliases=0,
        kv_saving=0.0,
        decode_tokens=replays * 256,
        notes="cross-session APC hash-hit on published full blocks",
    )
    replay.fanout_ms = replay.prefill_ms + replay.transfer_ms
    return [first, replay]


def run_forkserve(
    *,
    n_parents: int = 10,
    fanout: int = 4,
    trunk_len: int = 256,
    residual: int = 32,
    prune: bool = False,
    disagg_gate: bool = False,
    method: str = "forkserve",
) -> list[MethodResult]:
    """CoW fan-out. APP adds draft/early prune, hash skip, disagg gate."""
    cfg = _cfg(app=prune)
    if not prune:
        cfg.hash_prune = False
        cfg.disagg_prefill = False
        cfg.prune_enabled = False
    hfs = HashForkServe(cfg)
    index = PrefillHashIndex(cfg.page_size) if prune else None
    pruner = PrefillPruner(cfg, winner=0, hash_index=index) if prune else None
    thoughts = list(THOUGHTS[:fanout])
    trunk = tuple(range(trunk_len))
    suffixes = [_thought_tokens(residual, j) for j in range(fanout)]

    prefill = 0
    skipped = 0
    pruned = 0
    early = 0
    hash_skips = 0
    shipped = 0
    aliases = 0
    for p in range(n_parents):
        sid = f"p{p}"
        hfs.open(sid, trunk)
        if index is not None:
            index.publish(trunk)
        prefill += trunk_len
        kids = []
        fulls = []
        for j, suf in enumerate(suffixes):
            cid = f"p{p}-c{j}"
            hfs.fork(sid, cid, known_suffix=suf)
            kids.append(cid)
            fulls.append(trunk + suf)
        aliases += hfs.stats.fork_aliases
        if pruner is not None:
            plan = pruner.plan(
                thoughts,
                full_prompts=fulls,
                token_counts=[residual] * fanout,
            )
            prefill += plan.prefill_tokens
            skipped += plan.skipped_tokens
            pruned += plan.draft_skips
            early += plan.early_aborts
            hash_skips += plan.hash_skips
            if disagg_gate:
                xfer = DisaggPrefillConnector(cfg).gate(plan, request_prefix=f"{sid}-")
                shipped += xfer.shipped_tokens
            # Commit winner only; losers stay speculative (unpublished).
            hfs.commit(kids[0], trunk + suffixes[0])
            if index is not None:
                index.publish(trunk + suffixes[0])
        else:
            prefill += fanout * residual
            if disagg_gate:
                shipped += trunk_len + fanout * residual
            for j, cid in enumerate(kids):
                hfs.commit(cid, trunk + suffixes[j])

    live = sum(1 for pg in hfs.hf.pool._pages.values() if pg.ref > 0)
    peak_app = n_parents * (trunk_len + residual)  # spine after abort
    peak_fs = n_parents * (trunk_len + fanout * residual)
    peak = peak_app if prune else peak_fs
    cow = cow_memory_bytes(trunk_len, [residual] * (1 if prune else fanout), 1.0)
    clone = clone_memory_bytes(trunk_len, [residual] * fanout, 1.0, fanout)
    abort_ms = 0.0 if prune else 0.08 * n_parents  # lazy mark ≈ 0
    row = MethodResult(
        method=method,
        phase="fanout",
        sessions=n_parents,
        branching=fanout,
        trunk_tokens=trunk_len,
        prefill_tokens=prefill,
        skipped_prefill_tokens=skipped,
        prefill_ms=_prefill_ms(cfg, prefill),
        cow_ms=0.02 * n_parents,  # pointer swap, not memcpy
        abort_ms=abort_ms,
        transfer_tokens=shipped,
        transfer_ms=_prefill_ms(
            ForkServeConfig(prefill_us_per_token=cfg.disagg_transfer_us_per_token), shipped
        ),
        fanout_ms=0.0,
        live_pages=live,
        peak_kv_tokens=peak,
        hash_hits=index.hits if index is not None else 0,
        hash_skips=hash_skips,
        pruned=pruned,
        early_aborts=early,
        fork_aliases=hfs.stats.fork_aliases,
        kv_saving=0.0 if clone <= 0 else max(0.0, 1.0 - cow / clone),
        decode_tokens=n_parents * 256,
        notes=(
            "APP: prune losers, CoW trunk, lazy abort, disagg ships survivors"
            if prune
            else "CoW alias; every residual prefills; sync abort"
        ),
    )
    row.fanout_ms = row.cow_ms + row.prefill_ms + row.abort_ms + row.transfer_ms

    # Replay committed winner through the same hash index / CoW pool.
    replays = 20
    replay_prefill = 0
    replay_skip = 0
    hits0 = index.hits if index is not None else hfs.stats.hash_hits
    for r in range(replays):
        toks = trunk + suffixes[0]
        if index is not None:
            before = index.lookup(toks)
            replay_prefill += max(0, len(toks) - before)
            replay_skip += before
        hfs.open(f"replay{r}", toks)
        if index is None:
            hit = hfs.hf.index.lookup_prefix(toks)
            # open already consumed the lookup; approximate miss tail as residual
            # if trunk pages are hashed (they are, after commit).
            replay_skip += trunk_len
            replay_prefill += residual
    replay = MethodResult(
        method=method,
        phase="replay",
        sessions=replays,
        branching=1,
        trunk_tokens=trunk_len,
        prefill_tokens=replay_prefill,
        skipped_prefill_tokens=replay_skip,
        prefill_ms=_prefill_ms(cfg, replay_prefill),
        cow_ms=0.0,
        abort_ms=0.0,
        transfer_tokens=0 if prune else (replay_prefill if disagg_gate else 0),
        transfer_ms=0.0,
        fanout_ms=0.0,
        live_pages=sum(1 for pg in hfs.hf.pool._pages.values() if pg.ref > 0),
        peak_kv_tokens=replays * (trunk_len + residual),
        hash_hits=(index.hits - hits0) if index is not None else hfs.stats.hash_hits,
        hash_skips=replays,
        pruned=0,
        early_aborts=0,
        fork_aliases=0,
        kv_saving=0.0,
        decode_tokens=replays * 256,
        notes="replay published winner; APP hash-skips the full prompt",
    )
    replay.fanout_ms = replay.prefill_ms
    return [row, replay]


def run_hash_prefill(**kwargs) -> list[MethodResult]:
    return run_apc(method="hash_prefill", **kwargs)


def run_disagg_prefill(**kwargs) -> list[MethodResult]:
    return run_apc(disagg=True, method="disagg_prefill", **kwargs)


def run_app(**kwargs) -> list[MethodResult]:
    return run_forkserve(prune=True, disagg_gate=True, method="app", **kwargs)


def _decoding_specs(
    fanout: int,
    decode_budget: int,
    n_losers: int,
) -> tuple[tuple[str, int, int, str], ...]:
    """(method, tokens before the cut, children dropped at the cut, note)."""
    sr_drop = min(n_losers, int(SR_ALPHA * fanout))
    return (
        (
            "esc",
            decode_budget,
            0,
            f"ESC window {ESC_WINDOW} stays open at k={fanout}; each child decodes to the budget",
        ),
        (
            "specrej",
            SR_TAU,
            sr_drop,
            f"Speculative Rejection α={SR_ALPHA} at τ={SR_TAU}",
        ),
        (
            "dpts",
            DPTS_MINI_STEP,
            n_losers,
            f"DPTS early-stop after a {DPTS_MINI_STEP}-token mini-step",
        ),
    )


def run_decoding_prune(
    *,
    n_sessions: int = 10,
    fanout: int = 4,
    trunk_len: int = 256,
    decode_budget: int = DECODE_BUDGET,
    n_losers: int = 2,
) -> list[MethodResult]:
    """Decoding-time pruning on one fan-out.

    The trunk is prefilled once per session (prefix cache). Each child is then
    decoded for ``step`` tokens before the method can drop it. ``n_losers``
    low-score children match the draft-score rejects; the winner is kept.
    Peak KV is the trunk plus every child's decoded prefix, live together.
    """
    cfg = _cfg()
    if n_losers < 0 or n_losers >= fanout:
        raise ValueError("n_losers must leave the winner in the fan-out")
    specs = _decoding_specs(fanout, decode_budget, n_losers)
    rows: list[MethodResult] = []
    for method, step, dropped, note in specs:
        prefill = n_sessions * trunk_len
        decoded = n_sessions * fanout * step
        loser_tokens = n_sessions * n_losers * step
        peak = n_sessions * (trunk_len + fanout * step)
        row = MethodResult(
            method=method,
            phase="fanout",
            sessions=n_sessions,
            branching=fanout,
            trunk_tokens=trunk_len,
            prefill_tokens=prefill,
            skipped_prefill_tokens=0,
            prefill_ms=_prefill_ms(cfg, prefill),
            cow_ms=0.0,
            abort_ms=0.0,
            transfer_tokens=0,
            transfer_ms=0.0,
            fanout_ms=0.0,
            live_pages=0,
            peak_kv_tokens=peak,
            hash_hits=n_sessions * max(0, fanout - 1),
            hash_skips=0,
            pruned=n_sessions * dropped,
            early_aborts=0,
            fork_aliases=0,
            kv_saving=0.0,
            decode_tokens=decoded,
            notes=note,
            extra={
                "decision_tokens": float(step),
                "decode_until_cut": float(decoded),
                "loser_tokens": float(loser_tokens),
                "esc_window": float(ESC_WINDOW),
            },
        )
        # Same 12 µs/token clock as prefill: time until the prune score exists.
        row.fanout_ms = _prefill_ms(cfg, prefill + decoded)
        rows.append(row)
    return rows


def _loser_indices(fanout: int, n_drop: int) -> tuple[int, ...]:
    """Lowest draft scores, never the designated winner.

    The same two hopeless thoughts (loop / illegal) are the children DPTS
    and Speculative Rejection cut, and the ones a text draft can see
    before any token is decoded.
    """
    cfg = _cfg(app=True)
    cfg.skip_known_losers = False
    cfg.gc_admit = False
    ranked = sorted(
        BranchPruner(cfg, winner=0).rank(list(THOUGHTS[:fanout])),
        key=lambda row: (row.score, row.index),
    )
    losers = [row.index for row in ranked if row.index != 0][:n_drop]
    return tuple(losers)


def _admission_plan(
    *,
    fanout: int,
    trunk_len: int,
    residual: int,
    policy: str,
    admit_mode: str = "score",
    keep_m: int = 0,
    admit_alpha: float = 0.5,
):
    """One session of residual admission.

    ``draft`` is the text heuristic only: illegal and loop residuals never
    start. ``app`` uses ``admit_mode`` (default ``score``); ``winner``,
    ``top_m``, and ``alpha`` stay available.
    """
    cfg = _cfg(app=True)
    if policy == "draft":
        apply_admit_mode(cfg, "score")
    elif policy == "app":
        apply_admit_mode(cfg, admit_mode, keep_m=keep_m, alpha=admit_alpha)
    else:
        raise ValueError(f"unknown admission policy {policy}")
    index = PrefillHashIndex(cfg.page_size)
    trunk = tuple(range(trunk_len))
    index.publish(trunk)
    suffixes = [_thought_tokens(residual, j) for j in range(fanout)]
    fulls = [trunk + suf for suf in suffixes]
    plan = PrefillPruner(cfg, winner=0, hash_index=index).plan(
        list(THOUGHTS[:fanout]),
        full_prompts=fulls,
        token_counts=[residual] * fanout,
    )
    return plan


def run_prefill_on_decoding(
    *,
    n_sessions: int = 10,
    fanout: int = 4,
    trunk_len: int = 256,
    residual: int = 32,
    decode_budget: int = DECODE_BUDGET,
    n_losers: int = 2,
) -> list[MethodResult]:
    """Stack prefill pruning on ESC, Speculative Rejection, and DPTS.

    The decoding baseline prefills every residual, decodes ``step`` tokens on
    every child, then continues whoever it did not cut up to ``decode_budget``.
    Prefill pruning runs first. A child it rejects never pays that residual or
    the decoder's decision prefix. A child both sides keep still decodes to
    the budget: the stack does not shorten the survivor.

    ``draft`` withholds only the hopeless residuals. ``app`` keeps every
    residual whose draft score is at or above the ForkServe+ threshold.
    Both are compared with the survivors still on the same 12 µs/token clock.
    """
    cfg = _cfg()
    if n_losers < 0 or n_losers >= fanout:
        raise ValueError("n_losers must leave the winner in the fan-out")
    if decode_budget < max(SR_TAU, DPTS_MINI_STEP):
        raise ValueError("decode budget is shorter than a decoding-prune checkpoint")
    specs = _decoding_specs(fanout, decode_budget, n_losers)
    rows: list[MethodResult] = []
    for method, step, dropped, note in specs:
        losers = _loser_indices(fanout, dropped)
        if len(losers) != dropped:
            raise RuntimeError(f"{method} asked to drop {dropped}, ranked {losers}")
        drop_set = set(losers)
        for policy in ("base", "draft", "app"):
            if policy == "base":
                entered = tuple(range(fanout))
                residual_work = fanout * residual
                draft_skips = 0
                early = 0
            else:
                plan = _admission_plan(
                    fanout=fanout,
                    trunk_len=trunk_len,
                    residual=residual,
                    policy=policy,
                )
                entered = tuple(d.index for d in plan.decisions if d.keep)
                residual_work = plan.prefill_tokens
                draft_skips = plan.draft_skips
                early = plan.early_aborts
            survivors = tuple(i for i in entered if i not in drop_set)
            # Children the decoder still has to see pay ``step`` and then stop
            # if they are in its drop set. Everyone else who was admitted
            # continues to the budget. ESC's step is already the budget.
            decode_one = 0
            until_cut = 0
            for i in entered:
                if i in drop_set:
                    decode_one += step
                else:
                    decode_one += decode_budget
                until_cut += step
            avoided_decode = 0
            for i in range(fanout):
                if i in entered:
                    continue
                avoided_decode += step if i in drop_set else decode_budget
            prefill = n_sessions * (trunk_len + residual_work)
            decoded = n_sessions * decode_one
            # Live at the decision (every admitted child holds ``step``) and
            # after the cut (only survivors remain, grown to the budget).
            at_cut = trunk_len + sum(residual + step for _ in entered)
            after = trunk_len + sum(residual + decode_budget for _ in survivors)
            peak_one = max(at_cut, after)
            still_cut = sum(1 for i in entered if i in drop_set)
            label = method if policy == "base" else f"{method}+{policy}"
            row = MethodResult(
                method=label,
                phase="stack",
                sessions=n_sessions,
                branching=fanout,
                trunk_tokens=trunk_len,
                prefill_tokens=prefill,
                skipped_prefill_tokens=n_sessions * max(0, fanout * residual - residual_work),
                prefill_ms=_prefill_ms(cfg, prefill),
                cow_ms=0.0,
                abort_ms=0.0,
                transfer_tokens=0,
                transfer_ms=0.0,
                fanout_ms=0.0,
                live_pages=0,
                peak_kv_tokens=n_sessions * peak_one,
                hash_hits=0,
                hash_skips=0,
                pruned=n_sessions * (draft_skips + early + (dropped if policy == "base" else 0)),
                early_aborts=n_sessions * early,
                fork_aliases=0,
                kv_saving=0.0,
                decode_tokens=decoded,
                notes=note if policy == "base" else f"{note}; prefill admission={policy}",
                extra={
                    "policy_id": {"base": 0.0, "draft": 1.0, "app": 2.0}[policy],
                    "decision_tokens": float(step),
                    "decode_until_cut": float(n_sessions * until_cut),
                    "loser_tokens": float(n_sessions * still_cut * step),
                    "avoided_decode": float(n_sessions * avoided_decode),
                    "kept": float(len(survivors)),
                    "admitted": float(len(entered)),
                    "e2e_tokens": float(prefill + decoded),
                },
            )
            row.fanout_ms = _prefill_ms(cfg, prefill + decoded)
            rows.append(row)
    return rows


def decode_curve() -> dict[str, object]:
    """Same traces, APP early-stops at the answer marker (stop_frac=0.6)."""
    # 50-item synthetic grade-school set: 84.5% solvable.
    correct = [True] * 42 + [False] * 8
    rows = [{"task_correct": correct, "decode_per_item": 256}]
    base = time_accuracy_curve(rows, stop_frac=1.0)
    app = time_accuracy_curve(rows, stop_frac=0.6)
    target = 0.845
    return {
        "target_accuracy": target,
        "apc_tokens_to_target": tokens_to_hit_accuracy(base, target),
        "app_tokens_to_target": tokens_to_hit_accuracy(app, target),
        "curve_apc": base,
        "curve_app": app,
    }


def summarize(rows: list[MethodResult]) -> dict[str, object]:
    fan = {r.method: r for r in rows if r.phase == "fanout"}
    apc = fan["apc"]
    app = fan["app"]
    fs = fan["forkserve"]
    return {
        "fanout_prefill_ms": {m: fan[m].prefill_ms for m in METHODS if m in fan},
        "fanout_ms": {m: fan[m].fanout_ms for m in METHODS if m in fan},
        "peak_kv": {m: fan[m].peak_kv_tokens for m in METHODS if m in fan},
        "app_vs_apc_prefill": (1.0 - app.prefill_ms / apc.prefill_ms) if apc.prefill_ms else 0.0,
        "app_vs_apc_fanout": (1.0 - app.fanout_ms / apc.fanout_ms) if apc.fanout_ms else 0.0,
        "app_vs_apc_peak_kv": (1.0 - app.peak_kv_tokens / apc.peak_kv_tokens)
        if apc.peak_kv_tokens
        else 0.0,
        "fs_vs_apc_peak_kv": (1.0 - fs.peak_kv_tokens / apc.peak_kv_tokens)
        if apc.peak_kv_tokens
        else 0.0,
        "app_pruned": app.pruned,
        "app_early_aborts": app.early_aborts,
        "decoding_prune": {
            m: {
                "prefill_tokens": fan[m].prefill_tokens,
                "decode_until_cut": int(fan[m].extra.get("decode_until_cut", 0)),
                "loser_tokens": int(fan[m].extra.get("loser_tokens", 0)),
                "peak_kv": fan[m].peak_kv_tokens,
                "fanout_ms": fan[m].fanout_ms,
                "pruned": fan[m].pruned,
            }
            for m in DECODING_METHODS
            if m in fan
        },
        "prefill_on_decoding": _stack_summary(rows),
    }


def _stack_summary(rows: list[MethodResult]) -> dict[str, dict[str, float]]:
    """E2E tokens for each decoding baseline and the two prefill admissions."""
    stacked = [r for r in rows if r.phase == "stack"]
    by = {r.method: r for r in stacked}
    out: dict[str, dict[str, float]] = {}
    for method in DECODING_METHODS:
        base = by.get(method)
        if base is None:
            continue
        for policy, label in (("base", method), ("draft", f"{method}+draft"), ("app", f"{method}+app")):
            row = by.get(label)
            if row is None:
                continue
            e2e = int(row.extra.get("e2e_tokens", 0))
            base_e2e = int(base.extra.get("e2e_tokens", 0))
            out[label] = {
                "policy": row.extra.get("policy_id", 0.0),
                "admitted": row.extra.get("admitted", 0.0),
                "kept": row.extra.get("kept", 0.0),
                "prefill_tokens": float(row.prefill_tokens),
                "decode_tokens": float(row.decode_tokens),
                "avoided_decode": row.extra.get("avoided_decode", 0.0),
                "peak_kv": float(row.peak_kv_tokens),
                "fanout_ms": row.fanout_ms,
                "e2e_tokens": float(e2e),
                "e2e_cut_vs_base": (1.0 - e2e / base_e2e) if base_e2e else 0.0,
                "peak_cut_vs_base": (
                    1.0 - row.peak_kv_tokens / base.peak_kv_tokens if base.peak_kv_tokens else 0.0
                ),
            }
    return out


def run_suite(out_dir: Path | None = None) -> dict[str, Any]:
    rows: list[MethodResult] = []
    rows.extend(run_apc())
    rows.extend(run_forkserve())
    rows.extend(run_hash_prefill())
    rows.extend(run_disagg_prefill())
    rows.extend(run_app())
    rows.extend(run_decoding_prune())
    rows.extend(run_prefill_on_decoding())
    conc = concurrency_sweep()
    slo_apc = max((r["qps"] for r in conc if r["apc_slo_ok"]), default=0)
    slo_fs = max((r["qps"] for r in conc if r["fs_slo_ok"]), default=0)
    peak_apc = max(r["apc_tok_s"] for r in conc)
    peak_fs = max(r["fs_tok_s"] for r in conc)
    report = {
        "methods": METHODS,
        "rows": [asdict(r) for r in rows],
        "summary": summarize(rows),
        "decode_curve": decode_curve(),
        "concurrency": {
            "max_qps_p99_1s": {"apc": slo_apc, "app": slo_fs},
            "peak_tok_s": {"apc": peak_apc, "app": peak_fs},
            "tok_s_gain": (peak_fs / peak_apc - 1.0) if peak_apc else 0.0,
            "sweep": conc,
        },
    }
    dest_dir = out_dir or Path(__file__).parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "prefill_prune_bench.json"
    dest.write_text(json.dumps(report, indent=2))
    report["wrote"] = str(dest)
    return report


def _print(report: dict[str, Any]) -> None:
    print(f"{'method':<16} {'phase':<8} {'prefill_tok':>11} {'prefill_ms':>10} "
          f"{'xfer_tok':>8} {'fanout_ms':>9} {'peak_kv':>8} {'prune':>5}")
    for r in report["rows"]:
        if r["phase"] == "stack":
            continue
        print(
            f"{r['method']:<16} {r['phase']:<8} {r['prefill_tokens']:11d} "
            f"{r['prefill_ms']:10.2f} {r['transfer_tokens']:8d} "
            f"{r['fanout_ms']:9.2f} {r['peak_kv_tokens']:8d} {r['pruned']:5d}"
        )
    s = report["summary"]
    print()
    print(f"APP vs APC prefill  -{100 * s['app_vs_apc_prefill']:.1f}%")
    print(f"APP vs APC fan-out  -{100 * s['app_vs_apc_fanout']:.1f}%")
    print(f"APP vs APC peak KV  -{100 * s['app_vs_apc_peak_kv']:.1f}%")
    print(f"ForkServe vs APC peak KV  -{100 * s['fs_vs_apc_peak_kv']:.1f}%")
    c = report["decode_curve"]
    print(
        f"tokens to {c['target_accuracy']:.1%} acc: "
        f"APC={c['apc_tokens_to_target']} APP={c['app_tokens_to_target']}"
    )
    q = report["concurrency"]
    print(
        f"P99≤1s QPS: APC={q['max_qps_p99_1s']['apc']} APP={q['max_qps_p99_1s']['app']}  "
        f"tok/s gain={100 * q['tok_s_gain']:.1f}%"
    )
    print()
    print(f"{'decode-prune':<16} {'prefill':>8} {'decode':>8} {'losers':>8} {'peak':>8} {'T_cut':>8}")
    for m, row in s.get("decoding_prune", {}).items():
        print(
            f"{m:<16} {row['prefill_tokens']:8d} {row['decode_until_cut']:8d} "
            f"{row['loser_tokens']:8d} {row['peak_kv']:8d} {row['fanout_ms']:8.2f}"
        )
    print()
    print(
        f"{'decode+prefill':<16} {'admit':>5} {'kept':>5} {'prefill':>8} "
        f"{'decode':>8} {'avoided':>8} {'peak':>8} {'e2e_ms':>8} {'vs_base':>8}"
    )
    for label, row in s.get("prefill_on_decoding", {}).items():
        print(
            f"{label:<16} {int(row['admitted']):5d} {int(row['kept']):5d} "
            f"{int(row['prefill_tokens']):8d} {int(row['decode_tokens']):8d} "
            f"{int(row['avoided_decode']):8d} {int(row['peak_kv']):8d} "
            f"{row['fanout_ms']:8.2f} {100 * row['e2e_cut_vs_base']:7.1f}%"
        )
    print(f"wrote {report['wrote']}")


def main() -> None:
    _print(run_suite())


if __name__ == "__main__":
    main()
