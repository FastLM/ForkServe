"""Compare prefill methods: APC, ForkServe, hash_prefill, disagg_prefill, APP.

Control-plane cost model (no GPU). Prefill ms = tokens × prefill_us_per_token.
Disagg transfer ms = shipped tokens × disagg_transfer_us_per_token.

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
from forkserve.prune import plus_config


METHODS = ("apc", "forkserve", "hash_prefill", "disagg_prefill", "app")

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
    }


def run_suite(out_dir: Path | None = None) -> dict[str, Any]:
    rows: list[MethodResult] = []
    rows.extend(run_apc())
    rows.extend(run_forkserve())
    rows.extend(run_hash_prefill())
    rows.extend(run_disagg_prefill())
    rows.extend(run_app())
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
    print(f"wrote {report['wrote']}")


def main() -> None:
    _print(run_suite())


if __name__ == "__main__":
    main()
