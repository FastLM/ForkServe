"""System comparison: vLLM recompute vs APC vs ForkServe on 2/4 GPUs.

Workloads

* ``gsm8k`` / ``svamp`` / ``gsmhard`` — grade-school math, ToT fan-out.
* ``math500`` / ``aime`` / ``amc23`` — contest math, ToT fan-out, boxed gold.
* ``game24`` — ToT paper 24-game; high branching, tiny trunk.
* ``humaneval`` — function completion + pytest tool-idle (ReAct wrappers).
* ``tot`` / ``react`` / ``multi`` — synthetic microbenchmarks of the same verbs.

Baselines (offline ``vllm.LLM.generate``):

* ``vllm_recompute`` — prefix cache off; each branch prefills the full trunk.
* ``vllm_apc`` — automatic prefix cache; trunk is hashed and reused.
* ``forkserve`` — CoW alias + two-class engine loop + speculative prefill.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence


SYSTEMS = ("vllm_recompute", "vllm_apc", "forkserve", "forkserve_plus")


def _now_stamp() -> str:
    return time.strftime("%F %T")


def progress(args: argparse.Namespace | None, msg: str) -> None:
    """Timestamped line on stdout and optional ``FORKSERVE_PROGRESS_LOG`` file."""
    line = f"[{_now_stamp()}] {msg}"
    print(line, flush=True)
    path = ""
    if args is not None:
        path = str(getattr(args, "progress_log", "") or "")
    path = path or os.environ.get("FORKSERVE_PROGRESS_LOG", "")
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def _sys_tp(args: argparse.Namespace, default_system: str = "") -> str:
    system = str(getattr(args, "system", None) or default_system or "forkserve")
    tp = getattr(args, "tp", None)
    if isinstance(tp, (list, tuple)) and tp:
        tp_s = str(tp[0])
    else:
        tp_s = str(tp or "?")
    return f"{system} tp={tp_s}"


def _score_so_far(
    workload: str,
    texts: Sequence[str],
    golds: Sequence[str],
    prompts: Sequence[str] | None = None,
) -> str:
    from forkserve.quality import score_task

    n = min(len(texts), len(golds))
    if n <= 0:
        return "score=n/a"
    scored = score_task(workload, texts[:n], golds[:n], prompts[:n] if prompts else None)
    ok = int(sum(scored.correct))
    return f"{scored.metric}={scored.score:.3f} ({ok}/{scored.n})"


def _eta_s(done: int, total: int, elapsed_s: float) -> str:
    if done <= 0 or elapsed_s <= 0 or total <= done:
        return "eta=?"
    remain = elapsed_s * (total - done) / done
    if remain >= 3600:
        return f"eta={remain / 3600:.1f}h"
    if remain >= 90:
        return f"eta={remain / 60:.1f}m"
    return f"eta={remain:.0f}s"
WORKLOADS = (
    "gsm8k",
    "svamp",
    "gsmhard",
    "math500",
    "aime",
    "amc23",
    "game24",
    "humaneval",
    "tot",
    "react",
    "multi",
)
GRADE_MATH = ("gsm8k", "svamp", "gsmhard")
CONTEST_MATH = ("math500", "aime", "amc23")
# GSM8K needs a finished #### line; 256 tokens still truncates 8B/14B.
GSM8K_DECODE_DEFAULT = 512
CONTEST_DECODE_DEFAULT = 768


def workload_decode(args: argparse.Namespace, workload: str) -> int:
    """Per-workload decode. Tests that pass a tiny ``--decode`` keep that value."""
    n = int(getattr(args, "decode", 256) or 256)
    if n < 64:
        return n
    if workload in CONTEST_MATH:
        return max(n, CONTEST_DECODE_DEFAULT)
    if workload not in GRADE_MATH:
        return n
    gs = getattr(args, "gsm8k_decode", None)
    if gs is not None:
        return int(gs)
    return max(n, GSM8K_DECODE_DEFAULT)


@dataclass
class RunMetrics:
    system: str
    workload: str
    tp: int
    e2e_ms: float
    fanout_ms: float = 0.0
    decode_ms: float = 0.0
    ttft_from_obs_ms: float = 0.0
    spec_ms: float = 0.0
    idle_ms: float = 0.0
    trunk_tokens: int = 0
    residual_tokens: list[int] = field(default_factory=list)
    peak_kv_tokens: int = 0
    m_cow_mib: float = 0.0
    m_clone_mib: float = 0.0
    kv_saving: float = 0.0
    gpu_mem_mib: list[int] = field(default_factory=list)
    known_suffix_hit_rate: float = 0.0
    decode_tokens: int = 0
    decode_per_item: int = 0
    sessions: int = 1
    branching: int = 1
    notes: str = ""
    decode_ids: list[list[int]] = field(default_factory=list)
    decode_texts: list[str] = field(default_factory=list)
    golds: list[str] = field(default_factory=list)
    gold_prompts: list[str] = field(default_factory=list)
    item_ids: list[str] = field(default_factory=list)
    item_preds: list[str] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    task_metric: str = ""
    task_n: int = 0
    task_score: float = -1.0
    task_correct: list[bool] = field(default_factory=list)
    quality_vs: str = ""
    quality_ref_score: float = -1.0
    quality_delta: float = 0.0
    quality_collapsed: bool = False
    cow_ms: float = 0.0
    prefill_ms: float = 0.0
    abort_mark_ms: float = 0.0
    abort_reclaim_ms: float = 0.0
    pointer_swaps: int = 0
    pruned_branches: int = 0
    prefilled_branches: int = 0
    hash_skips: int = 0
    early_aborts: int = 0
    transfer_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def kv_bytes_per_token(
    layers: int, kv_heads: int, head_dim: int, kv_bytes: int = 2
) -> float:
    """K+V per token, all layers, one sequence."""
    return float(layers * kv_heads * head_dim * 2 * kv_bytes)


def bytes_per_token_from_config(model: str) -> float:
    cfg_path = Path(model) / "config.json"
    if not cfg_path.is_file():
        return 147_456.0  # Qwen3-8B GQA bf16 fallback
    cfg = json.loads(cfg_path.read_text())
    return kv_bytes_per_token(
        int(cfg.get("num_hidden_layers", 36)),
        int(cfg.get("num_key_value_heads", cfg.get("num_attention_heads", 8))),
        int(cfg.get("head_dim", 128)),
    )


def gpu_mem_mib() -> list[int]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    vals: list[int] = []
    for line in out.splitlines():
        s = line.strip()
        if s.isdigit():
            vals.append(int(s))
    return vals


def _mib(n_bytes: float) -> float:
    return n_bytes / (1024.0 * 1024.0)


def _now() -> float:
    return time.perf_counter()


# ----- prompts ---------------------------------------------------------------

_CONTEXT = (
    "The repository contains a recursive-descent parser, a type checker, and "
    "a small bytecode VM. Tests fail in test_parser.py around operator "
    "precedence for mixed `and`/`or` expressions and in test_vm.py when "
    "exception handlers unwind across native frames. "
)


def make_trunk(target_tokens: int, tokenize: Callable[[str], Sequence[int]]) -> str:
    from forkserve.adapters.templates import ToolWrappers

    wrap = ToolWrappers()
    system = (
        "You are a senior software engineer. Think step by step, then call tools."
    )
    user = "Fix the failing unit tests.\n\n"
    n = 1
    text = wrap.chat(system, user)
    while len(tokenize(text)) < target_tokens and n < 64:
        user += _CONTEXT
        n += 1
        text = wrap.chat(system, user)
    return text


def thought_residuals(branching: int) -> list[str]:
    from forkserve.adapters.templates import ToolWrappers

    w = ToolWrappers()
    extra = [
        "inspect the Pratt parser table and rewrite mixed boolean operators.",
        "add parentheses in the AST printer and re-run pytest.",
        "log bytecode offsets while unwinding exception handlers.",
        "bisect the last green commit and dump the failing assertion.",
        "replace the recursive visitor with an explicit stack.",
        "stub native frames and unit-test unwind in isolation.",
    ]
    out: list[str] = []
    for i in range(branching):
        out.append(w.thought_prefix(i) + extra[i % len(extra)])
    return out


def react_strings() -> tuple[str, str, str]:
    from forkserve.adapters.templates import ToolWrappers

    w = ToolWrappers()
    wrap = w.observation("bash")
    recov = w.recovery("bash")
    obs = (
        "stdout of bash(pytest -q)\n"
        "failed: test_parser.py::test_mixed_bool  AssertionError\n"
        "failed: test_vm.py::test_unwind  RuntimeError\n"
        + w.close_observation()
    )
    return wrap, recov, obs


# ----- ForkServe path --------------------------------------------------------

def _engine(model: str, tp: int, args: argparse.Namespace):
    from forkserve.api import Engine
    from forkserve.config import ForkServeConfig
    from forkserve.engine.vllm_backend import VllmBackend
    from forkserve.prune import plus_config
    from forkserve.spec_pool import extra_batched_tokens

    bpt = bytes_per_token_from_config(model)
    cfg = ForkServeConfig(
        max_batched_tokens=args.max_batched_tokens,
        hbm_capacity_bytes=40.0 * (1 << 30) * max(tp, 1),
        bytes_per_token=bpt,
        num_workers=1,
    )
    plus = str(getattr(args, "system", "")) == "forkserve_plus"
    if plus:
        cfg = plus_config(cfg)
        cfg.extra["spec_pool_tokens"] = float(
            extra_batched_tokens(cfg.max_batched_tokens, 0.26, cfg.spec_pool_frac)
        )
    backend = VllmBackend(
        config=cfg,
        model=model,
        tensor_parallel=tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_util,
        enable_prefix_caching=True,
        enforce_eager=args.enforce_eager,
        two_class=False,
        cow_blocks=True,
    )
    return Engine(backend, cfg), backend, cfg


def _mock_engine(bpt: float):
    from forkserve.api import Engine
    from forkserve.config import ForkServeConfig
    from forkserve.engine.mock import MockBackend

    cfg = ForkServeConfig(
        max_batched_tokens=4096,
        hbm_capacity_bytes=80.0 * (1 << 30),
        bytes_per_token=bpt,
    )
    return Engine(MockBackend(cfg), cfg), cfg


def _peak_memory(cfg, trunk: int, residuals: Sequence[int], k: int) -> tuple[float, float, float]:
    from forkserve.pages import clone_memory_bytes, cow_memory_bytes

    cow = cow_memory_bytes(trunk, residuals, cfg.bytes_per_token)
    clone = clone_memory_bytes(trunk, residuals, cfg.bytes_per_token, k)
    saving = 0.0 if clone <= 0 else 1.0 - (cow / clone)
    return _mib(cow), _mib(clone), saving


def _fanout_thoughts(eng: Any, session: Any, parent: Any, thoughts: Sequence[str], idle_ms: float) -> list[Any]:
    """Fork the same residual strings the vLLM baselines prefill."""
    from forkserve.planner import Candidate
    from forkserve.types import BranchId, SchemaKind

    kids = []
    cands: list[Candidate] = []
    k = max(len(thoughts), 1)
    for i, text in enumerate(thoughts):
        nid = eng.fork(session, parent, f"thought-{i}", text)
        kids.append(nid)
        cands.append(
            Candidate(
                branch_id=BranchId(f"thought-{i}"),
                node_id=nid,
                known=eng._tok(text),
                p_b=1.0 / k,
                schema=SchemaKind.FREEFORM,
                declared=True,
            )
        )
    eng.speculate_set(session, parent, cands, t_idle_ms=idle_ms)
    eng.drain_slack()
    return kids


def _resident_kv(eng: Any, handles: Sequence[Any] | None = None) -> int:
    """KV after loser abort: shared trunk + committed winner residual only."""
    forest = getattr(eng, "forest", None)
    if forest is not None and handles is None:
        return int(forest.resident_kv_tokens())
    if handles:
        return sum(int(eng.tree(h.id).live_kv_tokens()) for h in handles)
    return 0


def _select_winner(eng: Any, session: Any, kids: Sequence[Any], winner: int = 0) -> Any:
    """Abort losers and keep the winner tip — no join node (preserves CoW KV)."""
    keep = kids[winner]
    for i, cid in enumerate(kids):
        if i != winner:
            eng.abort(session, cid)
    promote = getattr(eng, "promote", None)
    if callable(promote):
        return promote(session, keep)
    from forkserve.types import JoinPolicy

    return eng.join(session, [keep], JoinPolicy.WINNER).node_id


def run_tot_forkserve(
    eng: Any,
    cfg: Any,
    trunk: str,
    thoughts: Sequence[str],
    *,
    decode_n: int,
    idle_ms: float,
    workload: str = "tot",
) -> RunMetrics:
    t0 = _now()
    h = eng.open(trunk)
    tree = eng.tree(h.id)
    trunk_n = len(tree.get(h.tip).tokens)
    t_open = _now()
    kids = _fanout_thoughts(eng, h.id, h.tip, thoughts, idle_ms)
    residuals = [len(tree.get(k).residual) for k in kids]
    t_fan = _now()
    _select_winner(eng, h.id, kids, winner=0)
    peak = tree.live_kv_tokens()
    out = eng.generate(h.id, decode_n)
    t1 = _now()
    cow, clone, saving = _peak_memory(cfg, trunk_n, residuals, len(thoughts))
    m = eng.metrics[h.id]
    return RunMetrics(
        system="forkserve",
        workload=workload,
        tp=0,
        e2e_ms=(t1 - t0) * 1000.0,
        fanout_ms=(t_fan - t_open) * 1000.0,
        decode_ms=(t1 - t_fan) * 1000.0,
        trunk_tokens=trunk_n,
        residual_tokens=residuals,
        peak_kv_tokens=int(peak),
        m_cow_mib=cow,
        m_clone_mib=clone,
        kv_saving=saving,
        gpu_mem_mib=gpu_mem_mib(),
        known_suffix_hit_rate=m.known_suffix_hit_rate,
        decode_tokens=len(out),
        decode_ids=_decode_ids(out),
        branching=len(thoughts),
        notes="open + CoW fan-out; abort losers; peak is committed spine",
    )


def run_react_forkserve(
    eng: Any,
    cfg: Any,
    trunk: str,
    *,
    decode_n: int,
    idle_ms: float,
    wrap: str | None = None,
    recov: str | None = None,
    obs: str | None = None,
    workload: str = "react",
) -> RunMetrics:
    from forkserve.adapters.react import ReActAdapter
    from forkserve.adapters.templates import ToolWrappers

    if wrap is None or recov is None or obs is None:
        wrap, recov, obs = react_strings()
    t0 = _now()
    h = eng.open(trunk)
    tree = eng.tree(h.id)
    parent = h.tip
    trunk_n = len(tree.get(parent).tokens)
    ad = ReActAdapter(eng, ToolWrappers())
    t_idle = _now()
    happy, fail = ad.on_tool_parsed(
        h.id, parent, "bash", t_idle_ms=idle_ms, include_recovery=True
    )
    eng.drain_slack()
    spec_ms = (_now() - t_idle) * 1000.0
    remain = idle_ms - spec_ms
    if remain > 0:
        time.sleep(remain / 1000.0)
    if fail is not None:
        eng.abort(h.id, fail)
    eng.promote(h.id, happy)
    peak = int(tree.live_kv_tokens())
    t_obs = _now()
    ad.bind_observation(h.id, parent, "bash", obs, ok=True)
    out = eng.generate(h.id, decode_n)
    ttft = (_now() - t_obs) * 1000.0
    t1 = _now()
    wrap_n = len(eng._tok(wrap))
    recov_n = len(eng._tok(recov))
    obs_n = len(eng._tok(obs))
    cow, clone, saving = _peak_memory(cfg, trunk_n, [wrap_n, recov_n], 2)
    m = eng.metrics[h.id]
    return RunMetrics(
        system="forkserve",
        workload=workload,
        tp=0,
        e2e_ms=(t1 - t0) * 1000.0,
        fanout_ms=spec_ms,
        decode_ms=ttft,
        ttft_from_obs_ms=ttft,
        spec_ms=spec_ms,
        idle_ms=idle_ms,
        trunk_tokens=trunk_n,
        residual_tokens=[wrap_n, recov_n, obs_n],
        peak_kv_tokens=peak,
        m_cow_mib=cow,
        m_clone_mib=clone,
        kv_saving=saving,
        gpu_mem_mib=gpu_mem_mib(),
        known_suffix_hit_rate=m.known_suffix_hit_rate,
        decode_tokens=len(out),
        decode_ids=_decode_ids(out),
        branching=2,
        notes="speculate wrap during idle; TTFT measured from observation arrival",
    )


def run_multi_forkserve(
    eng: Any,
    cfg: Any,
    trunks: Sequence[str],
    thoughts: Sequence[str],
    *,
    decode_n: int,
    idle_ms: float,
) -> RunMetrics:
    t0 = _now()
    handles = [eng.open(t) for t in trunks]
    t_open = _now()
    peaks: list[int] = []
    residuals_all: list[int] = []
    trunk_n = 0
    for h in handles:
        tree = eng.tree(h.id)
        trunk_n = len(tree.get(h.tip).tokens)
        kids = _fanout_thoughts(eng, h.id, h.tip, thoughts, idle_ms)
        residuals_all.extend(len(tree.get(k).residual) for k in kids)
        _select_winner(eng, h.id, kids, winner=0)
        peaks.append(int(tree.live_kv_tokens()))
    t_fan = _now()
    n_out = 0
    decoded: list[list[int]] = []
    for h in handles:
        out = eng.generate(h.id, decode_n)
        decoded.append([int(t) for t in out])
        n_out += len(out)
    t1 = _now()
    k = len(handles) * len(thoughts)
    cow, clone, saving = _peak_memory(
        cfg, trunk_n * len(handles), residuals_all, k
    )
    return RunMetrics(
        system="forkserve",
        workload="multi",
        tp=0,
        e2e_ms=(t1 - t0) * 1000.0,
        fanout_ms=(t_fan - t_open) * 1000.0,
        decode_ms=(t1 - t_fan) * 1000.0,
        trunk_tokens=trunk_n,
        residual_tokens=residuals_all,
        peak_kv_tokens=int(sum(peaks)),
        m_cow_mib=cow,
        m_clone_mib=clone,
        kv_saving=saving,
        gpu_mem_mib=gpu_mem_mib(),
        decode_tokens=n_out,
        decode_ids=decoded,
        sessions=len(handles),
        branching=len(thoughts),
        notes="S independent ToT trees on one engine",
    )


# ----- vLLM baselines --------------------------------------------------------

def _vllm_llm(model: str, tp: int, args: argparse.Namespace, prefix_cache: bool):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=model,
        tensor_parallel_size=tp,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_util,
        enable_prefix_caching=prefix_cache,
        enable_chunked_prefill=True,
        max_num_batched_tokens=args.max_batched_tokens,
        enforce_eager=args.enforce_eager,
    )
    return llm, SamplingParams, TokensPrompt


def _tok_ids(llm: Any, text: str) -> list[int]:
    ids = llm.get_tokenizer().encode(text, add_special_tokens=False)
    return [int(i) for i in ids]


def _gen(llm: Any, SamplingParams: Any, TokensPrompt: Any, seqs: Sequence[Sequence[int]], n: int) -> list[list[int]]:
    prompts = [TokensPrompt(prompt_token_ids=list(s)) for s in seqs]
    mt = int(os.environ.get("FORKSERVE_MIN_TOKENS", "0") or 0)
    kwargs: dict[str, Any] = dict(max_tokens=n, temperature=0.0)
    if mt > 0:
        kwargs["min_tokens"] = min(mt, n)
    params = SamplingParams(**kwargs)
    outs = llm.generate(prompts, params, use_tqdm=False)
    return [[int(t) for t in o.outputs[0].token_ids] for o in outs]


def _decode_ids(*seqs: Sequence[int]) -> list[list[int]]:
    return [[int(t) for t in s] for s in seqs if s is not None]


def _fs_texts(eng: Any, seqs: Sequence[Sequence[int]]) -> list[str]:
    return [eng.backend.detokenize(tuple(int(t) for t in s)) for s in seqs]


def _vllm_texts(llm: Any, seqs: Sequence[Sequence[int]]) -> list[str]:
    tok = llm.get_tokenizer()
    return [tok.decode(list(s), skip_special_tokens=True) for s in seqs]


def _set_golds(row: RunMetrics, golds: Sequence[str], texts: Sequence[str], prompts: Sequence[str] = ()) -> RunMetrics:
    row.golds = [str(g) for g in golds]
    row.decode_texts = [str(t) for t in texts]
    row.gold_prompts = [str(p) for p in prompts]
    if row.decode_per_item <= 0 and texts:
        row.decode_per_item = max((len(t) for t in row.decode_ids), default=0) if row.decode_ids else 0
    return row


def _chunk_size(args: argparse.Namespace) -> int:
    return max(1, int(getattr(args, "chunk", 4) or 4))


def _iter_chunks(items: Sequence[Any], args: argparse.Namespace):
    n = _chunk_size(args)
    for i in range(0, len(items), n):
        yield i, items[i : i + n]


def _log_forest_chunk(
    args: argparse.Namespace,
    system: str,
    workload: str,
    start: int,
    batch: Sequence[Any],
    n_total: int,
    row: RunMetrics,
    parts: Sequence[RunMetrics],
) -> None:
    texts = [t for p in parts for t in p.decode_texts]
    golds = [g for p in parts for g in p.golds]
    prompts = [x for p in parts for x in p.gold_prompts]
    progress(
        args,
        f"{_sys_tp(args, system)} forest {workload} {start + len(batch)}/{n_total} "
        f"+{len(batch)} peak={row.peak_kv_tokens} fanout={row.fanout_ms:.0f}ms "
        f"{_score_so_far(workload, texts, golds, prompts)}",
    )


def merge_metrics(parts: Sequence[RunMetrics]) -> RunMetrics:
    if not parts:
        raise ValueError("no chunks to merge")
    if len(parts) == 1:
        return parts[0]
    a = parts[0]
    texts = [t for p in parts for t in p.decode_texts]
    golds = [g for p in parts for g in p.golds]
    prompts = [x for p in parts for x in p.gold_prompts]
    ids = [x for p in parts for x in p.item_ids]
    decode_ids = [x for p in parts for x in p.decode_ids]
    out = RunMetrics(
        system=a.system,
        workload=a.workload,
        tp=a.tp,
        e2e_ms=sum(p.e2e_ms for p in parts),
        fanout_ms=sum(p.fanout_ms for p in parts),
        decode_ms=sum(p.decode_ms for p in parts),
        ttft_from_obs_ms=sum(p.ttft_from_obs_ms for p in parts),
        spec_ms=sum(p.spec_ms for p in parts),
        idle_ms=a.idle_ms,
        trunk_tokens=a.trunk_tokens,
        residual_tokens=[],
        peak_kv_tokens=max(p.peak_kv_tokens for p in parts),
        m_cow_mib=max(p.m_cow_mib for p in parts),
        m_clone_mib=max(p.m_clone_mib for p in parts),
        kv_saving=sum(p.kv_saving for p in parts) / len(parts),
        gpu_mem_mib=parts[-1].gpu_mem_mib,
        known_suffix_hit_rate=sum(p.known_suffix_hit_rate for p in parts) / len(parts),
        decode_tokens=sum(p.decode_tokens for p in parts),
        decode_per_item=a.decode_per_item,
        sessions=sum(p.sessions for p in parts),
        branching=a.branching,
        notes=f"{sum(p.sessions for p in parts)} items; {len(parts)} chunks; {a.notes}",
        decode_ids=decode_ids,
        item_ids=ids,
        cow_ms=sum(p.cow_ms for p in parts),
        prefill_ms=sum(p.prefill_ms for p in parts),
        abort_mark_ms=sum(p.abort_mark_ms for p in parts),
        abort_reclaim_ms=sum(p.abort_reclaim_ms for p in parts),
        pointer_swaps=sum(p.pointer_swaps for p in parts),
        pruned_branches=sum(p.pruned_branches for p in parts),
        prefilled_branches=sum(p.prefilled_branches for p in parts),
        hash_skips=sum(p.hash_skips for p in parts),
        early_aborts=sum(p.early_aborts for p in parts),
        transfer_tokens=sum(p.transfer_tokens for p in parts),
    )
    return _set_golds(out, golds, texts, prompts)


def _quality_generate_forkserve(eng: Any, prompts: Sequence[str], decode_n: int) -> tuple[list[str], list[list[int]]]:
    handles = [eng.open(p, flush=True) for p in prompts]
    if hasattr(eng, "generate_many"):
        outs = eng.generate_many([h.id for h in handles], decode_n)
    else:
        outs = [eng.generate(h.id, decode_n) for h in handles]
    ids = [[int(t) for t in o] for o in outs]
    return _fs_texts(eng, ids), ids


def _quality_generate_vllm(
    llm: Any,
    SamplingParams: Any,
    TokensPrompt: Any,
    prompts: Sequence[str],
    decode_n: int,
) -> tuple[list[str], list[list[int]]]:
    seqs = [_tok_ids(llm, p) for p in prompts]
    decoded = _gen(llm, SamplingParams, TokensPrompt, seqs, decode_n)
    return _vllm_texts(llm, decoded), decoded


def run_quality_forkserve(
    eng: Any,
    args: argparse.Namespace,
    workload: str,
    problems: Sequence[Any],
    prompts: Sequence[str],
    golds: Sequence[str],
    gold_prompts: Sequence[str] = (),
    decode_n: int | None = None,
) -> RunMetrics:
    """Single-path generate + official metric. Used for full-set solving ability."""
    n_dec = int(decode_n if decode_n is not None else workload_decode(args, workload))
    texts: list[str] = []
    e2e = 0.0
    peak = 0
    n_tok = 0
    t_job = _now()
    progress(
        args,
        f"start {_sys_tp(args, 'forkserve')} {workload} n={len(problems)} "
        f"chunk={_chunk_size(args)} decode={n_dec} quality-only",
    )
    for start, batch in _iter_chunks(problems, args):
        batch_p = prompts[start : start + len(batch)]
        t0 = _now()
        chunk_texts, ids = _quality_generate_forkserve(eng, batch_p, n_dec)
        dt = _now() - t0
        e2e += dt * 1000.0
        texts.extend(chunk_texts)
        n_tok += sum(len(x) for x in ids)
        done = len(texts)
        progress(
            args,
            f"{_sys_tp(args, 'forkserve')} {workload} {done}/{len(problems)} "
            f"+{len(batch)} {_score_so_far(workload, texts, golds, gold_prompts)} "
            f"chunk={dt:.1f}s total={e2e / 1000.0:.1f}s "
            f"{_eta_s(done, len(problems), _now() - t_job)}",
        )
        if hasattr(eng, "forest"):
            live = 0
            for sid in list(eng.forest.sessions):
                try:
                    live += int(eng.tree(sid).live_kv_tokens())
                except Exception:
                    pass
            peak = max(peak, live)
        _close_all(eng)
    row = RunMetrics(
        system="forkserve",
        workload=workload,
        tp=0,
        e2e_ms=e2e,
        decode_ms=e2e,
        peak_kv_tokens=int(peak),
        decode_tokens=n_tok,
        decode_per_item=n_dec,
        sessions=len(problems),
        branching=1,
        notes=f"{len(problems)} items; quality-only single-path generate",
        item_ids=[str(getattr(p, "item_id", i)) for i, p in enumerate(problems)],
    )
    return _set_golds(row, golds, texts, gold_prompts)


def run_quality_vllm(
    llm: Any,
    SamplingParams: Any,
    TokensPrompt: Any,
    system: str,
    args: argparse.Namespace,
    workload: str,
    problems: Sequence[Any],
    prompts: Sequence[str],
    golds: Sequence[str],
    gold_prompts: Sequence[str] = (),
    decode_n: int | None = None,
) -> RunMetrics:
    n_dec = int(decode_n if decode_n is not None else workload_decode(args, workload))
    texts: list[str] = []
    e2e = 0.0
    n_tok = 0
    t_job = _now()
    progress(
        args,
        f"start {_sys_tp(args, system)} {workload} n={len(problems)} "
        f"chunk={_chunk_size(args)} decode={n_dec} quality-only",
    )
    for start, batch in _iter_chunks(problems, args):
        batch_p = prompts[start : start + len(batch)]
        t0 = _now()
        chunk_texts, ids = _quality_generate_vllm(
            llm, SamplingParams, TokensPrompt, batch_p, n_dec
        )
        dt = _now() - t0
        e2e += dt * 1000.0
        texts.extend(chunk_texts)
        n_tok += sum(len(x) for x in ids)
        done = len(texts)
        progress(
            args,
            f"{_sys_tp(args, system)} {workload} {done}/{len(problems)} "
            f"+{len(batch)} {_score_so_far(workload, texts, golds, gold_prompts)} "
            f"chunk={dt:.1f}s total={e2e / 1000.0:.1f}s "
            f"{_eta_s(done, len(problems), _now() - t_job)}",
        )
    row = RunMetrics(
        system=system,
        workload=workload,
        tp=0,
        e2e_ms=e2e,
        decode_ms=e2e,
        peak_kv_tokens=0,
        decode_tokens=n_tok,
        decode_per_item=n_dec,
        sessions=len(problems),
        branching=1,
        notes=f"{len(problems)} items; quality-only single-path generate",
        item_ids=[str(getattr(p, "item_id", i)) for i, p in enumerate(problems)],
    )
    return _set_golds(row, golds, texts, gold_prompts)


def _humaneval_complete_forkserve(eng: Any, problems: Sequence[Any], decode_n: int) -> tuple[list[str], list[list[int]]]:
    """Official HumanEval completion (function body), not the ReAct/chat tail."""
    handles = [eng.open(item.prompt, flush=True) for item in problems]
    if hasattr(eng, "generate_many"):
        outs = eng.generate_many([h.id for h in handles], decode_n)
    else:
        outs = [eng.generate(h.id, decode_n) for h in handles]
    ids = [[int(t) for t in o] for o in outs]
    return _fs_texts(eng, ids), ids


def _humaneval_complete_vllm(
    llm: Any,
    SamplingParams: Any,
    TokensPrompt: Any,
    problems: Sequence[Any],
    decode_n: int,
) -> tuple[list[str], list[list[int]]]:
    prompts = [_tok_ids(llm, item.prompt) for item in problems]
    decoded = _gen(llm, SamplingParams, TokensPrompt, prompts, decode_n)
    return _vllm_texts(llm, decoded), decoded


def _warmup_vllm(llm: Any, SamplingParams: Any, TokensPrompt: Any) -> None:
    _gen(llm, SamplingParams, TokensPrompt, [[1, 2, 3, 4]], 1)


def run_tot_vllm(
    llm: Any,
    SamplingParams: Any,
    TokensPrompt: Any,
    system: str,
    cfg_bpt: float,
    trunk: str,
    thoughts: Sequence[str],
    *,
    decode_n: int,
    prefix_cache: bool,
    workload: str = "tot",
) -> RunMetrics:
    from forkserve.config import ForkServeConfig
    from forkserve.pages import clone_memory_bytes, cow_memory_bytes

    trunk_ids = _tok_ids(llm, trunk)
    res_ids = [_tok_ids(llm, t) for t in thoughts]
    branches = [trunk_ids + r for r in res_ids]
    residuals = [len(r) for r in res_ids]
    trunk_n = len(trunk_ids)
    k = len(thoughts)
    t0 = _now()
    if prefix_cache:
        _gen(llm, SamplingParams, TokensPrompt, [trunk_ids], 1)
    t_open = _now()
    _gen(llm, SamplingParams, TokensPrompt, branches, 1)
    t_fan = _now()
    decoded = _gen(llm, SamplingParams, TokensPrompt, [branches[0]], decode_n)
    t1 = _now()
    if prefix_cache:
        peak = trunk_n + sum(residuals)
    else:
        peak = sum(trunk_n + r for r in residuals)
    cfg = ForkServeConfig(bytes_per_token=cfg_bpt)
    cow = _mib(cow_memory_bytes(trunk_n, residuals, cfg_bpt))
    clone = _mib(clone_memory_bytes(trunk_n, residuals, cfg_bpt, k))
    return RunMetrics(
        system=system,
        workload=workload,
        tp=0,
        e2e_ms=(t1 - t0) * 1000.0,
        fanout_ms=(t_fan - t_open) * 1000.0,
        decode_ms=(t1 - t_fan) * 1000.0,
        trunk_tokens=trunk_n,
        residual_tokens=residuals,
        peak_kv_tokens=int(peak),
        m_cow_mib=cow,
        m_clone_mib=clone,
        kv_saving=0.0 if clone <= 0 else 1.0 - (cow / clone),
        gpu_mem_mib=gpu_mem_mib(),
        decode_tokens=decode_n,
        decode_ids=decoded,
        branching=k,
        notes="APC warms trunk then batches branches" if prefix_cache else "full trunk x B, no prefix cache",
    )


def run_react_vllm(
    llm: Any,
    SamplingParams: Any,
    TokensPrompt: Any,
    system: str,
    cfg_bpt: float,
    trunk: str,
    *,
    decode_n: int,
    idle_ms: float,
    prefix_cache: bool,
    wrap: str | None = None,
    recov: str | None = None,
    obs: str | None = None,
    workload: str = "react",
) -> RunMetrics:
    if wrap is None or recov is None or obs is None:
        wrap, recov, obs = react_strings()
    trunk_ids = _tok_ids(llm, trunk)
    wrap_ids = _tok_ids(llm, wrap)
    recov_ids = _tok_ids(llm, recov)
    obs_ids = _tok_ids(llm, obs)
    full = trunk_ids + wrap_ids + obs_ids
    t0 = _now()
    if prefix_cache:
        _gen(llm, SamplingParams, TokensPrompt, [trunk_ids], 1)
    t_idle = _now()
    _gen(llm, SamplingParams, TokensPrompt, [trunk_ids + wrap_ids, trunk_ids + recov_ids], 1)
    spec_ms = (_now() - t_idle) * 1000.0
    remain = idle_ms - spec_ms
    if remain > 0:
        time.sleep(remain / 1000.0)
    t_obs = _now()
    decoded = _gen(llm, SamplingParams, TokensPrompt, [full], decode_n)
    ttft = (_now() - t_obs) * 1000.0
    t1 = _now()
    trunk_n = len(trunk_ids)
    residuals = [len(wrap_ids), len(recov_ids)]
    from forkserve.pages import clone_memory_bytes, cow_memory_bytes

    cow = _mib(cow_memory_bytes(trunk_n, residuals, cfg_bpt))
    clone = _mib(clone_memory_bytes(trunk_n, residuals, cfg_bpt, 2))
    peak = (
        trunk_n + len(wrap_ids) + len(recov_ids)
        if prefix_cache
        else 2 * trunk_n + len(wrap_ids) + len(recov_ids)
    )
    return RunMetrics(
        system=system,
        workload=workload,
        tp=0,
        e2e_ms=(t1 - t0) * 1000.0,
        fanout_ms=spec_ms,
        decode_ms=ttft,
        ttft_from_obs_ms=ttft,
        spec_ms=spec_ms,
        idle_ms=idle_ms,
        trunk_tokens=trunk_n,
        residual_tokens=residuals + [len(obs_ids)],
        peak_kv_tokens=int(peak),
        m_cow_mib=cow,
        m_clone_mib=clone,
        kv_saving=0.0 if clone <= 0 else 1.0 - (cow / clone),
        gpu_mem_mib=gpu_mem_mib(),
        decode_tokens=decode_n,
        decode_ids=decoded,
        branching=2,
        notes="idle fan-out wrap+recovery; then wrap+obs decode",
    )


def run_multi_vllm(
    llm: Any,
    SamplingParams: Any,
    TokensPrompt: Any,
    system: str,
    cfg_bpt: float,
    trunks: Sequence[str],
    thoughts: Sequence[str],
    *,
    decode_n: int,
    prefix_cache: bool,
) -> RunMetrics:
    trunk_ids = [_tok_ids(llm, t) for t in trunks]
    res_ids = [_tok_ids(llm, th) for th in thoughts]
    branches = [t + r for t in trunk_ids for r in res_ids]
    residuals = [len(r) for _ in trunks for r in res_ids]
    trunk_n = len(trunk_ids[0]) if trunk_ids else 0
    t0 = _now()
    if prefix_cache:
        _gen(llm, SamplingParams, TokensPrompt, trunk_ids, 1)
    t_open = _now()
    _gen(llm, SamplingParams, TokensPrompt, branches, 1)
    t_fan = _now()
    winners = [t + res_ids[0] for t in trunk_ids]
    decoded = _gen(llm, SamplingParams, TokensPrompt, winners, decode_n)
    t1 = _now()
    k = len(branches)
    from forkserve.pages import clone_memory_bytes, cow_memory_bytes

    cow = _mib(cow_memory_bytes(trunk_n * len(trunks), residuals, cfg_bpt))
    clone = _mib(clone_memory_bytes(trunk_n, [len(r) for r in res_ids], cfg_bpt, k))
    peak = (
        sum(len(t) for t in trunk_ids) + sum(residuals)
        if prefix_cache
        else sum(len(b) for b in branches)
    )
    return RunMetrics(
        system=system,
        workload="multi",
        tp=0,
        e2e_ms=(t1 - t0) * 1000.0,
        fanout_ms=(t_fan - t_open) * 1000.0,
        decode_ms=(t1 - t_fan) * 1000.0,
        trunk_tokens=trunk_n,
        residual_tokens=residuals,
        peak_kv_tokens=int(peak),
        m_cow_mib=cow,
        m_clone_mib=clone,
        kv_saving=0.0 if clone <= 0 else 1.0 - (cow / clone),
        gpu_mem_mib=gpu_mem_mib(),
        decode_tokens=decode_n * len(trunks),
        decode_ids=decoded,
        sessions=len(trunks),
        branching=len(thoughts),
        notes="SxB independent prompts in one generate()",
    )


def _merge_metrics(parts: list[RunMetrics], workload: str) -> RunMetrics:
    n = len(parts)
    if n == 0:
        raise ValueError("no runs to merge")
    head = parts[0]
    residuals: list[int] = []
    for p in parts:
        residuals.extend(p.residual_tokens)
    clone = sum(p.m_clone_mib for p in parts) / n
    cow = sum(p.m_cow_mib for p in parts) / n
    return RunMetrics(
        system=head.system,
        workload=workload,
        tp=head.tp,
        e2e_ms=sum(p.e2e_ms for p in parts),
        fanout_ms=sum(p.fanout_ms for p in parts),
        decode_ms=sum(p.decode_ms for p in parts),
        ttft_from_obs_ms=sum(p.ttft_from_obs_ms for p in parts) / n,
        spec_ms=sum(p.spec_ms for p in parts),
        idle_ms=head.idle_ms,
        trunk_tokens=int(sum(p.trunk_tokens for p in parts) / n),
        residual_tokens=residuals,
        peak_kv_tokens=max(p.peak_kv_tokens for p in parts),
        m_cow_mib=cow,
        m_clone_mib=clone,
        kv_saving=0.0 if clone <= 0 else 1.0 - (cow / clone),
        gpu_mem_mib=parts[-1].gpu_mem_mib,
        known_suffix_hit_rate=sum(p.known_suffix_hit_rate for p in parts) / n,
        decode_tokens=sum(p.decode_tokens for p in parts),
        decode_per_item=head.decode_per_item,
        decode_ids=[ids for p in parts for ids in p.decode_ids],
        sessions=n,
        branching=head.branching,
        notes=f"{n} items; {head.notes}",
    )


def run_tot_forest_forkserve(
    eng: Any,
    cfg: Any,
    trunks: Sequence[str],
    thoughts: Sequence[str],
    *,
    decode_n: int,
    idle_ms: float,
    workload: str,
) -> RunMetrics:
    """One open / fan-out / decode ``LLM.generate`` for the whole slice."""
    # Match the vLLM baseline: tokenize off the timed path.
    trunk_ids = [eng._tok(t) for t in trunks]
    thought_ids = [eng._tok(th) for th in thoughts]
    t0 = _now()
    # Do not flush trunks alone. Children send trunk+residual; the backend
    # drops covered prefixes so fan-out is one generate (APC uses two).
    handles = [eng.open(ids, flush=False) for ids in trunk_ids]
    t_open = _now()
    all_kids: list[list[Any]] = []
    residuals: list[int] = []
    trunk_n = 0
    k = max(len(thought_ids), 1)
    plus = bool(getattr(eng.config, "prune_enabled", False) or getattr(eng.config, "lazy_abort", False))
    sys_name = "forkserve_plus" if plus else "forkserve"
    t_cow0 = _now()
    for h in handles:
        tree = eng.tree(h.id)
        trunk_n = len(tree.get(h.tip).tokens)
        kids: list[Any] = []
        for i, known in enumerate(thought_ids):
            nid = eng.fork(h.id, h.tip, f"thought-{i}", known)
            kids.append(nid)
            residuals.append(len(known))
        all_kids.append(kids)
        if hasattr(eng, "queue_fanout_prefills"):
            eng.queue_fanout_prefills(h.id, kids, list(thoughts), winner=0)
        else:
            for nid in kids:
                eng.queue_known_prefill(h.id, nid)
    t_cow = _now()
    eng.flush()
    t_fan = _now()
    t_ab0 = _now()
    for h, kids in zip(handles, all_kids, strict=True):
        _select_winner(eng, h.id, kids, winner=0)
    t_ab1 = _now()
    peak = _resident_kv(eng, handles)
    from forkserve.answer_stop import stop_mode_for

    prev_mode = str(getattr(eng.config, "answer_stop", "") or "")
    mode = stop_mode_for(workload, prev_mode)
    eng.config.answer_stop = mode
    try:
        if hasattr(eng, "generate_many"):
            outs = eng.generate_many([h.id for h in handles], decode_n)
        else:
            outs = [eng.generate(h.id, decode_n) for h in handles]
    finally:
        eng.config.answer_stop = prev_mode
    n_out = sum(len(o) for o in outs)
    t1 = _now()
    cow, clone, saving = _peak_memory(cfg, trunk_n * len(handles), residuals, len(handles) * k)
    hits = [eng.metrics[h.id].known_suffix_hit_rate for h in handles]
    bd = getattr(eng, "last_fanout", None)
    return RunMetrics(
        system=sys_name,
        workload=workload,
        tp=0,
        e2e_ms=(t1 - t0) * 1000.0,
        fanout_ms=(t_fan - t_open) * 1000.0 + (t_ab1 - t_ab0) * 1000.0,
        decode_ms=(t1 - t_ab1) * 1000.0,
        trunk_tokens=trunk_n,
        residual_tokens=residuals,
        peak_kv_tokens=int(peak),
        m_cow_mib=cow,
        m_clone_mib=clone,
        kv_saving=saving,
        gpu_mem_mib=gpu_mem_mib(),
        known_suffix_hit_rate=sum(hits) / max(len(hits), 1),
        decode_tokens=n_out,
        decode_per_item=decode_n,
        decode_ids=[[int(t) for t in o] for o in outs],
        sessions=len(handles),
        branching=len(thoughts),
        notes=(
            f"{len(handles)} items; CoW fan-out then "
            f"{'lazy' if plus else 'sync'} abort losers; peak is winner spine"
        ),
        cow_ms=(t_cow - t_cow0) * 1000.0,
        prefill_ms=(t_fan - t_cow) * 1000.0,
        abort_mark_ms=(t_ab1 - t_ab0) * 1000.0,
        pointer_swaps=int(getattr(bd, "pointer_swaps", 0) or 0),
        pruned_branches=int(getattr(bd, "pruned", 0) or 0),
        prefilled_branches=int(getattr(bd, "prefilled_branches", 0) or 0),
        hash_skips=int(getattr(bd, "hash_skips", 0) or 0),
        early_aborts=int(getattr(bd, "early_aborts", 0) or 0),
        transfer_tokens=int(getattr(bd, "transfer_tokens", 0) or 0),
    )


def run_react_forest_forkserve(
    eng: Any,
    cfg: Any,
    items: Sequence[tuple[str, str, str, str]],
    *,
    decode_n: int,
    idle_ms: float,
    workload: str,
) -> RunMetrics:
    """Speculate wraps in one idle generate; bind+decode is one generate."""
    from forkserve.adapters.react import ReActAdapter
    from forkserve.adapters.templates import ToolWrappers

    ad = ReActAdapter(eng, ToolWrappers())
    # Tokenize off the timed / TTFT path (same as the vLLM forest baseline).
    prepared = []
    for trunk, wrap, recov, obs in items:
        # Same token sequence as the vLLM forest baseline (wrap+obs, no extra close).
        prompt = wrap + obs
        prepared.append((eng._tok(trunk), wrap, recov, obs, eng._tok(prompt)))
    t0 = _now()
    handles = []
    for trunk_ids, wrap, recov, obs, commit_ids in prepared:
        h = eng.open(trunk_ids, flush=False)
        handles.append((h, wrap, recov, obs, commit_ids))
    t_idle = _now()
    spawned: list[tuple[Any, Any, Any]] = []
    for h, wrap, recov, obs, _commit in handles:
        happy, fail = ad.on_tool_parsed(
            h.id, h.tip, "bash", t_idle_ms=idle_ms, include_recovery=True
        )
        spawned.append((h, happy, fail))
    eng.drain_slack()
    spec_ms = (_now() - t_idle) * 1000.0
    remain = idle_ms - spec_ms
    slept = 0.0
    if remain > 0:
        time.sleep(remain / 1000.0)
        slept = remain
    for h, happy, fail in spawned:
        if fail is not None:
            eng.abort(h.id, fail)
        eng.promote(h.id, happy)
    peak = sum(int(eng.tree(h.id).live_kv_tokens()) for h, *_ in handles)
    t_obs = _now()
    for h, wrap, recov, obs, commit_ids in handles:
        eng.commit(h.id, h.tip, commit_ids, preferred_bid="bash")
    if hasattr(eng, "generate_many"):
        outs = eng.generate_many([h.id for h, *_ in handles], decode_n)
    else:
        outs = [eng.generate(h.id, decode_n) for h, *_ in handles]
    n_out = sum(len(o) for o in outs)
    ttft = (_now() - t_obs) * 1000.0
    t1 = _now()
    residuals: list[int] = []
    trunk_n = 0
    for h, wrap, recov, obs, _commit in handles:
        tree = eng.tree(h.id)
        parent = h.tip
        trunk_n = len(tree.get(parent).tokens)
        residuals.extend([len(eng._tok(wrap)), len(eng._tok(recov)), len(eng._tok(obs))])
    cow, clone, saving = _peak_memory(
        cfg, trunk_n * len(handles), residuals, 2 * len(handles)
    )
    hits = [eng.metrics[h.id].known_suffix_hit_rate for h, *_ in handles]
    return RunMetrics(
        system="forkserve",
        workload=workload,
        tp=0,
        e2e_ms=(t1 - t0) * 1000.0 - slept,
        fanout_ms=spec_ms,
        decode_ms=ttft,
        ttft_from_obs_ms=ttft,
        spec_ms=spec_ms,
        idle_ms=idle_ms,
        trunk_tokens=trunk_n,
        residual_tokens=residuals,
        peak_kv_tokens=int(peak),
        m_cow_mib=cow,
        m_clone_mib=clone,
        kv_saving=saving,
        gpu_mem_mib=gpu_mem_mib(),
        known_suffix_hit_rate=sum(hits) / max(len(hits), 1),
        decode_tokens=n_out,
        decode_per_item=decode_n,
        decode_ids=[[int(t) for t in o] for o in outs],
        sessions=len(handles),
        branching=2,
        notes=f"{len(handles)} items; idle CoW fan-out wrap+recovery; TTFT after LCP commit",
    )


def run_tot_forest_vllm(
    llm: Any,
    SamplingParams: Any,
    TokensPrompt: Any,
    system: str,
    cfg_bpt: float,
    trunks: Sequence[str],
    thoughts: Sequence[str],
    *,
    decode_n: int,
    prefix_cache: bool,
    workload: str,
) -> RunMetrics:
    trunk_ids = [_tok_ids(llm, t) for t in trunks]
    res_ids = [_tok_ids(llm, th) for th in thoughts]
    branches = [t + r for t in trunk_ids for r in res_ids]
    residuals = [len(r) for _ in trunks for r in res_ids]
    trunk_n = len(trunk_ids[0]) if trunk_ids else 0
    t0 = _now()
    if prefix_cache:
        _gen(llm, SamplingParams, TokensPrompt, trunk_ids, 1)
    t_open = _now()
    _gen(llm, SamplingParams, TokensPrompt, branches, 1)
    t_fan = _now()
    winners = [t + res_ids[0] for t in trunk_ids]
    decoded = _gen(llm, SamplingParams, TokensPrompt, winners, decode_n)
    t1 = _now()
    k = len(branches)
    from forkserve.pages import clone_memory_bytes, cow_memory_bytes

    cow = _mib(cow_memory_bytes(trunk_n * len(trunks), residuals, cfg_bpt))
    clone = _mib(clone_memory_bytes(trunk_n * len(trunks), residuals, cfg_bpt, k))
    peak = (
        sum(len(t) for t in trunk_ids) + sum(residuals)
        if prefix_cache
        else sum(len(b) for b in branches)
    )
    return RunMetrics(
        system=system,
        workload=workload,
        tp=0,
        e2e_ms=(t1 - t0) * 1000.0,
        fanout_ms=(t_fan - t_open) * 1000.0,
        decode_ms=(t1 - t_fan) * 1000.0,
        trunk_tokens=trunk_n,
        residual_tokens=residuals,
        peak_kv_tokens=int(peak),
        m_cow_mib=cow,
        m_clone_mib=clone,
        kv_saving=0.0 if clone <= 0 else 1.0 - (cow / clone),
        gpu_mem_mib=gpu_mem_mib(),
        decode_tokens=decode_n * len(trunks),
        decode_per_item=decode_n,
        decode_ids=decoded,
        sessions=len(trunks),
        branching=len(thoughts),
        notes=f"{len(trunks)} items; one fan-out generate + one winner decode",
    )


def run_react_forest_vllm(
    llm: Any,
    SamplingParams: Any,
    TokensPrompt: Any,
    system: str,
    cfg_bpt: float,
    items: Sequence[tuple[str, str, str, str]],
    *,
    decode_n: int,
    idle_ms: float,
    prefix_cache: bool,
    workload: str,
) -> RunMetrics:
    packed = []
    residuals: list[int] = []
    for trunk, wrap, recov, obs in items:
        trunk_ids = _tok_ids(llm, trunk)
        wrap_ids = _tok_ids(llm, wrap)
        recov_ids = _tok_ids(llm, recov)
        obs_ids = _tok_ids(llm, obs)
        packed.append((trunk_ids, wrap_ids, recov_ids, obs_ids))
        residuals.extend([len(wrap_ids), len(recov_ids), len(obs_ids)])
    t0 = _now()
    trunks = [t for t, *_ in packed]
    # Same ToT shape: optional trunk populate, then fan-out BOTH wrappers.
    if prefix_cache:
        _gen(llm, SamplingParams, TokensPrompt, trunks, 1)
    t_idle = _now()
    branches = [seq for t, w, r, _o in packed for seq in (t + w, t + r)]
    _gen(llm, SamplingParams, TokensPrompt, branches, 1)
    spec_ms = (_now() - t_idle) * 1000.0
    remain = idle_ms - spec_ms
    slept = 0.0
    if remain > 0:
        time.sleep(remain / 1000.0)
        slept = remain
    t_obs = _now()
    fulls = [t + w + o for t, w, _r, o in packed]
    decoded = _gen(llm, SamplingParams, TokensPrompt, fulls, decode_n)
    ttft = (_now() - t_obs) * 1000.0
    t1 = _now()
    trunk_n = len(packed[0][0]) if packed else 0
    from forkserve.pages import clone_memory_bytes, cow_memory_bytes

    cow = _mib(cow_memory_bytes(trunk_n * len(packed), residuals, cfg_bpt))
    clone = _mib(clone_memory_bytes(trunk_n * len(packed), residuals, cfg_bpt, 2 * len(packed)))
    wrap_recov = [len(w) + len(r) for _t, w, r, _o in packed]
    peak = (
        sum(len(t) for t, *_ in packed) + sum(wrap_recov)
        if prefix_cache
        else sum(len(b) for b in branches)
    )
    return RunMetrics(
        system=system,
        workload=workload,
        tp=0,
        e2e_ms=(t1 - t0) * 1000.0 - slept,
        fanout_ms=spec_ms,
        decode_ms=ttft,
        ttft_from_obs_ms=ttft,
        spec_ms=spec_ms,
        idle_ms=idle_ms,
        trunk_tokens=trunk_n,
        residual_tokens=residuals,
        peak_kv_tokens=int(peak),
        m_cow_mib=cow,
        m_clone_mib=clone,
        kv_saving=0.0 if clone <= 0 else 1.0 - (cow / clone),
        gpu_mem_mib=gpu_mem_mib(),
        decode_tokens=decode_n * len(packed),
        decode_per_item=decode_n,
        decode_ids=decoded,
        sessions=len(packed),
        branching=2,
        notes=f"{len(packed)} items; idle fan-out wrap+recovery; then wrap+obs decode",
    )


def _apply_depth(trunks: list[str], thoughts: Sequence[str]) -> list[str]:
    """Prefix the committed thought so a second fan-out sits on a longer spine."""
    depth = int(os.environ.get("FORKSERVE_DEPTH", "1") or 1)
    if depth <= 1 or not thoughts:
        return trunks
    extra = thoughts[0]
    return [t + extra for t in trunks]


def run_gsm8k_forkserve(eng: Any, cfg: Any, args: argparse.Namespace) -> RunMetrics:
    from forkserve.bench_tasks import gsm8k_thoughts, gsm8k_trunk, load_gsm8k

    problems = load_gsm8k(args.limit)
    if getattr(args, "quality_only", False):
        return run_quality_forkserve(
            eng, args, "gsm8k", problems,
            [gsm8k_trunk(p) for p in problems],
            [p.answer for p in problems],
            decode_n=workload_decode(args, "gsm8k"),
        )
    thoughts = gsm8k_thoughts(args.branching)
    parts: list[RunMetrics] = []
    for start, batch in _iter_chunks(problems, args):
        trunks = _apply_depth([gsm8k_trunk(item) for item in batch], thoughts)
        row = run_tot_forest_forkserve(
            eng, cfg, trunks, thoughts,
            decode_n=workload_decode(args, "gsm8k"), idle_ms=0.0, workload="gsm8k",
        )
        _set_golds(row, [p.answer for p in batch], _fs_texts(eng, row.decode_ids))
        row.item_ids = [p.item_id for p in batch]
        parts.append(row)
        _log_forest_chunk(args, "forkserve", "gsm8k", start, batch, len(problems), row, parts)
        _close_all(eng)
    return merge_metrics(parts)


def run_game24_forkserve(eng: Any, cfg: Any, args: argparse.Namespace) -> RunMetrics:
    from forkserve.bench_tasks import game24_thoughts, game24_trunk, load_game24

    problems = load_game24(args.limit)
    if getattr(args, "quality_only", False):
        return run_quality_forkserve(
            eng, args, "game24", problems,
            [game24_trunk(p) for p in problems],
            [p.question for p in problems],
        )
    thoughts = game24_thoughts(args.branching)
    parts: list[RunMetrics] = []
    os.environ["FORKSERVE_MIN_TOKENS"] = "24"
    try:
        for start, batch in _iter_chunks(problems, args):
            trunks = _apply_depth([game24_trunk(item) for item in batch], thoughts)
            eng.config.answer_stop_hints = tuple(p.question for p in batch)
            row = run_tot_forest_forkserve(
                eng, cfg, trunks, thoughts,
                decode_n=args.decode, idle_ms=0.0, workload="game24",
            )
            eng.config.answer_stop_hints = ()
            _set_golds(row, [p.question for p in batch], _fs_texts(eng, row.decode_ids))
            row.item_ids = [p.item_id for p in batch]
            parts.append(row)
            _log_forest_chunk(args, "forkserve", "game24", start, batch, len(problems), row, parts)
            _close_all(eng)
    finally:
        os.environ.pop("FORKSERVE_MIN_TOKENS", None)
    return merge_metrics(parts)


def run_humaneval_forkserve(eng: Any, cfg: Any, args: argparse.Namespace) -> RunMetrics:
    from forkserve.bench_tasks import humaneval_react_strings, humaneval_trunk, load_humaneval

    problems = load_humaneval(args.limit)
    if getattr(args, "quality_only", False):
        texts: list[str] = []
        e2e = 0.0
        n_tok = 0
        golds = [p.tests for p in problems]
        prompts = [p.prompt for p in problems]
        t_job = _now()
        progress(args, f"start {_sys_tp(args, 'forkserve')} humaneval n={len(problems)} chunk={_chunk_size(args)} decode={args.decode} quality-only")
        for start, batch in _iter_chunks(problems, args):
            t0 = _now()
            chunk_texts, ids = _humaneval_complete_forkserve(eng, batch, args.decode)
            dt = _now() - t0
            e2e += dt * 1000.0
            texts.extend(chunk_texts)
            n_tok += sum(len(x) for x in ids)
            done = len(texts)
            progress(
                args,
                f"{_sys_tp(args, 'forkserve')} humaneval {done}/{len(problems)} "
                f"+{len(batch)} {_score_so_far('humaneval', texts, golds, prompts)} "
                f"chunk={dt:.1f}s {_eta_s(done, len(problems), _now() - t_job)}",
            )
            _close_all(eng)
        row = RunMetrics(
            system="forkserve",
            workload="humaneval",
            tp=0,
            e2e_ms=e2e,
            decode_ms=e2e,
            decode_tokens=n_tok,
            decode_per_item=args.decode,
            sessions=len(problems),
            branching=1,
            notes=f"{len(problems)} items; quality-only official HumanEval completion",
            item_ids=[p.item_id for p in problems],
        )
        return _set_golds(row, [p.tests for p in problems], texts, [p.prompt for p in problems])
    parts: list[RunMetrics] = []
    all_texts: list[str] = []
    for start, batch in _iter_chunks(problems, args):
        items = []
        for item in batch:
            wrap, recov, obs = humaneval_react_strings(item)
            items.append((humaneval_trunk(item), wrap, recov, obs))
        row = run_react_forest_forkserve(
            eng, cfg, items,
            decode_n=args.decode, idle_ms=args.idle_ms, workload="humaneval",
        )
        texts, _ids = _humaneval_complete_forkserve(eng, batch, args.decode)
        all_texts.extend(texts)
        row.item_ids = [p.item_id for p in batch]
        _set_golds(row, [p.tests for p in batch], texts, [p.prompt for p in batch])
        parts.append(row)
        _log_forest_chunk(args, "forkserve", "humaneval", start, batch, len(problems), row, parts)
        _close_all(eng)
    merged = merge_metrics(parts)
    merged.decode_texts = all_texts
    merged.golds = [p.tests for p in problems]
    merged.gold_prompts = [p.prompt for p in problems]
    merged.item_ids = [p.item_id for p in problems]
    merged.notes = (merged.notes + "; quality=official HumanEval completion").strip("; ")
    return merged


def run_gsm8k_vllm(llm: Any, SamplingParams: Any, TokensPrompt: Any, system: str, cfg_bpt: float, args: argparse.Namespace) -> RunMetrics:
    from forkserve.bench_tasks import gsm8k_thoughts, gsm8k_trunk, load_gsm8k

    problems = load_gsm8k(args.limit)
    if getattr(args, "quality_only", False):
        return run_quality_vllm(
            llm, SamplingParams, TokensPrompt, system, args, "gsm8k", problems,
            [gsm8k_trunk(p) for p in problems],
            [p.answer for p in problems],
            decode_n=workload_decode(args, "gsm8k"),
        )
    thoughts = gsm8k_thoughts(args.branching)
    parts: list[RunMetrics] = []
    for start, batch in _iter_chunks(problems, args):
        trunks = _apply_depth([gsm8k_trunk(item) for item in batch], thoughts)
        row = run_tot_forest_vllm(
            llm, SamplingParams, TokensPrompt, system, cfg_bpt,
            trunks, thoughts,
            decode_n=workload_decode(args, "gsm8k"), prefix_cache=(system == "vllm_apc"), workload="gsm8k",
        )
        _set_golds(row, [p.answer for p in batch], _vllm_texts(llm, row.decode_ids))
        row.item_ids = [p.item_id for p in batch]
        parts.append(row)
        _log_forest_chunk(args, system, "gsm8k", start, batch, len(problems), row, parts)
    return merge_metrics(parts)


def _math_workload_spec(name: str):
    from forkserve import bench_tasks as t

    table = {
        "svamp": (t.load_svamp, t.numeric_math_trunk, t.gsm8k_thoughts),
        "gsmhard": (t.load_gsmhard, t.numeric_math_trunk, t.gsm8k_thoughts),
        "math500": (t.load_math500, t.contest_math_trunk, t.contest_thoughts),
        "aime": (t.load_aime, t.contest_math_trunk, t.contest_thoughts),
        "amc23": (t.load_amc23, t.contest_math_trunk, t.contest_thoughts),
    }
    if name not in table:
        raise KeyError(name)
    return table[name]


def run_named_math_forkserve(eng: Any, cfg: Any, args: argparse.Namespace, workload: str) -> RunMetrics:
    load, trunk_fn, thoughts_fn = _math_workload_spec(workload)
    problems = load(args.limit)
    n_dec = workload_decode(args, workload)
    if getattr(args, "quality_only", False):
        return run_quality_forkserve(
            eng, args, workload, problems,
            [trunk_fn(p) for p in problems],
            [p.answer for p in problems],
            decode_n=n_dec,
        )
    thoughts = thoughts_fn(args.branching)
    parts: list[RunMetrics] = []
    for start, batch in _iter_chunks(problems, args):
        trunks = [trunk_fn(item) for item in batch]
        row = run_tot_forest_forkserve(
            eng, cfg, trunks, thoughts,
            decode_n=n_dec, idle_ms=0.0, workload=workload,
        )
        _set_golds(row, [p.answer for p in batch], _fs_texts(eng, row.decode_ids))
        row.item_ids = [p.item_id for p in batch]
        parts.append(row)
        _log_forest_chunk(args, "forkserve", workload, start, batch, len(problems), row, parts)
        _close_all(eng)
    return merge_metrics(parts)


def run_named_math_vllm(
    llm: Any, SamplingParams: Any, TokensPrompt: Any, system: str, cfg_bpt: float,
    args: argparse.Namespace, workload: str,
) -> RunMetrics:
    load, trunk_fn, thoughts_fn = _math_workload_spec(workload)
    problems = load(args.limit)
    n_dec = workload_decode(args, workload)
    if getattr(args, "quality_only", False):
        return run_quality_vllm(
            llm, SamplingParams, TokensPrompt, system, args, workload, problems,
            [trunk_fn(p) for p in problems],
            [p.answer for p in problems],
            decode_n=n_dec,
        )
    thoughts = thoughts_fn(args.branching)
    parts: list[RunMetrics] = []
    for start, batch in _iter_chunks(problems, args):
        trunks = [trunk_fn(item) for item in batch]
        row = run_tot_forest_vllm(
            llm, SamplingParams, TokensPrompt, system, cfg_bpt,
            trunks, thoughts,
            decode_n=n_dec, prefix_cache=(system == "vllm_apc"), workload=workload,
        )
        _set_golds(row, [p.answer for p in batch], _vllm_texts(llm, row.decode_ids))
        row.item_ids = [p.item_id for p in batch]
        parts.append(row)
        _log_forest_chunk(args, system, workload, start, batch, len(problems), row, parts)
    return merge_metrics(parts)


def run_game24_vllm(llm: Any, SamplingParams: Any, TokensPrompt: Any, system: str, cfg_bpt: float, args: argparse.Namespace) -> RunMetrics:
    from forkserve.bench_tasks import game24_thoughts, game24_trunk, load_game24

    problems = load_game24(args.limit)
    if getattr(args, "quality_only", False):
        return run_quality_vllm(
            llm, SamplingParams, TokensPrompt, system, args, "game24", problems,
            [game24_trunk(p) for p in problems],
            [p.question for p in problems],
        )
    thoughts = game24_thoughts(args.branching)
    parts: list[RunMetrics] = []
    os.environ["FORKSERVE_MIN_TOKENS"] = "24"
    try:
        for start, batch in _iter_chunks(problems, args):
            trunks = _apply_depth([game24_trunk(item) for item in batch], thoughts)
            row = run_tot_forest_vllm(
                llm, SamplingParams, TokensPrompt, system, cfg_bpt,
                trunks, thoughts,
                decode_n=args.decode, prefix_cache=(system == "vllm_apc"), workload="game24",
            )
            _set_golds(row, [p.question for p in batch], _vllm_texts(llm, row.decode_ids))
            row.item_ids = [p.item_id for p in batch]
            parts.append(row)
            _log_forest_chunk(args, system, "game24", start, batch, len(problems), row, parts)
    finally:
        os.environ.pop("FORKSERVE_MIN_TOKENS", None)
    return merge_metrics(parts)


def run_humaneval_vllm(llm: Any, SamplingParams: Any, TokensPrompt: Any, system: str, cfg_bpt: float, args: argparse.Namespace) -> RunMetrics:
    from forkserve.bench_tasks import humaneval_react_strings, humaneval_trunk, load_humaneval

    problems = load_humaneval(args.limit)
    if getattr(args, "quality_only", False):
        texts: list[str] = []
        e2e = 0.0
        n_tok = 0
        golds = [p.tests for p in problems]
        prompts = [p.prompt for p in problems]
        t_job = _now()
        progress(args, f"start {_sys_tp(args, system)} humaneval n={len(problems)} chunk={_chunk_size(args)} decode={args.decode} quality-only")
        for start, batch in _iter_chunks(problems, args):
            t0 = _now()
            chunk_texts, ids = _humaneval_complete_vllm(
                llm, SamplingParams, TokensPrompt, batch, args.decode
            )
            dt = _now() - t0
            e2e += dt * 1000.0
            texts.extend(chunk_texts)
            n_tok += sum(len(x) for x in ids)
            done = len(texts)
            progress(
                args,
                f"{_sys_tp(args, system)} humaneval {done}/{len(problems)} "
                f"+{len(batch)} {_score_so_far('humaneval', texts, golds, prompts)} "
                f"chunk={dt:.1f}s {_eta_s(done, len(problems), _now() - t_job)}",
            )
        row = RunMetrics(
            system=system,
            workload="humaneval",
            tp=0,
            e2e_ms=e2e,
            decode_ms=e2e,
            decode_tokens=n_tok,
            decode_per_item=args.decode,
            sessions=len(problems),
            branching=1,
            notes=f"{len(problems)} items; quality-only official HumanEval completion",
            item_ids=[p.item_id for p in problems],
        )
        return _set_golds(row, [p.tests for p in problems], texts, [p.prompt for p in problems])
    parts: list[RunMetrics] = []
    all_texts: list[str] = []
    for start, batch in _iter_chunks(problems, args):
        packed = []
        for item in batch:
            wrap, recov, obs = humaneval_react_strings(item)
            packed.append((humaneval_trunk(item), wrap, recov, obs))
        row = run_react_forest_vllm(
            llm, SamplingParams, TokensPrompt, system, cfg_bpt, packed,
            decode_n=args.decode, idle_ms=args.idle_ms,
            prefix_cache=(system == "vllm_apc"), workload="humaneval",
        )
        texts, _ids = _humaneval_complete_vllm(
            llm, SamplingParams, TokensPrompt, batch, args.decode
        )
        all_texts.extend(texts)
        row.item_ids = [p.item_id for p in batch]
        _set_golds(row, [p.tests for p in batch], texts, [p.prompt for p in batch])
        parts.append(row)
        _log_forest_chunk(args, system, "humaneval", start, batch, len(problems), row, parts)
    merged = merge_metrics(parts)
    merged.decode_texts = all_texts
    merged.golds = [p.tests for p in problems]
    merged.gold_prompts = [p.prompt for p in problems]
    merged.item_ids = [p.item_id for p in problems]
    merged.notes = (merged.notes + "; quality=official HumanEval completion").strip("; ")
    return merged


# ----- mock accounting (no GPU) ---------------------------------------------

def run_mock(args: argparse.Namespace) -> list[RunMetrics]:
    bpt = 147_456.0
    eng, cfg = _mock_engine(bpt)
    tok = eng.backend.tokenize
    trunk = make_trunk(args.trunk_tokens, tok)
    thoughts = thought_residuals(args.branching)
    trunks = [
        make_trunk(args.trunk_tokens, tok) + f" session {i}."
        for i in range(args.sessions)
    ]
    rows: list[RunMetrics] = []
    if "gsm8k" in args.workloads:
        rows.append(run_gsm8k_forkserve(eng, cfg, args))
        eng, cfg = _mock_engine(bpt)
    for name in ("svamp", "gsmhard", "math500", "aime", "amc23"):
        if name in args.workloads:
            rows.append(run_named_math_forkserve(eng, cfg, args, name))
            eng, cfg = _mock_engine(bpt)
    if "game24" in args.workloads:
        rows.append(run_game24_forkserve(eng, cfg, args))
        eng, cfg = _mock_engine(bpt)
    if "humaneval" in args.workloads:
        he = argparse.Namespace(**{**vars(args), "idle_ms": 0.0})
        rows.append(run_humaneval_forkserve(eng, cfg, he))
        eng, cfg = _mock_engine(bpt)
    if "tot" in args.workloads:
        rows.append(run_tot_forkserve(eng, cfg, trunk, thoughts, decode_n=args.decode, idle_ms=args.idle_ms))
        # Rebuild engine so sessions do not accumulate.
        eng, cfg = _mock_engine(bpt)
    if "react" in args.workloads:
        rows.append(run_react_forkserve(eng, cfg, trunk, decode_n=args.decode, idle_ms=0.0))
        eng, cfg = _mock_engine(bpt)
    if "multi" in args.workloads:
        rows.append(run_multi_forkserve(eng, cfg, trunks, thoughts, decode_n=args.decode, idle_ms=args.idle_ms))
    for r in rows:
        r.tp = args.tp[0] if args.tp else 1
        r.system = "forkserve"
    return rows


# ----- worker / orchestrator -------------------------------------------------

def _close_all(eng: Any) -> None:
    for sid in list(eng.forest.sessions):
        eng.close(sid)


def worker_main(args: argparse.Namespace) -> int:
    os.environ.update(_sanitize_cuda_env())
    system = args.system
    if not args.tp:
        args.tp = [1]
    tp = int(args.tp[0])
    model = args.model
    bpt = bytes_per_token_from_config(model)
    rows: list[RunMetrics] = []

    if system in ("forkserve", "forkserve_plus"):
        eng, backend, cfg = _engine(model, tp, args)
        tokenize = backend.tokenize
        h = eng.open("warmup ping")
        eng.close(h.id)
    else:
        llm, SamplingParams, TokensPrompt = _vllm_llm(
            model, tp, args, prefix_cache=(system == "vllm_apc")
        )
        _warmup_vllm(llm, SamplingParams, TokensPrompt)
        tokenize = lambda text: _tok_ids(llm, text)  # noqa: E731
        eng = None
        cfg = None
        backend = None

    trunk = make_trunk(args.trunk_tokens, tokenize)
    thoughts = thought_residuals(args.branching)
    trunks = [
        make_trunk(max(args.trunk_tokens // 2, 128), tokenize)
        + f" session-{i} unique salt."
        for i in range(args.sessions)
    ]
    prefix = system == "vllm_apc"

    if system in ("forkserve", "forkserve_plus"):
        assert eng is not None and cfg is not None
        if "gsm8k" in args.workloads:
            rows.append(run_gsm8k_forkserve(eng, cfg, args))
            _close_all(eng)
        for name in ("svamp", "gsmhard", "math500", "aime", "amc23"):
            if name in args.workloads:
                rows.append(run_named_math_forkserve(eng, cfg, args, name))
                _close_all(eng)
        if "game24" in args.workloads:
            rows.append(run_game24_forkserve(eng, cfg, args))
            _close_all(eng)
        if "humaneval" in args.workloads:
            rows.append(run_humaneval_forkserve(eng, cfg, args))
            _close_all(eng)
        if "tot" in args.workloads:
            rows.append(run_tot_forkserve(eng, cfg, trunk, thoughts, decode_n=args.decode, idle_ms=args.idle_ms))
            _close_all(eng)
        if "react" in args.workloads:
            rows.append(run_react_forkserve(eng, cfg, trunk, decode_n=args.decode, idle_ms=args.idle_ms))
            _close_all(eng)
        if "multi" in args.workloads:
            rows.append(run_multi_forkserve(eng, cfg, trunks, thoughts, decode_n=args.decode, idle_ms=args.idle_ms))
            _close_all(eng)
    else:
        if "gsm8k" in args.workloads:
            rows.append(run_gsm8k_vllm(llm, SamplingParams, TokensPrompt, system, bpt, args))
        for name in ("svamp", "gsmhard", "math500", "aime", "amc23"):
            if name in args.workloads:
                rows.append(run_named_math_vllm(llm, SamplingParams, TokensPrompt, system, bpt, args, name))
        if "game24" in args.workloads:
            rows.append(run_game24_vllm(llm, SamplingParams, TokensPrompt, system, bpt, args))
        if "humaneval" in args.workloads:
            rows.append(run_humaneval_vllm(llm, SamplingParams, TokensPrompt, system, bpt, args))
        if "tot" in args.workloads:
            rows.append(
                run_tot_vllm(
                    llm, SamplingParams, TokensPrompt, system, bpt, trunk, thoughts,
                    decode_n=args.decode, prefix_cache=prefix,
                )
            )
        if "react" in args.workloads:
            rows.append(
                run_react_vllm(
                    llm, SamplingParams, TokensPrompt, system, bpt, trunk,
                    decode_n=args.decode, idle_ms=args.idle_ms, prefix_cache=prefix,
                )
            )
        if "multi" in args.workloads:
            rows.append(
                run_multi_vllm(
                    llm, SamplingParams, TokensPrompt, system, bpt, trunks, thoughts,
                    decode_n=args.decode, prefix_cache=prefix,
                )
            )

    for r in rows:
        r.tp = tp
    if system in ("forkserve", "forkserve_plus") and backend is not None:
        closer = getattr(backend, "shutdown", None)
        if callable(closer):
            closer()
    elif system not in ("forkserve", "forkserve_plus"):
        try:
            del llm
        except Exception:
            pass
        import gc

        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
    payload = {
        "system": system,
        "tp": tp,
        "model": model,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "rows": [r.to_dict() for r in rows],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    progress(args, f"worker done {system} tp={tp} rows={len(rows)} wrote {args.out}")
    return 0


def visible_gpu_count() -> int:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if vis:
        return len([x for x in vis.split(",") if x.strip() != ""])
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-L"], text=True, stderr=subprocess.DEVNULL
        )
        return sum(1 for line in out.splitlines() if line.startswith("GPU "))
    except (OSError, subprocess.CalledProcessError):
        return 0


def default_tps(n_gpu: int, requested: Sequence[int] | None) -> list[int]:
    if requested:
        tps = [t for t in requested if t <= max(n_gpu, 1)]
        return tps or [min(requested[0], max(n_gpu, 1))]
    if n_gpu >= 4:
        return [2, 4]
    if n_gpu >= 2:
        return [2]
    return [1]


def format_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "system           tp  workload    e2e_ms  fanout_ms  ttft_obs_ms  peak_kv  task_score  metric        kv_save  vs_recompute",
        "-" * 128,
    ]
    def _metric(row: dict[str, Any]) -> float:
        if row["workload"] in ("react", "humaneval") and float(row.get("ttft_from_obs_ms") or 0) > 0:
            return float(row["ttft_from_obs_ms"])
        return float(row["e2e_ms"])

    # speedup vs vllm_recompute on same (tp, workload)
    base: dict[tuple[int, str], float] = {}
    for r in rows:
        if r["system"] == "vllm_recompute":
            key = (int(r["tp"]), r["workload"])
            base[key] = _metric(r)

    for r in rows:
        key = (int(r["tp"]), r["workload"])
        metric = _metric(r)
        b = base.get(key, 0.0)
        speed = (b / metric) if metric > 0 and b > 0 else 0.0
        score = r.get("task_score")
        metric = str(r.get("task_metric") or "")
        sc_s = f"{float(score):10.3f}" if score is not None and float(score) >= 0 else "       n/a"
        lines.append(
            f"{r['system']:<16} {r['tp']:>2}  {r['workload']:<10} "
            f"{r['e2e_ms']:7.1f}  {r['fanout_ms']:9.1f}  {r['ttft_from_obs_ms']:11.1f}  "
            f"{r['peak_kv_tokens']:7d}  {sc_s}  {metric:<12}  "
            f"{r['kv_saving']:7.2f}  {speed:5.2f}x"
        )
    return "\n".join(lines)


def _sanitize_cuda_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """Drop CUDA 13 wheel paths; this box's driver is 12.9 / torch is cu128."""
    e = dict(os.environ if env is None else env)
    kept = [
        p
        for p in e.get("LD_LIBRARY_PATH", "").split(":")
        if p and "/cu13/" not in p and "/nvidia/cu13/" not in p
    ]
    e["LD_LIBRARY_PATH"] = ":".join(kept)
    e.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    e.setdefault("PYTHONUNBUFFERED", "1")
    return e


def _gpu_used_mib() -> dict[int, int]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return {}
    used: dict[int, int] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            used[int(parts[0])] = int(float(parts[1]))
    return used


def _wait_devices_free(dev_csv: str, timeout_s: float = 90.0, max_used_mib: int = 2048) -> bool:
    """Wait until listed physical GPUs have dropped leftover / occupy allocations."""
    want = [int(x) for x in dev_csv.split(",") if x.strip().isdigit()]
    if not want:
        return True
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        used = _gpu_used_mib()
        if used and all(used.get(i, 0) <= max_used_mib for i in want):
            return True
        time.sleep(0.25)
    used = _gpu_used_mib()
    print(f"GPUs still busy after {timeout_s:.0f}s: {used}", flush=True)
    return False


def _bench_lock_path() -> Path:
    root = Path(os.environ.get("FORKSERVE_ROOT", Path(__file__).resolve().parents[1]))
    return Path(
        os.environ.get("FORKSERVE_BENCH_LOCK", str(root / "logs" / ".forkserve_bench.lock"))
    )


def orchestrate(args: argparse.Namespace) -> int:
    lock = _bench_lock_path()
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(str(os.getpid()))
    try:
        return _orchestrate(args)
    finally:
        lock.unlink(missing_ok=True)


def _orchestrate(args: argparse.Namespace) -> int:
    n_gpu = visible_gpu_count()
    tps = default_tps(n_gpu, args.tp)
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    devices = [x.strip() for x in vis.split(",") if x.strip()] if vis else [str(i) for i in range(n_gpu)]
    systems = list(args.systems)
    out_dir = Path(args.out).parent if args.out else Path("logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    progress_log = str(getattr(args, "progress_log", "") or "") or os.environ.get(
        "FORKSERVE_PROGRESS_LOG", ""
    )
    if not progress_log:
        progress_log = str(Path(args.out).with_name("progress.log"))
    args.progress_log = progress_log
    os.environ["FORKSERVE_PROGRESS_LOG"] = progress_log
    os.environ["FORKSERVE_MODEL"] = str(args.model or "")
    model_l = str(args.model or "").lower()
    if "deepseek" in model_l or "r1-distill" in model_l:
        os.environ.setdefault("FORKSERVE_CHAT_STYLE", "deepseek_r1")
    elif "mistral" in model_l:
        os.environ.setdefault("FORKSERVE_CHAT_STYLE", "mistral")
    elif "llama" in model_l:
        os.environ.setdefault("FORKSERVE_CHAT_STYLE", "llama")
    progress(
        args,
        f"orchestrating systems={systems} tp={tps} gpus={devices} "
        f"workloads={args.workloads} limit={args.limit} chunk={getattr(args, 'chunk', 4)} "
        f"quality_only={bool(getattr(args, 'quality_only', False))} model={args.model} "
        f"progress={progress_log}",
    )
    for tp in tps:
        if tp > len(devices):
            print(f"skip tp={tp}: only {len(devices)} visible GPUs", flush=True)
            continue
        dev = ",".join(devices[:tp])
        for system in systems:
            shard = out_dir / f"bench_{system}_tp{tp}.json"
            cmd = [
                sys.executable,
                "-m",
                "forkserve.bench",
                "--worker",
                "--system",
                system,
                "--tp",
                str(tp),
                "--model",
                args.model,
                "--decode",
                str(args.decode),
                "--gsm8k-decode",
                str(workload_decode(args, "gsm8k")),
                "--idle-ms",
                str(args.idle_ms),
                "--branching",
                str(args.branching),
                "--sessions",
                str(args.sessions),
                "--trunk-tokens",
                str(args.trunk_tokens),
                "--max-model-len",
                str(args.max_model_len),
                "--max-batched-tokens",
                str(args.max_batched_tokens),
                "--gpu-util",
                str(args.gpu_util),
                "--workloads",
                ",".join(args.workloads),
                "--limit",
                str(args.limit),
                "--chunk",
                str(getattr(args, "chunk", 4)),
                "--out",
                str(shard),
            ]
            if getattr(args, "quality_only", False):
                cmd.append("--quality-only")
            if args.enforce_eager:
                cmd.append("--enforce-eager")
            env = _sanitize_cuda_env()
            env["CUDA_VISIBLE_DEVICES"] = dev
            env["PYTHONUNBUFFERED"] = "1"
            env["FORKSERVE_PROGRESS_LOG"] = progress_log
            env["FORKSERVE_MODEL"] = str(args.model or "")
            if os.environ.get("FORKSERVE_CHAT_STYLE"):
                env["FORKSERVE_CHAT_STYLE"] = os.environ["FORKSERVE_CHAT_STYLE"]
            progress(args, f"==> {system} tp={tp} devices={dev}")
            if not _wait_devices_free(dev, timeout_s=180.0, max_used_mib=4096):
                progress(args, f"worker skipped: {system} tp={tp} — GPUs {dev} still occupied")
                return 1
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                print(line, flush=True)
                # Worker already appended timestamped ``progress()`` lines.
                if line and not line.startswith("["):
                    progress(args, f"[{system} tp={tp}] {line}")
            rc = proc.wait()
            time.sleep(2.0)
            if rc != 0:
                progress(args, f"worker failed: {system} tp={tp} rc={rc}")
                return rc
            data = json.loads(shard.read_text())
            all_rows.extend(data["rows"])

    from forkserve.quality import annotate_quality

    annotate_quality(all_rows)
    report = {
        "model": args.model,
        "systems": systems,
        "tp": tps,
        "gpus": devices,
        "rows": all_rows,
        "table": format_table(all_rows),
    }
    dest = Path(args.out) if args.out else out_dir / "bench_gpu.json"
    dest.write_text(json.dumps({**report, "table": report["table"]}, indent=2))
    print("\n" + report["table"] + "\n", flush=True)
    progress(args, f"wrote {dest}")
    return 0


def _csv_list(raw: str, allowed: Sequence[str] | None = None) -> list[str]:
    xs = [x.strip() for x in raw.split(",") if x.strip()]
    if allowed:
        bad = [x for x in xs if x not in allowed]
        if bad:
            raise argparse.ArgumentTypeError(f"unknown {bad}; allowed {list(allowed)}")
    return xs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ForkServe vs vLLM multi-GPU comparison")
    p.add_argument("--worker", action="store_true", help="run one (system, tp) in this process")
    p.add_argument("--system", choices=SYSTEMS, default="forkserve")
    p.add_argument("--systems", type=lambda s: _csv_list(s, SYSTEMS), default=list(SYSTEMS))
    p.add_argument("--tp", type=lambda s: [int(x) for x in s.split(",") if x.strip()], default=None)
    p.add_argument("--backend", choices=("vllm", "mock"), default="vllm")
    p.add_argument("--model", default=os.environ.get("FORKSERVE_MODEL", "/home/dliu/models/Qwen3-8B"))
    p.add_argument(
        "--decode",
        type=int,
        default=int(os.environ.get("FORKSERVE_DECODE", "256")),
        help="tokens generated per winner/item; JSON decode_tokens is the sum across items",
    )
    p.add_argument(
        "--gsm8k-decode",
        type=int,
        default=None,
        help="GSM8K tokens per item (default max(--decode, 512) unless --decode < 64)",
    )
    p.add_argument("--idle-ms", type=float, default=2000.0)
    p.add_argument("--branching", type=int, default=4)
    p.add_argument("--sessions", type=int, default=2)
    p.add_argument("--trunk-tokens", type=int, default=512)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-batched-tokens", type=int, default=2048)
    p.add_argument("--gpu-util", type=float, default=float(os.environ.get("FORKSERVE_GPU_UTIL", "0.90")))
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument(
        "--limit",
        type=int,
        default=4,
        help="problems per math/coding slice; 0 = entire jsonl/csv",
    )
    p.add_argument(
        "--chunk",
        type=int,
        default=4,
        help="items per forest / generate batch (required when --limit 0)",
    )
    p.add_argument(
        "--quality-only",
        action="store_true",
        help="single-path generate only — skips ToT/ReAct fan-out; peak_kv/fanout stay 0",
    )
    p.add_argument("--workloads", type=lambda s: _csv_list(s, WORKLOADS), default=["gsm8k", "game24", "humaneval"])
    p.add_argument("--out", default="logs/bench_gpu.json")
    p.add_argument(
        "--progress-log",
        default=os.environ.get("FORKSERVE_PROGRESS_LOG", ""),
        help="append timestamped chunk progress here (also FORKSERVE_PROGRESS_LOG)",
    )
    args = p.parse_args(argv)
    if args.tp is None:
        args.tp = []
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.backend == "mock":
        rows = run_mock(args)
        payload = {"backend": "mock", "rows": [r.to_dict() for r in rows]}
        from forkserve.quality import annotate_quality

        annotate_quality(payload["rows"])
        print(format_table(payload["rows"]))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2))
        return 0
    if args.worker:
        return worker_main(args)
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
