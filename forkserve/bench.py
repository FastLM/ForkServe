"""System comparison: vLLM recompute vs APC vs ForkServe on 2/4 GPUs.

Three agent workloads, same prompts:

* ``tot`` — Tree-of-Thoughts fan-out: one long trunk, B thought prefixes.
* ``react`` — tool-idle TTFT: known wrapper is speculatively prefills during
  idle; observation residual is the only critical-path prefill.
* ``multi`` — S concurrent ToT sessions (engine batching / TP scaling).

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


SYSTEMS = ("vllm_recompute", "vllm_apc", "forkserve")
WORKLOADS = ("tot", "react", "multi")


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
    sessions: int = 1
    branching: int = 1
    notes: str = ""

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

    bpt = bytes_per_token_from_config(model)
    cfg = ForkServeConfig(
        max_batched_tokens=args.max_batched_tokens,
        hbm_capacity_bytes=40.0 * (1 << 30) * max(tp, 1),
        bytes_per_token=bpt,
        num_workers=1,
    )
    backend = VllmBackend(
        config=cfg,
        model=model,
        tensor_parallel=tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_util,
        enable_prefix_caching=True,
        enforce_eager=args.enforce_eager,
        two_class=True,
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


def _select_winner(eng: Any, session: Any, kids: Sequence[Any], winner: int = 0) -> Any:
    from forkserve.types import JoinPolicy

    keep = kids[winner]
    for i, cid in enumerate(kids):
        if i != winner:
            eng.abort(session, cid)
    return eng.join(session, [keep], JoinPolicy.WINNER).node_id


def run_tot_forkserve(
    eng: Any,
    cfg: Any,
    trunk: str,
    thoughts: Sequence[str],
    *,
    decode_n: int,
    idle_ms: float,
) -> RunMetrics:
    t0 = _now()
    h = eng.open(trunk)
    tree = eng.tree(h.id)
    trunk_n = len(tree.get(h.tip).tokens)
    t_open = _now()
    kids = _fanout_thoughts(eng, h.id, h.tip, thoughts, idle_ms)
    residuals = [len(tree.get(k).residual) for k in kids]
    peak = tree.live_kv_tokens()
    t_fan = _now()
    _select_winner(eng, h.id, kids, winner=0)
    out = eng.generate(h.id, decode_n)
    t1 = _now()
    cow, clone, saving = _peak_memory(cfg, trunk_n, residuals, len(thoughts))
    m = eng.metrics[h.id]
    return RunMetrics(
        system="forkserve",
        workload="tot",
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
        branching=len(thoughts),
        notes="open + CoW fan-out of identical residuals + winner decode",
    )


def run_react_forkserve(
    eng: Any,
    cfg: Any,
    trunk: str,
    *,
    decode_n: int,
    idle_ms: float,
) -> RunMetrics:
    from forkserve.adapters.react import ReActAdapter
    from forkserve.adapters.templates import ToolWrappers

    wrap, recov, obs = react_strings()
    t0 = _now()
    h = eng.open(trunk)
    tree = eng.tree(h.id)
    parent = h.tip
    trunk_n = len(tree.get(parent).tokens)
    ad = ReActAdapter(eng, ToolWrappers())
    t_idle = _now()
    _happy, _fail = ad.on_tool_parsed(
        h.id, parent, "bash", t_idle_ms=idle_ms, include_recovery=True
    )
    eng.drain_slack()
    spec_ms = (_now() - t_idle) * 1000.0
    remain = idle_ms - spec_ms
    if remain > 0:
        time.sleep(remain / 1000.0)
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
        workload="react",
        tp=0,
        e2e_ms=(t1 - t0) * 1000.0,
        fanout_ms=spec_ms,
        decode_ms=ttft,
        ttft_from_obs_ms=ttft,
        spec_ms=spec_ms,
        idle_ms=idle_ms,
        trunk_tokens=trunk_n,
        residual_tokens=[wrap_n, recov_n, obs_n],
        peak_kv_tokens=int(tree.live_kv_tokens()),
        m_cow_mib=cow,
        m_clone_mib=clone,
        kv_saving=saving,
        gpu_mem_mib=gpu_mem_mib(),
        known_suffix_hit_rate=m.known_suffix_hit_rate,
        decode_tokens=len(out),
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
        peaks.append(int(tree.live_kv_tokens()))
        _select_winner(eng, h.id, kids, winner=0)
    t_fan = _now()
    n_out = 0
    for h in handles:
        n_out += len(eng.generate(h.id, decode_n))
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


def _gen(llm: Any, SamplingParams: Any, TokensPrompt: Any, seqs: Sequence[Sequence[int]], n: int) -> None:
    prompts = [TokensPrompt(prompt_token_ids=list(s)) for s in seqs]
    params = SamplingParams(max_tokens=n, temperature=0.0)
    llm.generate(prompts, params, use_tqdm=False)


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
    _gen(llm, SamplingParams, TokensPrompt, [branches[0]], decode_n)
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
        workload="tot",
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
) -> RunMetrics:
    wrap, recov, obs = react_strings()
    trunk_ids = _tok_ids(llm, trunk)
    wrap_ids = _tok_ids(llm, wrap)
    recov_ids = _tok_ids(llm, recov)
    full = trunk_ids + wrap_ids + _tok_ids(llm, obs)
    t0 = _now()
    if prefix_cache:
        _gen(llm, SamplingParams, TokensPrompt, [trunk_ids], 1)
    time.sleep(idle_ms / 1000.0)
    t_obs = _now()
    _gen(llm, SamplingParams, TokensPrompt, [full], decode_n)
    ttft = (_now() - t_obs) * 1000.0
    t1 = _now()
    trunk_n = len(trunk_ids)
    residuals = [len(wrap_ids), len(recov_ids)]
    from forkserve.pages import clone_memory_bytes, cow_memory_bytes

    cow = _mib(cow_memory_bytes(trunk_n, residuals, cfg_bpt))
    clone = _mib(clone_memory_bytes(trunk_n, residuals, cfg_bpt, 2))
    peak = len(full) if not prefix_cache else (len(full))
    return RunMetrics(
        system=system,
        workload="react",
        tp=0,
        e2e_ms=(t1 - t0) * 1000.0,
        ttft_from_obs_ms=ttft,
        idle_ms=idle_ms,
        trunk_tokens=trunk_n,
        residual_tokens=residuals + [len(full) - trunk_n - len(wrap_ids)],
        peak_kv_tokens=int(peak),
        m_cow_mib=cow,
        m_clone_mib=clone,
        kv_saving=0.0 if clone <= 0 else 1.0 - (cow / clone),
        gpu_mem_mib=gpu_mem_mib(),
        decode_tokens=decode_n,
        branching=2,
        notes="sleep idle then prefill wrap+obs; no speculative overlap",
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
    _gen(llm, SamplingParams, TokensPrompt, winners, decode_n)
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
        sessions=len(trunks),
        branching=len(thoughts),
        notes="SxB independent prompts in one generate()",
    )


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
    system = args.system
    if not args.tp:
        args.tp = [1]
    tp = int(args.tp[0])
    model = args.model
    bpt = bytes_per_token_from_config(model)
    rows: list[RunMetrics] = []

    if system == "forkserve":
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

    if system == "forkserve":
        assert eng is not None and cfg is not None
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
    payload = {
        "system": system,
        "tp": tp,
        "model": model,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "rows": [r.to_dict() for r in rows],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)
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
        "system           tp  workload  e2e_ms  fanout_ms  ttft_obs_ms  peak_kv  M_CoW_MiB  M_clone_MiB  kv_save  vs_recompute",
        "-" * 118,
    ]
    # speedup vs vllm_recompute on same (tp, workload)
    base: dict[tuple[int, str], float] = {}
    for r in rows:
        if r["system"] == "vllm_recompute":
            key = (int(r["tp"]), r["workload"])
            metric = r["ttft_from_obs_ms"] if r["workload"] == "react" else r["e2e_ms"]
            base[key] = float(metric)

    for r in rows:
        key = (int(r["tp"]), r["workload"])
        metric = r["ttft_from_obs_ms"] if r["workload"] == "react" else r["e2e_ms"]
        b = base.get(key, 0.0)
        speed = (b / metric) if metric > 0 and b > 0 else 0.0
        lines.append(
            f"{r['system']:<16} {r['tp']:>2}  {r['workload']:<8} "
            f"{r['e2e_ms']:7.1f}  {r['fanout_ms']:9.1f}  {r['ttft_from_obs_ms']:11.1f}  "
            f"{r['peak_kv_tokens']:7d}  {r['m_cow_mib']:9.2f}  {r['m_clone_mib']:11.2f}  "
            f"{r['kv_saving']:7.2f}  {speed:5.2f}x"
        )
    return "\n".join(lines)


def orchestrate(args: argparse.Namespace) -> int:
    n_gpu = visible_gpu_count()
    tps = default_tps(n_gpu, args.tp)
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    devices = [x.strip() for x in vis.split(",") if x.strip()] if vis else [str(i) for i in range(n_gpu)]
    systems = list(args.systems)
    out_dir = Path(args.out).parent if args.out else Path("logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    print(
        f"orchestrating systems={systems} tp={tps} gpus={devices} "
        f"workloads={args.workloads} model={args.model}",
        flush=True,
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
                "--out",
                str(shard),
            ]
            if args.enforce_eager:
                cmd.append("--enforce-eager")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = dev
            env["PYTHONUNBUFFERED"] = "1"
            print(f"==> {system} tp={tp} devices={dev}", flush=True)
            proc = subprocess.run(cmd, env=env)
            if proc.returncode != 0:
                print(f"worker failed: {system} tp={tp} rc={proc.returncode}", flush=True)
                return proc.returncode
            data = json.loads(shard.read_text())
            all_rows.extend(data["rows"])

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
    print(f"wrote {dest}", flush=True)
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
    p.add_argument("--decode", type=int, default=16)
    p.add_argument("--idle-ms", type=float, default=2000.0)
    p.add_argument("--branching", type=int, default=4)
    p.add_argument("--sessions", type=int, default=2)
    p.add_argument("--trunk-tokens", type=int, default=512)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-batched-tokens", type=int, default=2048)
    p.add_argument("--gpu-util", type=float, default=float(os.environ.get("FORKSERVE_GPU_UTIL", "0.90")))
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--workloads", type=lambda s: _csv_list(s, WORKLOADS), default=list(WORKLOADS))
    p.add_argument("--out", default="logs/bench_gpu.json")
    args = p.parse_args(argv)
    if args.tp is None:
        args.tp = []
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.backend == "mock":
        rows = run_mock(args)
        payload = {"backend": "mock", "rows": [r.to_dict() for r in rows]}
        print(format_table(payload["rows"]))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2))
        return 0
    if args.worker:
        return worker_main(args)
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
