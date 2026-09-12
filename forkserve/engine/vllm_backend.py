"""vLLM backend: one ``LLM.generate`` per serving phase, CoW alias, stock async.

Queued prefills coalesce: a trunk row that is a prefix of a child is dropped
so fan-out is one generate (APC pays trunk + fan-out). Speculative siblings
are dropped on the committed decode path — they must not steal HumanEval TTFT.
Custom ``scheduler_cls`` is off by default: the two-class reorder only helps
when spec and committed share a step, and a non-Async hook disables vLLM
async scheduling (the 2/4-GPU decode tax).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Sequence
from uuid import uuid4

from forkserve.config import ForkServeConfig
from forkserve.engine.protocol import (
    BackendBatchResult,
    DecodeRequest,
    PrefillRequest,
)
from forkserve.engine.vllm_loop import (
    forkserve_extra,
    get_two_class_scheduler,
    install_vllm_cow,
    release_cow_node,
)
from forkserve.pages import PagePool, TokenKvStore
from forkserve.planner import PrefillChunk
from forkserve.types import NodeId, TokenId, TokenSeq


def _try_import_vllm() -> Any:
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    return vllm, LLM, SamplingParams, TokensPrompt


def prompt_ids_of(req: PrefillRequest) -> list[int]:
    if req.full_prompt:
        return list(req.full_prompt)
    return list(req.tokens)


def _rides_decode(seq: Sequence[int], targets: Sequence[Sequence[int]]) -> bool:
    if not seq:
        return False
    n = len(seq)
    return any(len(t) >= n and list(t[:n]) == list(seq) for t in targets)


def _is_prefix(short: Sequence[int], long: Sequence[int]) -> bool:
    n = len(short)
    return n > 0 and len(long) > n and list(long[:n]) == list(short)


def drop_covered_prefills(pending: Sequence[PrefillRequest]) -> list[PrefillRequest]:
    """Drop a row whose prompt is a proper prefix of another queued row.

    Trunk ``open()`` and child residuals used to be two ``LLM.generate``
    rounds. Children already send ``trunk+residual``, so the trunk row is
    dead weight: one batched generate populates the same KV.
    """
    seqs = [prompt_ids_of(r) for r in pending]
    keep: list[PrefillRequest] = []
    for i, req in enumerate(pending):
        seq = seqs[i]
        if any(_is_prefix(seq, other) for j, other in enumerate(seqs) if j != i):
            continue
        keep.append(req)
    return keep


def split_pending_for_decode(
    pending: Sequence[PrefillRequest],
    decode_seqs: Sequence[Sequence[int]],
) -> tuple[list[PrefillRequest], list[PrefillRequest], list[PrefillRequest]]:
    """Fused prefixes ride decode; speculative leftovers are dropped (not flushed)."""
    fused: list[PrefillRequest] = []
    committed_rest: list[PrefillRequest] = []
    dropped_spec: list[PrefillRequest] = []
    for req in pending:
        seq = prompt_ids_of(req)
        if _rides_decode(seq, decode_seqs) or any(_is_prefix(seq, t) for t in decode_seqs):
            fused.append(req)
        elif req.speculative:
            dropped_spec.append(req)
        else:
            committed_rest.append(req)
    return fused, committed_rest, dropped_spec


def partition_fused(
    pending: Sequence[PrefillRequest],
    decode_tokens: Sequence[int],
) -> tuple[list[PrefillRequest], list[PrefillRequest]]:
    """Pending rows whose prompt is a prefix of decode can ride the decode generate()."""
    fused, committed_rest, dropped = split_pending_for_decode(pending, [decode_tokens])
    return fused, committed_rest + dropped


@dataclass
class VllmBackend:
    """One ``LLM`` owns the GPU and the engine loop; ForkServe owns the tree."""

    config: ForkServeConfig
    model: str
    tensor_parallel: int = 1
    dtype: str = "bfloat16"
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.90
    enable_prefix_caching: bool = True
    enforce_eager: bool = False
    two_class: bool = False
    cow_blocks: bool = True
    _llm: Any = field(init=False, default=None)
    _SamplingParams: Any = field(init=False, default=None)
    _TokensPrompt: Any = field(init=False, default=None)
    pool: PagePool = field(init=False)
    _prompts: dict[NodeId, list[int]] = field(init=False, default_factory=dict)
    _pending: list[PrefillRequest] = field(init=False, default_factory=list)
    generate_calls: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.cow_blocks:
            install_vllm_cow()
        vllm, LLM, SamplingParams, TokensPrompt = _try_import_vllm()
        self._SamplingParams = SamplingParams
        self._TokensPrompt = TokensPrompt
        llm_kwargs: dict[str, Any] = dict(
            model=self.model,
            tensor_parallel_size=self.tensor_parallel,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enable_prefix_caching=self.enable_prefix_caching,
            enable_chunked_prefill=True,
            max_num_batched_tokens=self.config.max_batched_tokens,
            enforce_eager=self.enforce_eager,
            disable_log_stats=True,
        )
        if self.two_class:
            # Pass the AsyncScheduler subclass, not the factory — vLLM's
            # issubclass check otherwise falls back to sync Scheduler.
            llm_kwargs["scheduler_cls"] = get_two_class_scheduler()
        self._llm = LLM(**llm_kwargs)
        self.pool = PagePool(self.config, TokenKvStore())
        self._tokenizer = self._llm.get_tokenizer()
        self._prompts = {}
        self._pending = []
        self.generate_calls = 0

    def tokenize(self, text: str) -> TokenSeq:
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        return tuple(int(i) for i in ids)

    def detokenize(self, tokens: TokenSeq) -> str:
        return self._tokenizer.decode(list(tokens))

    def _params(
        self,
        max_tokens: int,
        *,
        speculative: bool,
        node_id: NodeId | None,
        parent_node: NodeId | None = None,
        seed: int | None = None,
        session: str | None = None,
    ) -> Any:
        return self._SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            seed=seed,
            extra_args=forkserve_extra(
                speculative=speculative,
                node_id=int(node_id) if node_id is not None else None,
                parent_node=int(parent_node) if parent_node is not None else None,
                session=session,
            ),
        )

    def _generate(
        self,
        seqs: Sequence[Sequence[int]],
        max_tokens: int,
        *,
        speculative: Sequence[bool] | bool = False,
        node_ids: Sequence[NodeId | None] | None = None,
        parent_nodes: Sequence[NodeId | None] | None = None,
        sessions: Sequence[str | None] | None = None,
        seed: int | None = None,
    ) -> list[list[int]]:
        if not seqs:
            return []
        n = len(seqs)
        spec_flags = (
            [bool(speculative)] * n
            if isinstance(speculative, bool)
            else list(speculative)
        )
        nodes = list(node_ids) if node_ids is not None else [None] * n
        parents = list(parent_nodes) if parent_nodes is not None else [None] * n
        sess = list(sessions) if sessions is not None else [None] * n
        prompts = [self._TokensPrompt(prompt_token_ids=list(s)) for s in seqs]
        params = [
            self._params(
                max_tokens,
                speculative=spec_flags[i],
                node_id=nodes[i],
                parent_node=parents[i],
                seed=seed,
                session=sess[i] if i < len(sess) else None,
            )
            for i in range(n)
        ]
        self.generate_calls += 1
        outs = self._llm.generate(prompts, params, use_tqdm=False)
        result: list[list[int]] = []
        for o in outs:
            result.append([int(t) for t in o.outputs[0].token_ids])
        return result

    def prefill(self, req: PrefillRequest) -> float:
        seq = prompt_ids_of(req)
        if not seq:
            seq = list(self._prompts.get(req.node_id, [])) + list(req.tokens)
        self._prompts[req.node_id] = seq
        self._pending.append(
            PrefillRequest(
                session=req.session,
                node_id=req.node_id,
                tokens=req.tokens,
                speculative=req.speculative,
                page_ids=req.page_ids,
                full_prompt=tuple(seq),
                parent_node=req.parent_node,
            )
        )
        return 0.0

    def flush_prefills(self, *, speculative_only: bool = False) -> float:
        if not self._pending:
            return 0.0
        if speculative_only:
            reqs = [r for r in self._pending if r.speculative]
            keep = [r for r in self._pending if not r.speculative]
        else:
            reqs = drop_covered_prefills(self._pending)
            keep = []
        self._pending = keep
        if not reqs:
            return 0.0
        t0 = perf_counter()
        self._generate(
            [prompt_ids_of(r) for r in reqs],
            1,
            speculative=[r.speculative for r in reqs],
            node_ids=[r.node_id for r in reqs],
            parent_nodes=[r.parent_node for r in reqs],
            sessions=[str(r.session) for r in reqs],
        )
        return (perf_counter() - t0) * 1000.0

    def decode(self, req: DecodeRequest) -> list[TokenId]:
        seq = tuple(self._prompts.get(req.node_id, ()))
        return self.generate_committed(seq, req.n_tokens, seed=req.seed, node_id=req.node_id)

    def generate_committed(
        self,
        tokens: TokenSeq,
        max_tokens: int,
        seed: int | None = None,
        node_id: NodeId | None = None,
        parent_node: NodeId | None = None,
        session: str | None = None,
    ) -> list[TokenId]:
        outs = self.generate_committed_many(
            [tokens],
            max_tokens,
            seed=seed,
            node_ids=[node_id],
            parent_nodes=[parent_node],
            sessions=[session],
        )
        return outs[0] if outs else []

    def generate_committed_many(
        self,
        seqs: Sequence[TokenSeq],
        max_tokens: int,
        seed: int | None = None,
        node_ids: Sequence[NodeId | None] | None = None,
        parent_nodes: Sequence[NodeId | None] | None = None,
        sessions: Sequence[str | None] | None = None,
    ) -> list[list[TokenId]]:
        fused, committed_rest, dropped = split_pending_for_decode(self._pending, seqs)
        # Known-suffix / commit-tail / covered trunks ride this generate.
        # Spec siblings stay off the critical path (commit is the only
        # user-visible verb).
        _ = fused, dropped
        self._pending = drop_covered_prefills(committed_rest)
        if self._pending:
            self.flush_prefills()
        return self._generate(
            [list(s) for s in seqs],
            max_tokens,
            speculative=False,
            node_ids=node_ids,
            parent_nodes=parent_nodes,
            sessions=sessions,
            seed=seed,
        )

    def shutdown(self) -> None:
        llm = self._llm
        self._llm = None
        if llm is None:
            return
        for name in ("shutdown", "close"):
            closer = getattr(llm, name, None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    pass
                break
        del llm
        import gc

        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    def run_batch(
        self,
        committed_prefills: Sequence[PrefillRequest],
        committed_decodes: Sequence[DecodeRequest],
        speculative: Sequence[PrefillChunk],
    ) -> BackendBatchResult:
        t0 = perf_counter()
        n = 0
        for p in committed_prefills:
            self.prefill(p)
            n += len(p.tokens)
        for s in speculative:
            self.prefill(
                PrefillRequest(
                    session=s.session,
                    node_id=s.node_id or NodeId(0),
                    tokens=s.tokens,
                    speculative=True,
                    page_ids=(),
                    full_prompt=s.tokens,
                )
            )
            n += len(s.tokens)
        self.flush_prefills()
        decoded: list[TokenId] = []
        for d in committed_decodes:
            decoded.extend(self.decode(d))
        return BackendBatchResult(
            prefilled=n,
            decoded=decoded,
            elapsed_ms=(perf_counter() - t0) * 1000.0,
            tbt_headroom_ms=self.tbt_headroom_ms(),
        )

    def cancel_prefill(self, node_id: NodeId) -> None:
        self._pending = [r for r in self._pending if r.node_id != node_id]
        return None

    def release_node(self, node_id: NodeId, session: str | None = None) -> None:
        """Forget CoW snapshots for an aborted thought / wrap sibling."""
        self._pending = [r for r in self._pending if r.node_id != node_id]
        release_cow_node(int(node_id), session=session)

    def tbt_headroom_ms(self) -> float:
        return max(0.0, self.config.tbt_slo_ms * 0.4)

    def free_hbm_bytes(self) -> float:
        return max(0.0, self.config.hbm_capacity_bytes - self.pool.footprint_bytes())

    def request_id(self) -> str:
        return f"fs-{uuid4().hex[:12]}"
