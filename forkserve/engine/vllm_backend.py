"""vLLM backend: CoW block-pool bit + two-class scheduler in the engine loop.

``LLM(..., scheduler_cls=TwoClassVllmScheduler)`` is the comparison
substrate. Speculative 1-token warmups are tagged and never surfaced
(Theorem 2).
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
)
from forkserve.pages import PagePool, TokenKvStore
from forkserve.planner import PrefillChunk
from forkserve.types import NodeId, TokenId, TokenSeq


def _try_import_vllm() -> Any:
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    return vllm, LLM, SamplingParams, TokensPrompt


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
    two_class: bool = True
    cow_blocks: bool = True
    _llm: Any = field(init=False, default=None)
    _SamplingParams: Any = field(init=False, default=None)
    _TokensPrompt: Any = field(init=False, default=None)
    pool: PagePool = field(init=False)
    _prompts: dict[NodeId, list[int]] = field(init=False, default_factory=dict)

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
        )
        if self.two_class:
            llm_kwargs["scheduler_cls"] = get_two_class_scheduler()
        self._llm = LLM(**llm_kwargs)
        self.pool = PagePool(self.config, TokenKvStore())
        self._tokenizer = self._llm.get_tokenizer()
        self._prompts = {}

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
    ) -> Any:
        return self._SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            seed=seed,
            extra_args=forkserve_extra(
                speculative=speculative,
                node_id=int(node_id) if node_id is not None else None,
                parent_node=int(parent_node) if parent_node is not None else None,
            ),
        )

    def prefill(self, req: PrefillRequest) -> float:
        t0 = perf_counter()
        seq = list(req.full_prompt) if req.full_prompt else (
            list(self._prompts.get(req.node_id, [])) + list(req.tokens)
        )
        self._prompts[req.node_id] = seq
        params = self._params(
            1,
            speculative=req.speculative,
            node_id=req.node_id,
            parent_node=req.parent_node,
        )
        prompt = self._TokensPrompt(prompt_token_ids=seq)
        _ = self._llm.generate([prompt], params, use_tqdm=False)
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
    ) -> list[TokenId]:
        params = self._params(
            max_tokens,
            speculative=False,
            node_id=node_id,
            seed=seed,
        )
        prompt = self._TokensPrompt(prompt_token_ids=list(tokens))
        outs = self._llm.generate([prompt], params, use_tqdm=False)
        text_ids = outs[0].outputs[0].token_ids
        return [int(t) for t in text_ids]

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
        return BackendBatchResult(
            prefilled=n,
            decoded=[],
            elapsed_ms=(perf_counter() - t0) * 1000.0,
            tbt_headroom_ms=self.tbt_headroom_ms(),
        )

    def cancel_prefill(self, node_id: NodeId) -> None:
        return None

    def tbt_headroom_ms(self) -> float:
        return max(0.0, self.config.tbt_slo_ms * 0.4)

    def free_hbm_bytes(self) -> float:
        return max(0.0, self.config.hbm_capacity_bytes - self.pool.footprint_bytes())

    def request_id(self) -> str:
        return f"fs-{uuid4().hex[:12]}"
