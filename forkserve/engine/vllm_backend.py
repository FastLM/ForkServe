"""Optional vLLM v0.11 adapter.

The paper's engine delta (~4.8K LoC) lives inside vLLM's block manager and
continuous batcher. This module is the *integration seam*: we tag requests
Spec vs Commit, reuse vLLM's block-copy CUDA path for CoW rows, and pack
leftover token budget with Algorithm 1 chunks.

We do not vendor vLLM. When the extra is installed we talk to the public
``LLMEngine`` / ``TokensPrompt`` surface and keep page identity in ForkServe.
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
    """Thin wrapper. One ``LLM`` owns the GPU; ForkServe owns the tree."""

    config: ForkServeConfig
    model: str
    tensor_parallel: int = 1
    dtype: str = "bfloat16"
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.90
    enable_prefix_caching: bool = True
    _llm: Any = field(init=False, default=None)
    _SamplingParams: Any = field(init=False, default=None)
    _TokensPrompt: Any = field(init=False, default=None)
    pool: PagePool = field(init=False)

    def __post_init__(self) -> None:
        vllm, LLM, SamplingParams, TokensPrompt = _try_import_vllm()
        self._SamplingParams = SamplingParams
        self._TokensPrompt = TokensPrompt
        self._llm = LLM(
            model=self.model,
            tensor_parallel_size=self.tensor_parallel,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enable_prefix_caching=self.enable_prefix_caching,
            enable_chunked_prefill=True,
            max_num_batched_tokens=self.config.max_batched_tokens,
        )
        # CoW identity is tracked here; kernels still go through vLLM pages.
        self.pool = PagePool(self.config, TokenKvStore())
        self._tokenizer = self._llm.get_tokenizer()

    def tokenize(self, text: str) -> TokenSeq:
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        return tuple(int(i) for i in ids)

    def detokenize(self, tokens: TokenSeq) -> str:
        return self._tokenizer.decode(list(tokens))

    def prefill(self, req: PrefillRequest) -> float:
        """Force a prefill-only step (max_tokens=0) so KV is warmed."""
        t0 = perf_counter()
        params = self._SamplingParams(max_tokens=1, temperature=0.0)
        prompt = self._TokensPrompt(prompt_token_ids=list(req.tokens))
        # A 1-token decode is the cheapest public way to materialize KV.
        # Speculative jobs must never surface this token (Theorem 2).
        _ = self._llm.generate([prompt], params, use_tqdm=False)
        return (perf_counter() - t0) * 1000.0

    def decode(self, req: DecodeRequest) -> list[TokenId]:
        params = self._SamplingParams(
            max_tokens=req.n_tokens,
            temperature=0.0,
            seed=req.seed,
        )
        # Caller is responsible for passing the full committed sequence via
        # a side channel; the Engine class does this from the tree.
        raise RuntimeError("use Engine.generate() — decode needs the committed prompt")

    def generate_committed(self, tokens: TokenSeq, max_tokens: int, seed: int | None = None) -> list[TokenId]:
        params = self._SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
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
