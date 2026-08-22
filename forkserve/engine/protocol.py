"""Engine-agnostic backend protocol.

ForkServe programs state and prefill, not sampling. The backend owns
kernels; we own page tables, admission, and commit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from forkserve.pages import PagePool
from forkserve.planner import PrefillChunk
from forkserve.types import NodeId, SessionId, TokenId, TokenSeq


@dataclass(slots=True)
class PrefillRequest:
    session: SessionId
    node_id: NodeId
    tokens: TokenSeq
    speculative: bool
    page_ids: Sequence[int]


@dataclass(slots=True)
class DecodeRequest:
    session: SessionId
    node_id: NodeId
    n_tokens: int
    seed: int | None = None


@dataclass(slots=True)
class BackendBatchResult:
    prefilled: int
    decoded: list[TokenId]
    elapsed_ms: float
    tbt_headroom_ms: float


class EngineBackend(Protocol):
    pool: PagePool

    def prefill(self, req: PrefillRequest) -> float:
        """Return wall-ms. Must be bit-identical for a given (tokens, parent KV)."""
        ...

    def decode(self, req: DecodeRequest) -> list[TokenId]: ...

    def run_batch(
        self,
        committed_prefills: Sequence[PrefillRequest],
        committed_decodes: Sequence[DecodeRequest],
        speculative: Sequence[PrefillChunk],
    ) -> BackendBatchResult: ...

    def cancel_prefill(self, node_id: NodeId) -> None: ...

    def tbt_headroom_ms(self) -> float: ...

    def free_hbm_bytes(self) -> float: ...

    def tokenize(self, text: str) -> TokenSeq: ...

    def detokenize(self, tokens: TokenSeq) -> str: ...
