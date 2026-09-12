"""Deterministic CPU backend for tests and paper-algorithm replay.

Tokenization is a stable hash of whitespace-split pieces plus a small
byte-fallback so LCP behaves like a real tokenizer without a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Sequence

from forkserve.config import ForkServeConfig
from forkserve.engine.protocol import (
    BackendBatchResult,
    DecodeRequest,
    PrefillRequest,
)
from forkserve.pages import PagePool, TokenKvStore
from forkserve.planner import PrefillChunk
from forkserve.types import NodeId, TokenId, TokenSeq, as_tokens


class HashTokenizer:
    """Stable, reversible-enough tokenizer for wrappers and tests."""

    def __init__(self) -> None:
        self._tok2id: dict[str, int] = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
        self._id2tok: dict[int, str] = {0: "<pad>", 1: "<bos>", 2: "<eos>"}
        self._next = 3

    def encode(self, text: str) -> TokenSeq:
        if not text:
            return ()
        pieces = _split(text)
        ids: list[int] = []
        for p in pieces:
            if p not in self._tok2id:
                self._tok2id[p] = self._next
                self._id2tok[self._next] = p
                self._next += 1
            ids.append(self._tok2id[p])
        return tuple(ids)

    def decode(self, tokens: TokenSeq) -> str:
        return "".join(self._id2tok.get(t, f"<{t}>") for t in tokens)


def _split(text: str) -> list[str]:
    out: list[str] = []
    buf = ""
    for ch in text:
        if ch.isspace():
            if buf:
                out.append(buf)
                buf = ""
            out.append(ch)
        elif ch in "{}[]()<>,.:;|/\\\"'`":
            if buf:
                out.append(buf)
                buf = ""
            out.append(ch)
        else:
            buf += ch
    if buf:
        out.append(buf)
    return out


@dataclass
class MockBackend:
    config: ForkServeConfig
    tokenizer: HashTokenizer = field(default_factory=HashTokenizer)
    pool: PagePool = field(init=False)
    kv: dict[NodeId, list[TokenId]] = field(default_factory=dict)
    decode_script: dict[NodeId, list[TokenId]] = field(default_factory=dict)
    _headroom_ms: float = 20.0

    def __post_init__(self) -> None:
        self.pool = PagePool(self.config, TokenKvStore())

    def tokenize(self, text: str) -> TokenSeq:
        return self.tokenizer.encode(text)

    def detokenize(self, tokens: TokenSeq) -> str:
        return self.tokenizer.decode(tokens)

    def prefill(self, req: PrefillRequest) -> float:
        t0 = perf_counter()
        seq = self.kv.setdefault(req.node_id, [])
        seq.extend(req.tokens)
        return (perf_counter() - t0) * 1000.0 + self.config.prefill_ms(len(req.tokens))

    def flush_prefills(self, *, speculative_only: bool = False) -> float:
        return 0.0

    def decode(self, req: DecodeRequest) -> list[TokenId]:
        script = self.decode_script.get(req.node_id, [])
        take = script[: req.n_tokens]
        self.decode_script[req.node_id] = script[req.n_tokens :]
        if not take:
            take = [2]  # eos
        self.kv.setdefault(req.node_id, []).extend(take)
        return list(take)

    def run_batch(
        self,
        committed_prefills: Sequence[PrefillRequest],
        committed_decodes: Sequence[DecodeRequest],
        speculative: Sequence[PrefillChunk],
    ) -> BackendBatchResult:
        t0 = perf_counter()
        n = 0
        decoded: list[TokenId] = []
        for p in committed_prefills:
            self.prefill(p)
            n += len(p.tokens)
        for d in committed_decodes:
            decoded.extend(self.decode(d))
        for s in speculative:
            fake = PrefillRequest(
                session=s.session,
                node_id=s.node_id or NodeId(0),
                tokens=as_tokens(s.tokens),
                speculative=True,
                page_ids=(),
            )
            self.prefill(fake)
            n += len(s.tokens)
        elapsed = (perf_counter() - t0) * 1000.0
        return BackendBatchResult(
            prefilled=n,
            decoded=decoded,
            elapsed_ms=elapsed,
            tbt_headroom_ms=self._headroom_ms,
        )

    def cancel_prefill(self, node_id: NodeId) -> None:
        return None

    def tbt_headroom_ms(self) -> float:
        return self._headroom_ms

    def free_hbm_bytes(self) -> float:
        return max(0.0, self.config.hbm_capacity_bytes - self.pool.footprint_bytes())
