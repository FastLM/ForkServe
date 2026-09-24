"""Disaggregated prefill connector (vLLM-shaped, control-plane).

vLLM splits prefill and decode across instances and moves KV through a
connector (``insert`` / ``drop_select``, see
https://docs.vllm.ai/en/stable/features/disagg_prefill/). Stock disagg
does **not** raise tokens/sec — it isolates TTFT from ITL. Combined with
APP it *does* cut prefill-instance work: losers are never inserted, so
the decode instance never receives their pages.

This module is the control-plane pipe. A production mapping replaces
``insert`` with NixlConnector / LMCache / Mooncake and keeps the gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from forkserve.config import ForkServeConfig
from forkserve.hash_forkserve import BlockHash, hash_block
from forkserve.prefill_prune import PrefillAction, PrefillDecision, PrefillPlan
from forkserve.types import TokenSeq, as_tokens


@dataclass(slots=True)
class KvEnvelope:
    """One sequence's KV staged for the decode instance."""

    request_id: str
    tokens: int
    shipped: bool
    reason: str


@dataclass(slots=True)
class TransferResult:
    inserted: int = 0
    dropped: int = 0
    shipped_tokens: int = 0
    skipped_tokens: int = 0
    envelopes: list[KvEnvelope] = field(default_factory=list)
    transfer_ms: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "inserted": float(self.inserted),
            "dropped": float(self.dropped),
            "shipped_tokens": float(self.shipped_tokens),
            "skipped_tokens": float(self.skipped_tokens),
            "transfer_ms": self.transfer_ms,
        }


class DisaggPrefillConnector:
    """LookupBuffer-style pipe: ``insert`` is non-blocking, ``drop_select`` waits.

    APP decisions gate ``insert``. Hash-local hits and draft-skipped
    branches never enter the buffer. Early-abort ships only the prefix
    that was actually prefilled (usually dropped before decode).
    """

    def __init__(self, config: ForkServeConfig | None = None) -> None:
        self.cfg = config or ForkServeConfig()
        self._buf: dict[str, KvEnvelope] = {}
        # Blocks already inserted for this decode instance. A later sibling
        # or session that repeats the prefix does not cross the connector.
        self._shipped: set[BlockHash] = set()

    def insert(
        self,
        request_id: str,
        tokens: TokenSeq | Sequence[int] | int,
        *,
        ship: bool,
        reason: str = "prefill",
    ) -> KvEnvelope:
        n = tokens if isinstance(tokens, int) else len(as_tokens(tokens))
        env = KvEnvelope(request_id, n, ship, reason)
        if ship:
            self._buf[request_id] = env
        return env

    def drop_select(self, request_id: str) -> KvEnvelope | None:
        """Blocking take: decode instance pulls the staged KV."""
        return self._buf.pop(request_id, None)

    def gate(self, plan: PrefillPlan, *, request_prefix: str = "b") -> TransferResult:
        """Insert only APP survivors. Returns transfer accounting."""
        result = TransferResult()
        us = self.cfg.disagg_transfer_us_per_token
        page = max(1, self.cfg.page_size)
        for dec in plan.decisions:
            rid = f"{request_prefix}{dec.index}"
            ship, reason, n = _should_ship(dec)
            if ship and dec.prompt_tokens:
                begin = dec.matched_tokens
                if begin <= 0:
                    begin = max(0, len(dec.prompt_tokens) - dec.work_tokens)
                n = _novel_tokens(
                    dec.prompt_tokens,
                    begin,
                    dec.work_tokens,
                    page,
                    self._shipped,
                )
                ship = n > 0
                if not ship:
                    reason = "prefix_dedup"
            env = self.insert(rid, n, ship=ship, reason=reason)
            result.envelopes.append(env)
            if ship:
                result.inserted += 1
                result.shipped_tokens += n
            else:
                result.dropped += 1
                result.skipped_tokens += dec.residual_tokens
        result.transfer_ms = (result.shipped_tokens * us) / 1000.0
        return result

    def decode_pull(self, request_ids: Sequence[str]) -> list[KvEnvelope]:
        out: list[KvEnvelope] = []
        for rid in request_ids:
            env = self.drop_select(rid)
            if env is not None:
                out.append(env)
        return out


def _should_ship(dec: PrefillDecision) -> tuple[bool, str, int]:
    if dec.action is PrefillAction.DRAFT_SKIP:
        return False, "draft_skip", 0
    if dec.action is PrefillAction.EARLY_ABORT:
        return False, "early_abort", 0
    if dec.action is PrefillAction.HASH_SKIP:
        # Decode instance can APC-hit the same published blocks.
        return False, "hash_local", 0
    if not dec.keep:
        return False, dec.reason, 0
    n = dec.transfer_tokens or dec.work_tokens or dec.residual_tokens
    return True, dec.action.value, n


def _novel_tokens(
    prompt: TokenSeq,
    matched: int,
    work: int,
    page_size: int,
    shipped: set[BlockHash],
) -> int:
    """Tokens in ``[matched, matched+work)`` whose block is not yet inserted.

    Blocks are hashed on the parent chain of the full prompt, so a sibling
    that repeats a prefix does not cross the connector again.
    """
    tokens = as_tokens(prompt)
    if not tokens or work <= 0:
        return 0
    begin = min(len(tokens), max(0, matched))
    end = min(len(tokens), begin + work)
    n = 0
    parent: BlockHash | None = None
    i = 0
    while i + page_size <= len(tokens):
        chunk = tokens[i : i + page_size]
        bh = hash_block(parent, chunk)
        parent = bh
        block_end = i + page_size
        overlaps = block_end > begin and i < end
        if bh not in shipped and overlaps:
            left = max(i, begin)
            right = min(block_end, end)
            n += max(0, right - left)
            shipped.add(bh)
        elif bh not in shipped and block_end <= begin:
            shipped.add(bh)
        i += page_size
    if i < len(tokens):
        bh = hash_block(parent, tokens[i:], extra=b"tail")
        if bh not in shipped and len(tokens) > begin and i < end:
            n += max(0, min(len(tokens), end) - max(i, begin))
            shipped.add(bh)
    return n


def transfer_ms(tokens: int, us_per_token: float) -> float:
    return (max(0, tokens) * us_per_token) / 1000.0
