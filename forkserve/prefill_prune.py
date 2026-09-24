"""Advanced Prefill Pruning (APP): hash skip ⊕ draft prune ⊕ early abort.

Prefill is the expensive half of a ToT / ReAct fan-out. APC (hash_prefill)
skips *already published* full blocks. vLLM disagg_prefill ships whatever
the prefill instance computed. APP sits between them and decides, per
branch, whether GPU prefill (and later KV transfer) should happen at all.

Skip layers, cheapest first:

1. **Hash skip** — ``hash(parent, block_tokens, extra)`` already published
   (HashForkServe / APC). Zero FLOPs for the matched prefix.
2. **Draft prune** — residual looks hopeless (illegal / loop / high entropy,
   or an optional draft-model score). Never start prefill.
3. **Early abort** — first ``early_prune_frac`` of a residual already fails;
   cancel the remaining chunks (chunked / disagg prefill).
4. **Disagg gate** — only survivors are inserted into the KV pipe
   (see ``forkserve.disagg``). Losers never leave the prefill instance.

Winner index is always kept (output identity). Spec pages stay unpublished
until LCP commit (Theorem 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from forkserve.config import ForkServeConfig
from forkserve.hash_forkserve import BlockHash, hash_block
from forkserve.prune import BranchPruner, BranchScore
from forkserve.types import TokenSeq, as_tokens, lcp_len


class PrefillAction(str, Enum):
    PREFILL = "prefill"
    HASH_SKIP = "hash_skip"
    HASH_PARTIAL = "hash_partial"
    DRAFT_SKIP = "draft_skip"
    EARLY_ABORT = "early_abort"


@dataclass(slots=True)
class PrefillDecision:
    """One branch residual after APP."""

    index: int
    action: PrefillAction
    keep: bool
    score: float
    reason: str
    residual_tokens: int
    matched_tokens: int = 0
    work_tokens: int = 0  # tokens that still need GPU prefill
    transfer_tokens: int = 0  # tokens that would ship under disagg
    # Miss tail actually prefilled. ``prompt_tokens`` is the full sequence
    # so the connector can hash blocks on the same parent chain.
    span: TokenSeq = ()
    prompt_tokens: TokenSeq = ()

    @property
    def prefill(self) -> bool:
        return self.action in (PrefillAction.PREFILL, PrefillAction.HASH_PARTIAL)


@dataclass(slots=True)
class PrefillPlan:
    decisions: list[PrefillDecision] = field(default_factory=list)
    prefill_tokens: int = 0
    skipped_tokens: int = 0
    transfer_tokens: int = 0
    hash_skips: int = 0
    draft_skips: int = 0
    early_aborts: int = 0
    kept: int = 0

    def to_dict(self) -> dict[str, float]:
        return {
            "prefill_tokens": float(self.prefill_tokens),
            "skipped_tokens": float(self.skipped_tokens),
            "transfer_tokens": float(self.transfer_tokens),
            "hash_skips": float(self.hash_skips),
            "draft_skips": float(self.draft_skips),
            "early_aborts": float(self.early_aborts),
            "kept": float(self.kept),
            "branches": float(len(self.decisions)),
        }


class PrefillHashIndex:
    """APC-style full-block index without owning physical pages.

    Engine and the APP bench publish committed prompts here so a later
    fan-out / replay can skip GPU prefill on the matched prefix. Partial
    tails are never inserted (APC note 1). Speculative pages must not
    call ``publish``.
    """

    def __init__(self, page_size: int = 16, *, algo: str = "sha256") -> None:
        self.page_size = page_size
        self.algo = algo
        self.cached: set[BlockHash] = set()
        self.hits = 0
        self.misses = 0
        self.published = 0

    def publish(
        self,
        tokens: TokenSeq | Sequence[int],
        *,
        extra: bytes = b"",
        cache_salt: bytes = b"",
    ) -> int:
        tokens = as_tokens(tokens)
        n = 0
        parent: BlockHash | None = None
        i = 0
        while i + self.page_size <= len(tokens):
            chunk = tokens[i : i + self.page_size]
            ex = (cache_salt + extra) if i == 0 else extra
            bh = hash_block(parent, chunk, extra=ex, algo=self.algo)
            if bh not in self.cached:
                self.cached.add(bh)
                self.published += 1
            parent = bh
            i += self.page_size
            n += 1
        return n

    def lookup(
        self,
        tokens: TokenSeq | Sequence[int],
        *,
        extra: bytes = b"",
        cache_salt: bytes = b"",
    ) -> int:
        """Return how many leading tokens are full-block hash hits."""
        tokens = as_tokens(tokens)
        parent: BlockHash | None = None
        i = 0
        while i + self.page_size <= len(tokens):
            chunk = tokens[i : i + self.page_size]
            ex = (cache_salt + extra) if i == 0 else extra
            bh = hash_block(parent, chunk, extra=ex, algo=self.algo)
            if bh not in self.cached:
                self.misses += 1
                return i
            self.hits += 1
            parent = bh
            i += self.page_size
        if i < len(tokens):
            self.misses += 1
        return i


@dataclass
class PrefillPruner:
    """Compose hash / draft / early-abort into one per-branch plan."""

    config: ForkServeConfig
    winner: int = 0
    hash_index: PrefillHashIndex | None = None

    def __post_init__(self) -> None:
        self._draft = BranchPruner(self.config, winner=self.winner)

    def plan(
        self,
        residuals: Sequence[str] | Sequence[TokenSeq],
        *,
        full_prompts: Sequence[TokenSeq] | None = None,
        token_counts: Sequence[int] | None = None,
        threshold: float | None = None,
        disagg: bool | None = None,
    ) -> PrefillPlan:
        """Rank ``k`` residuals. Winner is never dropped."""
        ship = self.config.disagg_prefill if disagg is None else disagg
        scores = self._draft.rank(residuals, threshold=threshold)
        out = PrefillPlan()
        prompts: list[TokenSeq] = []
        for row in scores:
            residual = residuals[row.index]
            n_tok = (
                int(token_counts[row.index])
                if token_counts is not None and row.index < len(token_counts)
                else _residual_len(residual)
            )
            prompt = (
                as_tokens(full_prompts[row.index])
                if full_prompts is not None and row.index < len(full_prompts)
                else _as_tokenish(residual)
            )
            dec = self._decide(row, n_tok, prompt, residual)
            while len(prompts) <= row.index:
                prompts.append(())
            prompts[row.index] = prompt
            dec.prompt_tokens = prompt
            dec.span = _miss_span(prompt, dec.matched_tokens, dec.work_tokens)
            out.decisions.append(dec)
        self._share_prefixes(out.decisions, prompts)
        self._admit_by_gc(out.decisions)
        for dec in out.decisions:
            n_tok = dec.residual_tokens
            if ship and dec.keep and dec.action is not PrefillAction.HASH_SKIP:
                dec.transfer_tokens = dec.work_tokens
                dec.span = _miss_span(
                    prompts[dec.index] if dec.index < len(prompts) else (),
                    dec.matched_tokens,
                    dec.work_tokens,
                )
            else:
                dec.transfer_tokens = 0
                if not dec.keep:
                    dec.span = ()
            out.prefill_tokens += dec.work_tokens
            out.skipped_tokens += max(0, n_tok - dec.work_tokens)
            out.transfer_tokens += dec.transfer_tokens
            if dec.keep:
                out.kept += 1
            if dec.action is PrefillAction.HASH_SKIP:
                out.hash_skips += 1
            elif dec.action is PrefillAction.DRAFT_SKIP:
                out.draft_skips += 1
            elif dec.action is PrefillAction.EARLY_ABORT:
                out.early_aborts += 1
        return out

    def _share_prefixes(self, decisions: list[PrefillDecision], prompts: list[TokenSeq]) -> None:
        """Page-align the LCP of kept prompts and prefill each page once."""
        if not self.config.share_prefixes:
            return
        page = max(1, self.config.page_size)
        admitted: list[TokenSeq] = []
        order = sorted(
            decisions,
            key=lambda d: (d.index == self.winner, d.score, -d.index),
            reverse=True,
        )
        for dec in order:
            if not dec.keep or dec.action is PrefillAction.HASH_SKIP:
                if dec.keep and dec.action is PrefillAction.HASH_SKIP:
                    prompt = prompts[dec.index] if dec.index < len(prompts) else ()
                    if prompt:
                        admitted.append(prompt)
                continue
            prompt = prompts[dec.index] if dec.index < len(prompts) else ()
            if not prompt:
                continue
            best = dec.matched_tokens
            for prev in admitted:
                best = max(best, lcp_len(prompt, prev))
            shared = (best // page) * page
            if shared > dec.matched_tokens:
                dec.matched_tokens = shared
                miss = max(0, len(prompt) - shared)
                if miss <= 0:
                    dec.action = PrefillAction.HASH_SKIP
                    dec.work_tokens = 0
                    dec.reason = "sibling_prefix"
                    dec.span = ()
                else:
                    dec.action = PrefillAction.HASH_PARTIAL
                    dec.reason = "sibling_prefix"
                    dec.work_tokens = miss
                    dec.span = prompt[shared:]
            if dec.keep:
                admitted.append(prompt)

    def _admit_by_gc(self, decisions: list[PrefillDecision]) -> None:
        """Keep the winner and the next best G/C residuals. The rest never start."""
        cap = self.config.prefill_keep_m
        if not self.config.gc_admit or cap <= 0:
            return
        ranked = sorted(
            (d for d in decisions if d.keep and d.index != self.winner),
            key=lambda d: d.score / max(d.work_tokens, 1),
            reverse=True,
        )
        for dec in ranked[max(0, cap - 1) :]:
            dec.keep = False
            dec.action = PrefillAction.DRAFT_SKIP
            dec.reason = "marginal_gc"
            dec.work_tokens = 0
            dec.transfer_tokens = 0
            dec.span = ()

    def _decide(
        self,
        row: BranchScore,
        n_tok: int,
        prompt: TokenSeq,
        residual: str | TokenSeq,
    ) -> PrefillDecision:
        if (
            self.config.skip_known_losers
            and self.winner >= 0
            and row.index != self.winner
        ):
            return PrefillDecision(
                index=row.index,
                action=PrefillAction.DRAFT_SKIP,
                keep=False,
                score=row.score,
                reason="winner_selected",
                residual_tokens=n_tok,
                work_tokens=0,
            )
        matched = 0
        if self.config.hash_prune and self.hash_index is not None and prompt:
            matched = self.hash_index.lookup(prompt)
        # ``matched`` is a prefix of the *full* prompt (trunk+residual).
        # Residual work is the miss tail, not ``n_tok - matched``.
        full_n = len(prompt) if prompt else n_tok
        miss = max(0, full_n - matched) if matched else n_tok
        # Hash hit on the residual (or the residual portion of the prompt).
        if self.config.hash_prune and matched > 0 and miss <= 0:
            return PrefillDecision(
                index=row.index,
                action=PrefillAction.HASH_SKIP,
                keep=True,
                score=1.0,
                reason="hash_hit",
                residual_tokens=n_tok,
                matched_tokens=matched,
                work_tokens=0,
            )
        if self.config.hash_prune and matched > 0 and miss > 0 and row.keep:
            return PrefillDecision(
                index=row.index,
                action=PrefillAction.HASH_PARTIAL,
                keep=True,
                score=row.score,
                reason="hash_partial",
                residual_tokens=n_tok,
                matched_tokens=matched,
                work_tokens=miss,
            )
        if not row.keep:
            text = residual if isinstance(residual, str) else ""
            # Prefer early-abort accounting when the *prefix* already fails
            # (chunked prefill paid a fraction) vs never starting.
            if self.config.prune_enabled and text and self._draft.early_abort_prefix(text):
                frac = max(0.0, min(1.0, self.config.early_prune_frac))
                work = max(1, int(n_tok * frac)) if n_tok else 0
                return PrefillDecision(
                    index=row.index,
                    action=PrefillAction.EARLY_ABORT,
                    keep=False,
                    score=row.score,
                    reason=row.reason,
                    residual_tokens=n_tok,
                    work_tokens=work,
                )
            return PrefillDecision(
                index=row.index,
                action=PrefillAction.DRAFT_SKIP,
                keep=False,
                score=row.score,
                reason=row.reason,
                residual_tokens=n_tok,
                work_tokens=0,
            )
        return PrefillDecision(
            index=row.index,
            action=PrefillAction.PREFILL,
            keep=True,
            score=row.score,
            reason=row.reason,
            residual_tokens=n_tok,
            matched_tokens=matched,
            work_tokens=n_tok,
        )


def _miss_span(prompt: TokenSeq, matched: int, work: int) -> TokenSeq:
    if not prompt or work <= 0:
        return ()
    start = min(len(prompt), max(0, matched))
    return prompt[start : start + work]


def _residual_len(residual: str | TokenSeq) -> int:
    if isinstance(residual, str):
        return max(len(residual.split()), len(residual) // 4, 1 if residual else 0)
    return len(residual)


def _as_tokenish(residual: str | TokenSeq) -> TokenSeq:
    if isinstance(residual, str):
        return ()
    return as_tokens(residual)


def app_config(base: ForkServeConfig | None = None) -> ForkServeConfig:
    """ForkServe+ / APP: lazy abort, pointer swap, prune, hash skip, disagg gate."""
    from forkserve.prune import plus_config

    cfg = plus_config(base)
    cfg.hash_prune = True
    cfg.disagg_prefill = True
    return cfg
