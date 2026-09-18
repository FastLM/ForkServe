"""Draft-score and early-abort speculative branches before they burn prefill.

Known-suffix ToT residuals are short; the win is skipping GPU prefill on
low-mass siblings, not rewriting the winner decode. A draft model is optional:
without weights we use n-gram / entropy / illegal-trace heuristics so the
control plane still drops hopeless branches.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Sequence

from forkserve.config import ForkServeConfig
from forkserve.types import TokenSeq

_REPEAT = re.compile(r"(.{8,40})\1{3,}")
_ILLEGAL_MATH = re.compile(r"\*{4,}|unk|nan|undefined", re.I)


@dataclass(slots=True)
class BranchScore:
    index: int
    keep: bool
    score: float
    reason: str
    early_abort: bool = False


@dataclass
class BranchPruner:
    """Rank residuals; keep the winner plus any sibling above ``threshold``."""

    config: ForkServeConfig
    winner: int = 0

    def score_text(self, text: str) -> tuple[float, str]:
        raw = text or ""
        if not raw.strip():
            return 0.0, "empty"
        if _REPEAT.search(raw) or _ILLEGAL_MATH.search(raw):
            return 0.02, "illegal_or_loop"
        n = max(len(raw), 1)
        uniq = len(set(raw))
        entropy = uniq / n
        # Collapse whitespace-only thought prefixes.
        alpha = sum(ch.isalnum() for ch in raw) / n
        score = 0.55 * min(1.0, alpha * 2.0) + 0.45 * min(1.0, entropy * 8.0)
        return float(min(1.0, max(0.0, score))), "ok"

    def score_tokens(self, tokens: TokenSeq) -> tuple[float, str]:
        if not tokens:
            return 0.0, "empty"
        n = len(tokens)
        uniq = len(set(int(t) for t in tokens))
        # High unique ratio on a short wrapper is fine; long high-entropy junk is not.
        if n >= 16 and uniq / n > 0.95:
            return 0.08, "high_entropy"
        return min(1.0, 0.4 + 0.6 * (uniq / n)), "ok"

    def rank(
        self,
        residuals: Sequence[str] | Sequence[TokenSeq],
        *,
        threshold: float | None = None,
    ) -> list[BranchScore]:
        thresh = self.config.prune_threshold if threshold is None else threshold
        out: list[BranchScore] = []
        for i, residual in enumerate(residuals):
            if isinstance(residual, str):
                score, reason = self.score_text(residual)
            else:
                score, reason = self.score_tokens(tuple(residual))
            keep = i == self.winner or score >= thresh
            out.append(BranchScore(index=i, keep=keep, score=score, reason=reason))
        if not any(s.keep for s in out) and out:
            out[self.winner if self.winner < len(out) else 0].keep = True
        return out

    def early_abort_prefix(self, text: str, *, frac: float | None = None) -> bool:
        """True when the first ``early_prune_frac`` of a residual already looks dead."""
        use = self.config.early_prune_frac if frac is None else frac
        if not text:
            return False
        cut = max(1, int(math.ceil(len(text) * use)))
        score, reason = self.score_text(text[:cut])
        return score < self.config.prune_threshold and reason != "ok"


def plus_config(base: ForkServeConfig | None = None) -> ForkServeConfig:
    """Knobs that turn baseline ForkServe into ForkServe+."""
    cfg = ForkServeConfig() if base is None else replace(base)
    cfg.lazy_abort = True
    cfg.pointer_swap = True
    cfg.prune_enabled = True
    cfg.spec_pool_frac = 0.25 if cfg.spec_pool_frac <= 0 else cfg.spec_pool_frac
    if not cfg.decode_stop:
        cfg.decode_stop = ("####", "\\boxed", "</think>")
    return cfg
