"""Draft-score and early-abort speculative branches before they burn prefill.

Known-suffix ToT residuals are short; the win is skipping GPU prefill on
low-mass siblings, not rewriting the winner decode. A draft model is optional:
without weights we use n-gram / entropy / illegal-trace heuristics so the
control plane still drops hopeless branches.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, replace
from typing import Sequence

from forkserve.config import ForkServeConfig
from forkserve.types import TokenSeq

ADMIT_MODES = ("score", "winner", "top_m", "alpha")
_ADMIT_ALIASES = {
    "winner_only": "winner",
    "skip": "winner",
    "topk": "top_m",
    "keep_m": "top_m",
    "frac": "alpha",
}

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
        if _REPEAT.search(raw):
            return 0.02, "loop"
        if _ILLEGAL_MATH.search(raw):
            # Junk text, not a repeat loop. Below the 0.45 contest bar,
            # above 0.15 so a low threshold still starts it.
            return 0.22, "illegal"
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


def apply_admit_mode(
    cfg: ForkServeConfig,
    mode: str = "score",
    *,
    keep_m: int = 0,
    alpha: float = 0.5,
    threshold: float | None = None,
) -> ForkServeConfig:
    """Select one APP admission rule. The others stay off.

    ``score`` keeps every residual at or above ``prune_threshold``.
    ``winner`` is skip_known_losers. ``top_m`` keeps ``keep_m`` by G/C.
    ``alpha`` keeps ``floor(alpha * k)`` by draft score, winner included.
    """
    raw = str(mode or "score").strip().replace("-", "_")
    key = _ADMIT_ALIASES.get(raw, raw)
    if key not in ADMIT_MODES:
        raise ValueError(f"unknown admit_mode {mode!r}; want {ADMIT_MODES}")
    cfg.admit_mode = key
    if threshold is not None:
        cfg.prune_threshold = float(threshold)
    if key == "winner":
        cfg.skip_known_losers = True
        cfg.gc_admit = False
        cfg.prefill_keep_m = 0
    elif key == "top_m":
        cap = int(keep_m) if keep_m else int(cfg.prefill_keep_m or 0)
        cfg.skip_known_losers = False
        cfg.gc_admit = True
        cfg.prefill_keep_m = cap if cap > 0 else 2
    elif key == "alpha":
        cfg.skip_known_losers = False
        cfg.gc_admit = False
        cfg.prefill_keep_m = 0
        cfg.admit_alpha = float(alpha)
    else:
        cfg.skip_known_losers = False
        cfg.gc_admit = False
        cfg.prefill_keep_m = 0
    return cfg


def plus_config(base: ForkServeConfig | None = None) -> ForkServeConfig:
    """Knobs that turn baseline ForkServe into ForkServe+ / APP."""
    cfg = ForkServeConfig() if base is None else replace(base)
    cfg.lazy_abort = True
    cfg.pointer_swap = True
    cfg.prune_enabled = True
    cfg.hash_prune = True
    cfg.disagg_prefill = True
    cfg.share_prefixes = True
    cfg.slack_fill = True
    cfg.spec_pool_frac = 0.25 if cfg.spec_pool_frac <= 0 else cfg.spec_pool_frac
    # Marker strings cut the answer off (#### before the number, </think>
    # before the R1 reply). Answer-complete stop replaces them.
    cfg.decode_stop = ()
    # 0.15 only drops illegal loops. Contest fan-out uses a higher bar so
    # low-mass siblings are skipped before prefill. Override with
    # FORKSERVE_PRUNE_THRESHOLD.
    raw_thr = os.environ.get("FORKSERVE_PRUNE_THRESHOLD", "").strip()
    cfg.prune_threshold = float(raw_thr) if raw_thr else max(cfg.prune_threshold, 0.45)
    raw_m = os.environ.get("FORKSERVE_KEEP_M", "").strip()
    raw_a = os.environ.get("FORKSERVE_ADMIT_ALPHA", "").strip()
    apply_admit_mode(
        cfg,
        os.environ.get("FORKSERVE_ADMIT_MODE", "score").strip() or "score",
        keep_m=int(raw_m) if raw_m else 0,
        alpha=float(raw_a) if raw_a else 0.5,
    )
    flags = os.environ.get("FORKSERVE_PLUS_FLAGS", "").strip()
    if flags:
        want = {p.strip() for p in flags.split(",") if p.strip()}
        if "skip" in want:
            apply_admit_mode(cfg, "winner")
        elif "topm" in want or "top_m" in want:
            apply_admit_mode(cfg, "top_m", keep_m=int(raw_m) if raw_m else 2)
        elif "alpha" in want:
            apply_admit_mode(cfg, "alpha", alpha=float(raw_a) if raw_a else 0.5)
        cfg.prune_enabled = bool(want & {"skip", "prune", "topm", "top_m", "alpha"}) or cfg.prune_enabled
        cfg.hash_prune = "hash" in want or cfg.hash_prune
        cfg.answer_stop = "auto" if "stop" in want else cfg.answer_stop
        cfg.lazy_abort = "lazy" in want or cfg.lazy_abort
    # Answer-stop stays off unless the caller or FORKSERVE_PLUS_FLAGS=stop
    # asks for it. APC runs to max_tokens; a default stop would shorten
    # only the ForkServe+ decode.
    return cfg
