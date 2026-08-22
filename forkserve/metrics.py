"""Counters that feed §9 tables: hit rates, residual/trunk HBM, TBT, goodput."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SessionMetrics:
    forks: int = 0
    speculates: int = 0
    commits: int = 0
    joins: int = 0
    aborts: int = 0
    known_suffix_hits: int = 0
    residual_full_hits: int = 0
    residual_partial_hits: int = 0
    known_suffix_misses: int = 0
    aborted_residual_tokens: int = 0
    spec_tokens: int = 0
    committed_tokens: int = 0
    commit_tail_tokens: int = 0
    cancelled_chunks: int = 0
    ttft_ms: list[float] = field(default_factory=list)
    tbt_ms: list[float] = field(default_factory=list)

    def record_commit(
        self,
        *,
        known_hit: bool,
        residual_full: bool,
        residual_partial: bool,
        tail_tokens: int,
        aborted_residual: int,
        ttft_ms: float,
    ) -> None:
        self.commits += 1
        if known_hit:
            self.known_suffix_hits += 1
        else:
            self.known_suffix_misses += 1
        if residual_full:
            self.residual_full_hits += 1
        if residual_partial:
            self.residual_partial_hits += 1
        self.commit_tail_tokens += tail_tokens
        self.aborted_residual_tokens += aborted_residual
        self.ttft_ms.append(ttft_ms)

    @property
    def known_suffix_hit_rate(self) -> float:
        n = self.known_suffix_hits + self.known_suffix_misses
        return self.known_suffix_hits / n if n else 0.0

    @property
    def residual_hit_rate(self) -> float:
        n = self.commits
        return self.residual_full_hits / n if n else 0.0

    @property
    def spec_over_committed(self) -> float:
        return self.spec_tokens / self.committed_tokens if self.committed_tokens else 0.0

    def summary(self) -> dict[str, float]:
        return {
            "forks": float(self.forks),
            "commits": float(self.commits),
            "known_suffix_hit_rate": self.known_suffix_hit_rate,
            "residual_hit_rate": self.residual_hit_rate,
            "residual_partial_rate": (self.residual_partial_hits / self.commits) if self.commits else 0.0,
            "median_aborted_residual": float(self.aborted_residual_tokens / max(self.aborts, 1)),
            "spec_over_committed": self.spec_over_committed,
            "median_ttft_ms": _median(self.ttft_ms),
            "p95_tbt_ms": _percentile(self.tbt_ms, 0.95),
        }


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else 0.5 * (s[mid - 1] + s[mid])


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(q * (len(s) - 1))))
    return s[i]
