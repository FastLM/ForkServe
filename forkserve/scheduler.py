"""Two-class continuous batcher (§7.1). Speculation is slack, not load.

Committed jobs consume token budget first until their JIT margin is met.
Speculative chunks fill the remainder, are preemptible at chunk boundaries,
and are cancelled the instant a commit needs the same SM / generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Iterable

from forkserve.config import ForkServeConfig
from forkserve.planner import PrefillChunk
from forkserve.types import JobClass, NodeId, SessionId


@dataclass(slots=True)
class CommittedJob:
    session: SessionId
    node_id: NodeId
    tokens: int
    kind: str  # "decode" | "commit_tail" | "root_prefill"
    slo_tokens_per_s: float
    attained_s: float = 0.0  # PLAS aging term
    arrived_at: float = field(default_factory=monotonic)
    tenant: str = "default"


@dataclass(slots=True)
class BatchPlan:
    tick: int
    dt_ms: float
    B_t: int
    B_c: int
    B_s: int
    committed: list[CommittedJob] = field(default_factory=list)
    speculative: list[PrefillChunk] = field(default_factory=list)
    cancelled: list[PrefillChunk] = field(default_factory=list)
    saturated: bool = False


@dataclass(slots=True)
class FairMeter:
    """VTC-style second meter (§7.4): spec billed at κ, TBT-miss victims at 1."""

    committed: float = 0.0
    speculative: float = 0.0
    victim: float = 0.0


class TwoClassScheduler:
    def __init__(self, config: ForkServeConfig) -> None:
        self.cfg = config
        self.tick = 0
        self.meters: dict[str, FairMeter] = {}
        self._waiting_c: list[CommittedJob] = []
        self._waiting_s: list[PrefillChunk] = []
        self._stale_gen: dict[NodeId, int] = {}

    def submit_committed(self, job: CommittedJob) -> None:
        self._waiting_c.append(job)

    def submit_speculative(self, chunks: Iterable[PrefillChunk]) -> None:
        self._waiting_s.extend(chunks)
        self._waiting_s.sort(key=lambda c: (c.gain / c.cost) if c.cost else 0.0, reverse=True)

    def invalidate(self, parent: NodeId, generation: int) -> int:
        """O(1) generation drop of siblings whose parent just committed."""
        self._stale_gen[parent] = generation
        drop = [c for c in self._waiting_s if c.parent == parent and c.generation != generation]
        self._waiting_s = [c for c in self._waiting_s if c not in drop]
        return len(drop)

    def cancel_node(self, node_id: NodeId) -> int:
        n = len([c for c in self._waiting_s if c.node_id == node_id])
        self._waiting_s = [c for c in self._waiting_s if c.node_id != node_id]
        return n

    def schedule(self, *, leftover_hint: int | None = None) -> BatchPlan:
        """Realize Equations (6) and (7) for one engine tick."""
        self.tick += 1
        dt = self.cfg.tick_ms
        B_t = leftover_hint if leftover_hint is not None else self.cfg.max_batched_tokens

        # Committed demand first (program-FCFS + PLAS). Eq. (6)/(7): B^s is
        # whatever remains after committed work takes its tokens this tick.
        self._waiting_c.sort(key=lambda j: (j.attained_s, j.arrived_at))
        need = 0
        chosen_c: list[CommittedJob] = []
        remain = list(self._waiting_c)
        for job in remain:
            take = min(job.tokens, B_t - need)
            if take <= 0:
                break
            need += take
            job.tokens -= take
            chosen_c.append(job)
            self._bill(job.tenant, JobClass.COMMITTED, take)
            if job.tokens <= 0:
                self._waiting_c.remove(job)
        B_c = min(B_t, need)
        B_s = max(0, B_t - B_c)

        chosen_s: list[PrefillChunk] = []
        cancelled: list[PrefillChunk] = []
        used_s = 0
        if B_s > 0:
            still: list[PrefillChunk] = []
            for chunk in self._waiting_s:
                stale = self._stale_gen.get(chunk.parent)
                if stale is not None and chunk.generation != stale:
                    cancelled.append(chunk)
                    continue
                n = len(chunk.tokens)
                if used_s + n > B_s:
                    still.append(chunk)
                    continue
                # Cap a single leftover swallow.
                if n > self.cfg.c_spec:
                    still.append(chunk)
                    continue
                chosen_s.append(chunk)
                used_s += n
                self._bill("default", JobClass.SPECULATIVE, n)
            self._waiting_s = still
            B_s = used_s
        else:
            cancelled.extend(self._waiting_s)
            # Do not drop the queue — just admit nothing this tick (principle 4).

        return BatchPlan(
            tick=self.tick,
            dt_ms=dt,
            B_t=B_t,
            B_c=B_c,
            B_s=B_s,
            committed=chosen_c,
            speculative=chosen_s,
            cancelled=cancelled,
            saturated=B_s == 0 and B_c >= B_t,
        )

    def note_tbt_miss(self, tenant: str, tokens: int) -> None:
        meter = self.meters.setdefault(tenant, FairMeter())
        meter.victim += tokens

    def pending_spec_tokens(self) -> int:
        return sum(len(c.tokens) for c in self._waiting_s)

    def pending_committed_tokens(self) -> int:
        return sum(j.tokens for j in self._waiting_c)

    def _bill(self, tenant: str, cls: JobClass, tokens: int) -> None:
        if not self.cfg.vtc_enabled:
            return
        meter = self.meters.setdefault(tenant, FairMeter())
        if cls is JobClass.COMMITTED:
            meter.committed += tokens
        else:
            meter.speculative += tokens * self.cfg.spec_bill_kappa
