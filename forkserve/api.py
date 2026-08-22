"""Harness verbs: open / fork / speculate / commit / join / abort / close.

``commit`` is the only verb on the critical path of user-visible tokens.
Everything else is asynchronous. Speculation never enters the sampler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Iterable
from uuid import uuid4

from forkserve.commit import CommitProtocol, CommitResult
from forkserve.config import ForkServeConfig
from forkserve.engine.protocol import DecodeRequest, EngineBackend, PrefillRequest
from forkserve.join import JoinExecutor, JoinResult
from forkserve.metrics import SessionMetrics
from forkserve.pages import PagePool
from forkserve.planner import (
    Candidate,
    CountMinPrior,
    NGramResidual,
    PrefillChunk,
    SpeculatePlanner,
)
from forkserve.retention import RetentionManager
from forkserve.router import TreeStickyRouter
from forkserve.scheduler import CommittedJob, TwoClassScheduler
from forkserve.tree import ContextTree, Forest, InvariantError
from forkserve.types import (
    BranchId,
    JoinPolicy,
    NodeId,
    NodeMode,
    SchemaKind,
    SessionId,
    TokenSeq,
    WorkKind,
    as_tokens,
)


@dataclass(slots=True)
class SessionHandle:
    id: SessionId
    root: NodeId
    tip: NodeId
    tenant: str


@dataclass
class Engine:
    backend: EngineBackend
    config: ForkServeConfig = field(default_factory=ForkServeConfig)
    forest: Forest = field(init=False)
    planner: SpeculatePlanner = field(init=False)
    scheduler: TwoClassScheduler = field(init=False)
    committer: CommitProtocol = field(init=False)
    joiner: JoinExecutor = field(init=False)
    retention: RetentionManager = field(init=False)
    router: TreeStickyRouter = field(init=False)
    priors: CountMinPrior = field(init=False)
    ngrams: NGramResidual = field(init=False)
    metrics: dict[SessionId, SessionMetrics] = field(default_factory=dict)
    _pending_spec: dict[SessionId, list[PrefillChunk]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        pool: PagePool = self.backend.pool
        self.forest = Forest(pool, self.config)
        self.planner = SpeculatePlanner(self.config)
        self.scheduler = TwoClassScheduler(self.config)
        self.committer = CommitProtocol()
        self.joiner = JoinExecutor()
        self.retention = RetentionManager(self.forest, self.config)
        self.router = TreeStickyRouter(self.config)
        self.priors = CountMinPrior(
            self.config.prior_width, self.config.prior_depth, self.config.prior_decay
        )
        self.ngrams = NGramResidual(self.config.ngram_order, self.config.ngram_prefix)

    # ----- verbs -------------------------------------------------------------

    def open(
        self,
        tokens: TokenSeq | str,
        *,
        session: str | None = None,
        tenant: str = "default",
        prefill: bool = True,
    ) -> SessionHandle:
        sid = SessionId(session or f"s-{uuid4().hex[:10]}")
        tok = self._tok(tokens)
        tree = self.forest.create(sid, tenant=tenant, worker=int(self.router.pin_root(sid)))
        root = tree.open_root(tok)
        self.metrics[sid] = SessionMetrics()
        if prefill:
            self.backend.prefill(
                PrefillRequest(sid, root.id, tok, speculative=False, page_ids=())
            )
            self.metrics[sid].committed_tokens += len(tok)
        return SessionHandle(id=sid, root=root.id, tip=root.id, tenant=tenant)

    def fork(
        self,
        session: SessionId | str,
        parent: NodeId,
        bid: str,
        known_suffix: TokenSeq | str = (),
        *,
        speculate: bool = False,
        priority: float = 1.0,
    ) -> NodeId:
        tree = self.forest.get(SessionId(str(session)))
        known = self._tok(known_suffix)
        # Security: wrappers may carry placeholders; args are a later residual.
        node = tree.fork(parent, BranchId(bid), known)
        self.metrics[tree.session].forks += 1
        place = self.router.place_fork(tree.session, len(known), speculative=True)
        node.worker = int(place.worker)
        if speculate:
            self.speculate(tree.session, node.id, priority=priority)
        return node.id

    def speculate(
        self,
        session: SessionId | str,
        node: NodeId,
        *,
        priority: float = 1.0,
        residual_hat: TokenSeq | str = (),
        q_b: float = 0.0,
        schema: SchemaKind = SchemaKind.FREEFORM,
        t_idle_ms: float = 1e9,
    ) -> list[PrefillChunk]:
        tree = self.forest.get(SessionId(str(session)))
        child = tree.get(node)
        if child.parent is None:
            raise InvariantError("cannot speculate the root")
        parent = tree.get(child.parent)
        cand = Candidate(
            branch_id=child.branch_id,
            node_id=child.id,
            known=child.residual,
            residual_hat=self._tok(residual_hat),
            p_b=max(priority, 0.0),
            q_b=q_b,
            schema=schema,
            declared=True,
        )
        gamma = self.backend.tbt_headroom_ms()
        plan = self.planner.allocate(
            tree.session,
            parent.id,
            [cand],
            t_idle_ms=t_idle_ms,
            gamma_ms=gamma,
            m_free=self.backend.free_hbm_bytes(),
            parent_mode=parent.mode,
            generation=parent.generation,
        )
        if plan.chunks:
            self.scheduler.submit_speculative(plan.chunks)
            self._pending_spec.setdefault(tree.session, []).extend(plan.chunks)
            self.metrics[tree.session].speculates += 1
        return plan.chunks

    def speculate_set(
        self,
        session: SessionId | str,
        parent: NodeId,
        candidates: Iterable[Candidate],
        *,
        t_idle_ms: float,
    ) -> list[PrefillChunk]:
        tree = self.forest.get(SessionId(str(session)))
        pnode = tree.get(parent)
        plan = self.planner.allocate(
            tree.session,
            parent,
            list(candidates),
            t_idle_ms=t_idle_ms,
            gamma_ms=self.backend.tbt_headroom_ms(),
            m_free=self.backend.free_hbm_bytes(),
            parent_mode=pnode.mode,
            generation=pnode.generation,
        )
        if plan.chunks:
            self.scheduler.submit_speculative(plan.chunks)
            self._pending_spec.setdefault(tree.session, []).extend(plan.chunks)
            self.metrics[tree.session].speculates += 1
        return plan.chunks

    def commit(
        self,
        session: SessionId | str,
        parent: NodeId,
        prompt: TokenSeq | str,
        *,
        preferred_bid: str | None = None,
    ) -> CommitResult:
        t0 = monotonic()
        tree = self.forest.get(SessionId(str(session)))
        tok = self._tok(prompt)
        # Cancel in-flight speculative siblings before mutating the tree.
        result = self.committer.apply(tree, parent, tok, preferred_bid=preferred_bid)
        n_cancel = self.scheduler.invalidate(parent, result.generation)
        self.metrics[tree.session].cancelled_chunks += n_cancel
        aborted_res = 0
        for aid in result.aborted:
            # already dead; residual length recorded on counters
            aborted_res += tree.get(aid).counters.residual_tokens
            self.metrics[tree.session].aborts += 1

        if result.tail:
            self.scheduler.submit_committed(
                CommittedJob(
                    session=tree.session,
                    node_id=result.winner,
                    tokens=len(result.tail),
                    kind="commit_tail",
                    slo_tokens_per_s=len(result.tail) / max(self.config.ttft_slo_ms / 1000.0, 1e-3),
                    tenant=tree.tenant,
                )
            )
            self.backend.prefill(
                PrefillRequest(
                    tree.session, result.winner, result.tail, speculative=False, page_ids=()
                )
            )
            self.metrics[tree.session].committed_tokens += len(result.tail)

        ttft = (monotonic() - t0) * 1000.0
        # Plus residual prefill time already paid on the critical path only.
        if result.tail:
            ttft += self.config.prefill_ms(len(result.tail))
        self.metrics[tree.session].record_commit(
            known_hit=result.known_suffix_hit,
            residual_full=result.residual_full_hit,
            residual_partial=result.residual_partial_hit,
            tail_tokens=len(result.tail),
            aborted_residual=aborted_res,
            ttft_ms=ttft,
        )
        tree.set_mode(result.winner, NodeMode.COMMIT)
        return result

    def join(
        self,
        session: SessionId | str,
        children: list[NodeId],
        policy: JoinPolicy,
        *,
        scaffold: TokenSeq | str = (),
        blend: TokenSeq | str = (),
        k: int = 1,
        parent: NodeId | None = None,
    ) -> JoinResult:
        tree = self.forest.get(SessionId(str(session)))
        result = self.joiner.apply(
            tree,
            children,
            policy,
            scaffold=self._tok(scaffold),
            blend=self._tok(blend),
            k=k,
            parent=parent,
        )
        if result.tail:
            self.backend.prefill(
                PrefillRequest(
                    tree.session, result.node_id, result.tail, speculative=False, page_ids=()
                )
            )
        self.metrics[tree.session].joins += 1
        return result

    def abort(self, session: SessionId | str, node: NodeId) -> int:
        tree = self.forest.get(SessionId(str(session)))
        n = tree.abort(node)
        self.scheduler.cancel_node(node)
        self.metrics[tree.session].aborts += 1
        return n

    def close(self, session: SessionId | str) -> None:
        sid = SessionId(str(session))
        self.forest.close(sid)
        self.router.close(sid)
        self._pending_spec.pop(sid, None)

    # ----- decode / slack ----------------------------------------------------

    def generate(
        self,
        session: SessionId | str,
        n_tokens: int,
        *,
        seed: int | None = None,
    ) -> TokenSeq:
        """Committed decode only. Speculative nodes never enter the sampler."""
        tree = self.forest.get(SessionId(str(session)))
        if tree.tip is None:
            raise InvariantError("empty session")
        tip = tree.get(tree.tip)
        if tip.mode is NodeMode.SPEC:
            raise InvariantError("refuse to sample a speculative node")
        # Drain leftover budget into speculative chunks (decode slack, Insight 3).
        self.drain_slack()
        req = DecodeRequest(tree.session, tip.id, n_tokens, seed=seed)
        out = self.backend.decode(req)
        if out:
            tree.append_tokens(tip.id, tuple(out), committed=True)
            self.metrics[tree.session].committed_tokens += len(out)
        return tuple(out)

    def drain_slack(self) -> int:
        """Run leftover token budget as speculative prefills. No-op if saturated."""
        plan = self.scheduler.schedule()
        ran = 0
        for chunk in plan.speculative:
            if chunk.node_id is None:
                continue
            tree = self.forest.get(chunk.session)
            node = tree.get(chunk.node_id)
            if node.mode is NodeMode.DEAD:
                continue
            # Append only the tokens not already in the residual (known suffix
            # was materialized at fork; extra residual_hat still needs writes).
            extra = chunk.tokens
            if chunk.kind is WorkKind.OBS_RESIDUAL:
                tree.append_tokens(node.id, extra, committed=False)
            self.backend.prefill(
                PrefillRequest(
                    chunk.session, node.id, extra, speculative=True, page_ids=()
                )
            )
            self.metrics[chunk.session].spec_tokens += len(extra)
            ran += len(extra)
        return ran

    def mark_tool_idle(self, session: SessionId | str, node: NodeId, tool_s: float) -> float:
        tree = self.forest.get(SessionId(str(session)))
        return self.retention.arm_idle(tree, node, tool_s=tool_s)

    def evict(self) -> None:
        self.retention.expire()
        self.retention.plan_offload()

    # ----- accessors ---------------------------------------------------------

    def tree(self, session: SessionId | str) -> ContextTree:
        return self.forest.get(SessionId(str(session)))

    def _tok(self, value: TokenSeq | str | None) -> TokenSeq:
        if value is None:
            return ()
        if isinstance(value, str):
            return self.backend.tokenize(value)
        return as_tokens(value)
