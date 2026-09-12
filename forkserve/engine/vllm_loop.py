"""vLLM engine-loop hooks: CoW bit on the block pool + two-class schedule.

Installed at process start (see ``install_vllm_cow``). Does not vendor vLLM;
we subclass the public ``scheduler_cls`` hook and wrap ``KVCacheManager`` /
``BlockPool`` methods.

CoW vs prefix cache
-------------------
Radix / APC share a block *after* two requests hash to the same tokens.
Fork aliases the parent's snapshotted block table *before* the child's residual
exists: ``touch`` + pin ``ro``. A write never mutates a shared/ro trunk
block; ``allocate_slots`` appends a private residual. Abort ``free``s the
child; the trunk stays while any sibling still holds a ref.

``LLM.generate`` retires the parent request, so live ``req_to_blocks`` is
gone by the time children arrive. We snapshot + extra-pin on
``cache_blocks`` so alias stays O(1) across generate() calls.

Two-class schedule
------------------
Subclass ``AsyncScheduler`` (not ``Scheduler``) so vLLM keeps async
scheduling. ``schedule()`` reorders ``running`` / ``waiting`` so committed
work takes the token budget first. Speculative work
(``extra_args['forkserve_class'] == 'speculative'``) fills the leftover.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

FS_COMMITTED = "committed"
FS_SPECULATIVE = "speculative"

_INSTALLED = False


def extra_of(request: Any) -> dict[str, Any]:
    sp = getattr(request, "sampling_params", None)
    if sp is None:
        return {}
    return dict(getattr(sp, "extra_args", None) or {})


def is_speculative(request: Any) -> bool:
    return extra_of(request).get("forkserve_class") == FS_SPECULATIVE


def committed_first(requests: Sequence[Any]) -> list[Any]:
    """Stable partition: committed, then speculative. Used by the engine loop."""
    c = [r for r in requests if not is_speculative(r)]
    s = [r for r in requests if is_speculative(r)]
    return c + s


def forkserve_extra(
    *,
    speculative: bool,
    node_id: int | None = None,
    parent_node: int | None = None,
    generation: int = 0,
) -> dict[str, Any]:
    extra: dict[str, Any] = {
        "forkserve_class": FS_SPECULATIVE if speculative else FS_COMMITTED,
        "forkserve_generation": int(generation),
    }
    if node_id is not None:
        extra["forkserve_node"] = int(node_id)
    if parent_node is not None:
        extra["forkserve_parent_node"] = int(parent_node)
    return extra


def select_full_blocks(
    blocks: Sequence[Any],
    block_size: int,
    parent_tokens: int,
) -> tuple[list[Any], int]:
    """Keep only complete pages. Partial tail stays with the parent.

    Previous code dropped ``blocks[-1]`` even when that page was full, forcing
    the child to recompute up to ``block_size`` trunk tokens.
    """
    if block_size <= 0 or parent_tokens <= 0 or not blocks:
        return [], 0
    n_full = parent_tokens // block_size
    if n_full <= 0:
        return [], 0
    use = list(blocks[:n_full])
    return use, n_full * block_size


@dataclass
class NodeBlockSnap:
    groups: list[list[Any]]
    num_tokens: int
    block_size: int
    pinned: bool = False


@dataclass
class CowBlockTable:
    """Physical CoW metadata living beside vLLM's ``BlockPool``.

    ``ro`` is the paper's CoW bit: shared trunk pages are pinned read-only.
    """

    ro: set[int] = field(default_factory=set)
    node_to_req: dict[int, str] = field(default_factory=dict)
    node_snap: dict[int, NodeBlockSnap] = field(default_factory=dict)
    forks: int = 0
    alias_blocks: int = 0
    cow_copies: int = 0

    def pin_ro(self, block_ids: Iterable[int] | int) -> None:
        if isinstance(block_ids, int):
            self.ro.add(int(block_ids))
            return
        self.ro.update(int(i) for i in block_ids)

    def writable(self, block_id: int, ref_cnt: int) -> bool:
        return ref_cnt <= 1 and int(block_id) not in self.ro

    def bind_node(self, node_id: int, request_id: str) -> None:
        self.node_to_req[int(node_id)] = request_id

    def req_for_node(self, node_id: int) -> str | None:
        return self.node_to_req.get(int(node_id))

    def note_fork(self, n_blocks: int) -> None:
        self.forks += 1
        self.alias_blocks += n_blocks

    def note_cow_copy(self) -> None:
        self.cow_copies += 1

    def store_snapshot(
        self,
        node_id: int,
        groups: list[list[Any]],
        num_tokens: int,
        block_size: int,
    ) -> NodeBlockSnap:
        snap = NodeBlockSnap(
            groups=groups,
            num_tokens=int(num_tokens),
            block_size=int(block_size),
            pinned=self.node_snap.get(int(node_id), NodeBlockSnap([], 0, 0)).pinned,
        )
        self.node_snap[int(node_id)] = snap
        return snap


def cow_of(block_pool: Any) -> CowBlockTable:
    table = getattr(block_pool, "_forkserve_cow", None)
    if table is None:
        table = CowBlockTable()
        block_pool._forkserve_cow = table
    return table


def install_vllm_cow() -> None:
    """Idempotent monkey-patch of vLLM V1 BlockPool + KVCacheManager."""
    global _INSTALLED
    if _INSTALLED:
        return
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    _orig_init = BlockPool.__init__
    _orig_touch = BlockPool.touch
    _orig_get_computed = KVCacheManager.get_computed_blocks
    _orig_cache_blocks = KVCacheManager.cache_blocks

    def _init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        _orig_init(self, *args, **kwargs)
        self._forkserve_cow = CowBlockTable()

    def _touch(self, blocks):  # type: ignore[no-untyped-def]
        _orig_touch(self, blocks)
        table = cow_of(self)
        pin = table.ro.add
        for block in blocks:
            if getattr(block, "is_null", False):
                continue
            if block.ref_cnt > 1:
                pin(int(block.block_id))

    def _get_computed(self, request):  # type: ignore[no-untyped-def]
        aliased = _alias_parent_blocks(self, request)
        if aliased is not None:
            return aliased
        return _orig_get_computed(self, request)

    def _cache_blocks(self, request, num_computed_tokens):  # type: ignore[no-untyped-def]
        _orig_cache_blocks(self, request, num_computed_tokens)
        extra = extra_of(request)
        if extra.get("forkserve_node") is None:
            return
        try:
            _snapshot_node_blocks(self, request)
        except Exception:
            # CoW snapshot must never kill the engine core.
            return

    BlockPool.__init__ = _init  # type: ignore[method-assign]
    BlockPool.touch = _touch  # type: ignore[method-assign]
    KVCacheManager.get_computed_blocks = _get_computed  # type: ignore[method-assign]
    KVCacheManager.cache_blocks = _cache_blocks  # type: ignore[method-assign]
    _INSTALLED = True


def _snapshot_node_blocks(mgr: Any, request: Any) -> None:
    extra = extra_of(request)
    node = extra.get("forkserve_node")
    if node is None:
        return
    pool = getattr(mgr, "block_pool", None)
    coord = getattr(mgr, "coordinator", None)
    if pool is None or coord is None:
        return
    table = cow_of(pool)
    table.bind_node(int(node), request.request_id)
    groups: list[list[Any]] = []
    block_size = 16
    for stm in coord.single_type_managers:
        groups.append(list(stm.req_to_blocks.get(request.request_id, ())))
        block_size = int(getattr(stm, "block_size", block_size))
    # Prompt only — a prefill ``max_tokens=1`` sample must not enter the CoW alias.
    num_tokens = int(
        getattr(request, "num_prompt_tokens", 0)
        or getattr(request, "num_tokens", 0)
    )
    prev = table.node_snap.get(int(node))
    snap = table.store_snapshot(int(node), groups, num_tokens, block_size)
    if prev is not None and prev.pinned:
        return
    flat = [
        b
        for g in groups
        for b in g
        if b is not None and not getattr(b, "is_null", False)
    ]
    if not flat:
        return
    # Extra ref so ``LLM.generate`` retiring the parent does not free the trunk.
    pool.touch(flat)
    snap.pinned = True
    table.pin_ro(b.block_id for b in flat)


def _groups_from_live(coord: Any, parent_req: str) -> tuple[list[list[Any]], int, int]:
    groups: list[list[Any]] = []
    block_size = 16
    for stm in coord.single_type_managers:
        blocks = list(stm.req_to_blocks.get(parent_req, ()))
        if not blocks:
            return [], 0, block_size
        groups.append(blocks)
        block_size = int(getattr(stm, "block_size", block_size))
    n_est = max((len(g) * block_size for g in groups), default=0)
    return groups, n_est, block_size


def _alias_parent_blocks(mgr: Any, request: Any) -> tuple[Any, int] | None:
    """O(1) fork: reuse the parent's snapshotted (or live) block table."""
    extra = extra_of(request)
    parent_node = extra.get("forkserve_parent_node")
    if parent_node is None:
        return None
    pool = getattr(mgr, "block_pool", None)
    if pool is None:
        return None
    table = cow_of(pool)
    coord = getattr(mgr, "coordinator", None)
    snap = table.node_snap.get(int(parent_node))
    groups_src: list[list[Any]] = []
    parent_tokens = 0
    block_size = 16
    if snap is not None and snap.groups:
        groups_src = snap.groups
        parent_tokens = snap.num_tokens
        block_size = snap.block_size
    elif coord is not None:
        parent_req = table.req_for_node(int(parent_node))
        if not parent_req:
            return None
        groups_src, parent_tokens, block_size = _groups_from_live(coord, parent_req)
        if not groups_src:
            return None
    else:
        return None

    groups: list[list[Any]] = []
    n_tokens = 0
    for raw in groups_src:
        full, nt = select_full_blocks(raw, block_size, parent_tokens)
        if not full:
            return None
        groups.append(full)
        n_tokens = max(n_tokens, nt)
    if n_tokens <= 0:
        return None
    n_tokens = min(n_tokens, max(int(getattr(request, "num_tokens", 1)) - 1, 0))
    ids = [b.block_id for g in groups for b in g]
    table.pin_ro(ids)
    table.note_fork(len(ids))
    return mgr.create_kv_cache_blocks(tuple(groups)), n_tokens


def TwoClassVllmScheduler(*args: Any, **kwargs: Any):  # noqa: N802
    """Factory so ``scheduler_cls`` can be a dotted path or this callable."""
    cls = get_two_class_scheduler()
    return cls(*args, **kwargs)


_SCHED_CLS: type | None = None


def get_two_class_scheduler() -> type:
    global _SCHED_CLS
    if _SCHED_CLS is not None:
        return _SCHED_CLS

    from vllm.v1.core.sched.request_queue import FCFSRequestQueue

    try:
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler as _Base
    except ImportError:  # pragma: no cover - older vLLM
        from vllm.v1.core.sched.scheduler import Scheduler as _Base

    class _TwoClassVllmScheduler(_Base):
        """Committed ≻ spec inside vLLM's iteration-level ``schedule()``."""

        def add_request(self, request):  # type: ignore[no-untyped-def]
            extra = extra_of(request)
            node = extra.get("forkserve_node")
            pool = getattr(self.kv_cache_manager, "block_pool", None)
            if pool is not None and node is not None:
                cow_of(pool).bind_node(int(node), request.request_id)
            return super().add_request(request)

        def schedule(self, throttle_prefills: bool = False):  # type: ignore[no-untyped-def]
            # Skip the partition when the batch is uniform (common decode).
            running = self.running
            if running and any(is_speculative(r) for r in running):
                self.running = committed_first(running)
            waiting = self.waiting
            if isinstance(waiting, FCFSRequestQueue) and waiting:
                raw = list(waiting)
                if any(is_speculative(r) for r in raw):
                    waiting.clear()
                    waiting.extend(committed_first(raw))
            skipped = getattr(self, "skipped_waiting", None)
            if isinstance(skipped, FCFSRequestQueue) and skipped:
                raw = list(skipped)
                if any(is_speculative(r) for r in raw):
                    skipped.clear()
                    skipped.extend(committed_first(raw))
            return super().schedule(throttle_prefills)

    _TwoClassVllmScheduler.__name__ = "TwoClassVllmScheduler"
    _TwoClassVllmScheduler.__qualname__ = "TwoClassVllmScheduler"
    _SCHED_CLS = _TwoClassVllmScheduler
    return _TwoClassVllmScheduler
