"""vLLM engine-loop hooks: CoW bit on the block pool + two-class schedule.

Installed at process start (see ``install_vllm_cow``). Does not vendor vLLM;
we subclass the public ``scheduler_cls`` hook and wrap ``KVCacheManager`` /
``BlockPool`` methods.

CoW vs prefix cache
-------------------
Radix / APC share a block *after* two requests hash to the same tokens.
Fork aliases the parent's live block table *before* the child's residual
exists: ``touch`` + pin ``ro``. A write never mutates a shared/ro trunk
block; ``allocate_slots`` appends a private residual. Abort ``free``s the
child; the trunk stays while any sibling still holds a ref.

Two-class schedule
------------------
``schedule()`` reorders ``running`` / ``waiting`` so committed requests
take the token budget first. Speculative work (``extra_args['forkserve_class']
== 'speculative'``) fills the leftover. Under saturation they are not
scheduled — principle 4, inside the engine loop, not after it.
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


@dataclass
class CowBlockTable:
    """Physical CoW metadata living beside vLLM's ``BlockPool``.

    ``ro`` is the paper's CoW bit: shared trunk pages are pinned read-only.
    """

    ro: set[int] = field(default_factory=set)
    node_to_req: dict[int, str] = field(default_factory=dict)
    forks: int = 0
    alias_blocks: int = 0
    cow_copies: int = 0

    def pin_ro(self, block_ids: Iterable[int]) -> None:
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

    def _init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        _orig_init(self, *args, **kwargs)
        self._forkserve_cow = CowBlockTable()

    def _touch(self, blocks):  # type: ignore[no-untyped-def]
        _orig_touch(self, blocks)
        table = cow_of(self)
        for block in blocks:
            if getattr(block, "is_null", False):
                continue
            if block.ref_cnt > 1:
                table.pin_ro(block.block_id)

    def _get_computed(self, request):  # type: ignore[no-untyped-def]
        aliased = _alias_parent_blocks(self, request)
        if aliased is not None:
            return aliased
        return _orig_get_computed(self, request)

    BlockPool.__init__ = _init  # type: ignore[method-assign]
    BlockPool.touch = _touch  # type: ignore[method-assign]
    KVCacheManager.get_computed_blocks = _get_computed  # type: ignore[method-assign]
    _INSTALLED = True


def _alias_parent_blocks(mgr: Any, request: Any) -> tuple[Any, int] | None:
    """O(1) fork: reuse the parent's live block table instead of hashing."""
    extra = extra_of(request)
    parent_node = extra.get("forkserve_parent_node")
    if parent_node is None:
        return None
    pool = getattr(mgr, "block_pool", None)
    if pool is None:
        return None
    table = cow_of(pool)
    parent_req = table.req_for_node(int(parent_node))
    if not parent_req:
        return None
    coord = getattr(mgr, "coordinator", None)
    if coord is None:
        return None
    groups: list[list[Any]] = []
    n_tokens = 0
    for stm in coord.single_type_managers:
        blocks = list(stm.req_to_blocks.get(parent_req, ()))
        if not blocks:
            return None
        # Drop the possibly-partial tail so the child never writes the trunk.
        full = blocks[:-1] if len(blocks) > 1 else []
        groups.append(full)
        n_tokens = max(n_tokens, len(full) * int(stm.block_size))
    if n_tokens <= 0:
        return None
    n_tokens = min(n_tokens, max(int(request.num_tokens) - 1, 0))
    ids = [b.block_id for g in groups for b in g]
    table.pin_ro(ids)
    table.note_fork(len(ids))
    return mgr.create_kv_cache_blocks(tuple(groups)), n_tokens


def TwoClassVllmScheduler(*args: Any, **kwargs: Any):  # noqa: N802
    """Factory so ``scheduler_cls`` can be a dotted path or this callable.

    vLLM wants a class; we expose the real subclass below as the attribute
    ``TwoClassVllmScheduler`` after first import. This wrapper keeps import
    of vLLM lazy for unit tests of ``committed_first``.
    """
    cls = get_two_class_scheduler()
    return cls(*args, **kwargs)


_SCHED_CLS: type | None = None


def get_two_class_scheduler() -> type:
    global _SCHED_CLS
    if _SCHED_CLS is not None:
        return _SCHED_CLS

    from vllm.v1.core.sched.request_queue import FCFSRequestQueue
    from vllm.v1.core.sched.scheduler import Scheduler

    class _TwoClassVllmScheduler(Scheduler):
        """Committed ≻ spec inside vLLM's iteration-level ``schedule()``."""

        def add_request(self, request):  # type: ignore[no-untyped-def]
            extra = extra_of(request)
            node = extra.get("forkserve_node")
            pool = getattr(self.kv_cache_manager, "block_pool", None)
            if pool is not None and node is not None:
                cow_of(pool).bind_node(int(node), request.request_id)
            return super().add_request(request)

        def schedule(self, throttle_prefills: bool = False):  # type: ignore[no-untyped-def]
            self.running = committed_first(self.running)
            waiting = self.waiting
            if isinstance(waiting, FCFSRequestQueue) and waiting:
                ordered = committed_first(list(waiting))
                waiting.clear()
                waiting.extend(ordered)
            skipped = getattr(self, "skipped_waiting", None)
            if isinstance(skipped, FCFSRequestQueue) and skipped:
                ordered = committed_first(list(skipped))
                skipped.clear()
                skipped.extend(ordered)
            return super().schedule(throttle_prefills)

    _TwoClassVllmScheduler.__name__ = "TwoClassVllmScheduler"
    _TwoClassVllmScheduler.__qualname__ = "TwoClassVllmScheduler"
    _SCHED_CLS = _TwoClassVllmScheduler
    return _TwoClassVllmScheduler
