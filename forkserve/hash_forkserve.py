"""HashForkServe: content-addressed prefix blocks + ForkServe CoW branches.

vLLM Automatic Prefix Caching (APC) indexes *full* KV blocks by
``hash(parent_hash, block_tokens, extra)``.  It discovers sharing *after*
tokens exist and across unrelated requests.

ForkServe forks a context tree *before* residual tokens exist, so siblings
inherit the trunk by reference (CoW) and can be speculatively prefilled.

HashForkServe keeps both:

* **Hash index** — cross-session / opportunistic reuse of committed blocks
  (APC strength).
* **Fork / CoW tree** — same-session branch sharing and speculative leaves
  (ForkServe strength).
* **Unified page** — one physical page may be reachable both by a node
  page-table and by a content hash.  Eviction prefers speculative leaves,
  then LRU among hash-cached free blocks (APC free-queue order).

Only *full* pages enter the hash index (APC note 1).  Speculative pages are
never hashed until LCP commit promotes them.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from enum import Enum
from time import monotonic
from typing import Iterable

from forkserve.config import ForkServeConfig
from forkserve.pages import PagePool, PhysicalPage, TokenKvStore, pages_for_tokens
from forkserve.types import NodeId, PageId, TokenSeq, as_tokens


class PageKind(str, Enum):
    COMMITTED = "committed"
    SPECULATIVE = "speculative"


@dataclass(frozen=True, slots=True)
class BlockHash:
    """Content address of a full KV page. Matches APC tuple components."""

    digest: bytes

    def __str__(self) -> str:
        return self.digest.hex()[:16]


def hash_block(
    parent: BlockHash | None,
    tokens: TokenSeq,
    *,
    extra: bytes = b"",
    algo: str = "sha256",
) -> BlockHash:
    """``hash(parent_hash, block_tokens, extra)`` — APC block key."""
    h = hashlib.new(algo)
    if parent is None:
        h.update(b"\x00")
    else:
        h.update(b"\x01")
        h.update(parent.digest)
    h.update(struct.pack("<I", len(tokens)))
    for t in tokens:
        h.update(struct.pack("<i", int(t)))
    h.update(struct.pack("<I", len(extra)))
    h.update(extra)
    return BlockHash(h.digest())


@dataclass(slots=True)
class HashPageMeta:
    page_id: PageId
    block_hash: BlockHash | None
    kind: PageKind
    last_touch: float
    n_tokens: int  # valid tokens when hashed (must equal page_size)


@dataclass(slots=True)
class LookupHit:
    """Result of hashing a token prefix against the APC index."""

    matched_pages: list[PageId]
    matched_tokens: int
    hashes: list[BlockHash]
    miss_from: int  # token index where miss begins


@dataclass(slots=True)
class HybridStats:
    hash_hits: int = 0
    hash_misses: int = 0
    fork_aliases: int = 0
    pages_hashed: int = 0
    pages_evicted: int = 0
    duplicate_suppressed: int = 0  # APC v1 duplicate full-block case

    @property
    def hit_rate(self) -> float:
        n = self.hash_hits + self.hash_misses
        return self.hash_hits / n if n else 0.0


class HashPageIndex:
    """APC-style ``cached_blocks: hash -> page_id`` plus free-queue LRU."""

    def __init__(self, pool: PagePool, config: ForkServeConfig, *, algo: str = "sha256") -> None:
        self.pool = pool
        self.cfg = config
        self.algo = algo
        self.cached: dict[BlockHash, PageId] = {}
        self.meta: dict[PageId, HashPageMeta] = {}
        self._free_lru: list[PageId] = []  # head = LRU victim among free cached
        self.stats = HybridStats()

    @property
    def page_size(self) -> int:
        return self.cfg.page_size

    def lookup_prefix(
        self,
        tokens: TokenSeq,
        *,
        extra: bytes = b"",
        cache_salt: bytes = b"",
    ) -> LookupHit:
        """Walk full blocks of ``tokens``; stop at first hash miss."""
        tokens = as_tokens(tokens)
        matched: list[PageId] = []
        hashes: list[BlockHash] = []
        parent: BlockHash | None = None
        i = 0
        salt_extra = cache_salt + extra if cache_salt else extra
        while i + self.page_size <= len(tokens):
            chunk = tokens[i : i + self.page_size]
            # Salt only on the first block (APC cache_salt).
            ex = salt_extra if i == 0 else extra
            bh = hash_block(parent, chunk, extra=ex, algo=self.algo)
            pid = self.cached.get(bh)
            if pid is None or pid not in self.meta:
                self.stats.hash_misses += 1
                return LookupHit(matched, i, hashes, i)
            self._touch(pid)
            matched.append(pid)
            hashes.append(bh)
            parent = bh
            i += self.page_size
            self.stats.hash_hits += 1
        if i < len(tokens):
            # Partial last block is never cached (APC note 1).
            self.stats.hash_misses += 1
        return LookupHit(matched, i, hashes, i)

    def publish_full(
        self,
        page_id: PageId,
        tokens: TokenSeq,
        parent_hash: BlockHash | None,
        *,
        extra: bytes = b"",
        kind: PageKind = PageKind.COMMITTED,
    ) -> BlockHash | None:
        """Insert a *full* committed page into the hash index. Spec pages skip."""
        tokens = as_tokens(tokens)
        if kind is PageKind.SPECULATIVE:
            self.meta[page_id] = HashPageMeta(
                page_id, None, kind, monotonic(), len(tokens)
            )
            return None
        if len(tokens) != self.page_size:
            return None
        bh = hash_block(parent_hash, tokens, extra=extra, algo=self.algo)
        existing = self.cached.get(bh)
        if existing is not None and existing != page_id:
            # APC v1 duplicate: keep the first, caller should prefer it.
            self.stats.duplicate_suppressed += 1
            self.meta[page_id] = HashPageMeta(
                page_id, bh, kind, monotonic(), len(tokens)
            )
            return bh
        self.cached[bh] = page_id
        self.meta[page_id] = HashPageMeta(
            page_id, bh, kind, monotonic(), len(tokens)
        )
        self.stats.pages_hashed += 1
        page = self.pool.get(page_id)
        page.ro = True  # published pages are read-only for CoW safety
        return bh

    def promote_spec_to_committed(
        self,
        page_id: PageId,
        tokens: TokenSeq,
        parent_hash: BlockHash | None,
        *,
        extra: bytes = b"",
    ) -> BlockHash | None:
        """After LCP commit: speculative residual becomes hashable."""
        meta = self.meta.get(page_id)
        if meta is not None:
            meta.kind = PageKind.COMMITTED
        return self.publish_full(
            page_id, tokens, parent_hash, extra=extra, kind=PageKind.COMMITTED
        )

    def touch_computed(self, page_ids: Iterable[PageId]) -> None:
        """APC ``touch``: incref computed hits so they are not evicted."""
        for pid in page_ids:
            self.pool.incref(pid)
            self._touch(pid)
            if pid in self._free_lru:
                self._free_lru.remove(pid)

    def release_to_free(self, page_id: PageId) -> None:
        """On ref→0: append to free LRU tail (reverse-order release is caller's job)."""
        page = self.pool.get(page_id)
        if page.ref > 0:
            return
        meta = self.meta.get(page_id)
        if meta is not None and meta.block_hash is not None and meta.kind is PageKind.COMMITTED:
            if page_id not in self._free_lru:
                self._free_lru.append(page_id)
        else:
            self._evict_hash(page_id)
            if page.ref <= 0 and page.handle is not None:
                # Already freed by pool.decref; nothing else.
                pass

    def alloc_or_evict(self, owner: NodeId, *, tenant: str | None = None) -> PageId:
        """Pop free LRU head; if it was cached, evict its hash first (APC)."""
        while self._free_lru:
            victim = self._free_lru.pop(0)
            page = self.pool.get(victim)
            if page.ref > 0:
                continue
            self._evict_hash(victim)
            # Reuse the physical slot via pool freelist if available.
            break
        return self.pool.alloc_page(owner, tenant=tenant)

    def _evict_hash(self, page_id: PageId) -> None:
        meta = self.meta.pop(page_id, None)
        if meta is None:
            return
        if meta.block_hash is not None and self.cached.get(meta.block_hash) == page_id:
            del self.cached[meta.block_hash]
        self.stats.pages_evicted += 1

    def _touch(self, page_id: PageId) -> None:
        meta = self.meta.get(page_id)
        if meta is not None:
            meta.last_touch = monotonic()
        if page_id in self._free_lru:
            self._free_lru.remove(page_id)
            self._free_lru.append(page_id)


@dataclass(slots=True)
class MaterializeResult:
    page_ids: list[PageId]
    reused_hash_pages: int
    new_pages: int
    matched_tokens: int
    last_hash: BlockHash | None


class HashForkPool:
    """Unified allocator: hash lookup first, then CoW residual allocation.

    Typical paths
    -------------
    * **open / commit (cross-session):** ``materialize_with_hash`` fills as many
      full blocks as APC hits, then allocates the miss tail.
    * **fork (same-session):** alias parent pages by refcount (O(1)); no hash
      walk required because the parent pointer already names the trunk.
    * **after commit:** promote full residual pages into the hash index so the
      next unrelated session can hit them via APC.
    """

    def __init__(
        self,
        config: ForkServeConfig | None = None,
        *,
        algo: str = "sha256",
        store: TokenKvStore | None = None,
    ) -> None:
        self.cfg = config or ForkServeConfig()
        self.pool = PagePool(self.cfg, store or TokenKvStore())
        self.index = HashPageIndex(self.pool, self.cfg, algo=algo)

    @property
    def stats(self) -> HybridStats:
        return self.index.stats

    def materialize_with_hash(
        self,
        owner: NodeId,
        tokens: TokenSeq,
        *,
        extra: bytes = b"",
        cache_salt: bytes = b"",
        tenant: str | None = None,
        publish: bool = True,
    ) -> MaterializeResult:
        """Allocate a page spine for ``tokens``, reusing APC hits."""
        tokens = as_tokens(tokens)
        hit = self.index.lookup_prefix(tokens, extra=extra, cache_salt=cache_salt)
        pages = list(hit.matched_pages)
        if pages:
            self.index.touch_computed(pages)
        new_pages = 0
        parent_hash = hit.hashes[-1] if hit.hashes else None
        i = hit.matched_tokens
        ps = self.cfg.page_size
        while i < len(tokens):
            take = tokens[i : i + ps]
            pid = self.index.alloc_or_evict(owner, tenant=tenant)
            self.pool.write_tokens(pid, take, 0)
            pages.append(pid)
            new_pages += 1
            if publish and len(take) == ps:
                # First block may carry cache_salt.
                ex = (cache_salt + extra) if (i == 0 and cache_salt) else extra
                parent_hash = self.index.publish_full(
                    pid, take, parent_hash, extra=ex, kind=PageKind.COMMITTED
                )
            elif len(take) < ps:
                self.index.meta[pid] = HashPageMeta(
                    pid, None, PageKind.COMMITTED, monotonic(), len(take)
                )
            i += len(take)
        return MaterializeResult(
            page_ids=pages,
            reused_hash_pages=len(hit.matched_pages),
            new_pages=new_pages,
            matched_tokens=hit.matched_tokens,
            last_hash=parent_hash,
        )

    def fork_alias(self, parent_pages: list[PageId]) -> list[PageId]:
        """O(1) per page: child inherits trunk by incref (ForkServe fork)."""
        for pid in parent_pages:
            self.pool.incref(pid)
        self.stats.fork_aliases += len(parent_pages)
        return list(parent_pages)

    def append_residual(
        self,
        owner: NodeId,
        existing: list[PageId],
        tokens: TokenSeq,
        *,
        speculative: bool = False,
        tenant: str | None = None,
        parent_hash: BlockHash | None = None,
        publish_committed: bool = False,
    ) -> tuple[list[PageId], BlockHash | None]:
        """Append residual tokens with CoW on shared tail pages."""
        tokens = as_tokens(tokens)
        if not tokens:
            return existing, parent_hash
        pages = list(existing)
        remaining = list(tokens)
        ps = self.cfg.page_size
        kind = PageKind.SPECULATIVE if speculative else PageKind.COMMITTED

        if pages:
            last = pages[-1]
            last_page = self.pool.get(last)
            space = ps - last_page.n_valid
            if space > 0:
                last = self.pool.cow_if_needed(last, owner)
                pages[-1] = last
                chunk = tuple(remaining[:space])
                self.pool.write_tokens(last, chunk, self.pool.get(last).n_valid)
                remaining = remaining[len(chunk) :]
                if not speculative and self.pool.get(last).n_valid == ps and publish_committed:
                    # Token content for hash: we only have n_valid; caller should
                    # pass exact tokens when publishing. Skip partial publish here.
                    pass

        while remaining:
            pid = self.index.alloc_or_evict(owner, tenant=tenant)
            take = tuple(remaining[:ps])
            self.pool.write_tokens(pid, take, 0)
            pages.append(pid)
            remaining = remaining[len(take) :]
            if speculative:
                self.index.meta[pid] = HashPageMeta(
                    pid, None, PageKind.SPECULATIVE, monotonic(), len(take)
                )
            elif publish_committed and len(take) == ps:
                parent_hash = self.index.publish_full(
                    pid, take, parent_hash, kind=PageKind.COMMITTED
                )
            else:
                self.index.meta[pid] = HashPageMeta(
                    pid, None, kind, monotonic(), len(take)
                )
        return pages, parent_hash

    def commit_publish(
        self,
        page_ids: list[PageId],
        full_tokens: TokenSeq,
        *,
        extra: bytes = b"",
        cache_salt: bytes = b"",
    ) -> list[BlockHash]:
        """After LCP commit: hash every full page so other sessions can APC-hit."""
        full_tokens = as_tokens(full_tokens)
        ps = self.cfg.page_size
        hashes: list[BlockHash] = []
        parent: BlockHash | None = None
        for i, pid in enumerate(page_ids):
            start = i * ps
            chunk = full_tokens[start : start + ps]
            if len(chunk) < ps:
                meta = self.index.meta.get(pid)
                if meta is not None:
                    meta.kind = PageKind.COMMITTED
                break
            ex = (cache_salt + extra) if i == 0 else extra
            bh = self.index.promote_spec_to_committed(pid, chunk, parent, extra=ex)
            if bh is not None:
                hashes.append(bh)
                parent = bh
        return hashes

    def release_pages_reverse(self, page_ids: list[PageId]) -> None:
        """APC free: release last block first (least reusable)."""
        for pid in reversed(page_ids):
            self.pool.decref(pid)
            if self.pool.get(pid).ref == 0:
                self.index.release_to_free(pid)


# ---------------------------------------------------------------------------
# Session facade used by demos / tests
# ---------------------------------------------------------------------------


@dataclass
class HybridSession:
    session_id: str
    owner: NodeId
    pages: list[PageId] = field(default_factory=list)
    tokens: TokenSeq = ()
    last_hash: BlockHash | None = None
    children: list[str] = field(default_factory=list)
    speculative: bool = False


class HashForkServe:
    """Small orchestrator exposing APC + fork verbs on one pool."""

    def __init__(self, config: ForkServeConfig | None = None, *, algo: str = "sha256") -> None:
        self.hf = HashForkPool(config, algo=algo)
        self.sessions: dict[str, HybridSession] = {}
        self._next_owner = 1

    def open(
        self,
        session_id: str,
        tokens: TokenSeq | list[int],
        *,
        cache_salt: bytes = b"",
    ) -> HybridSession:
        owner = NodeId(self._next_owner)
        self._next_owner += 1
        mat = self.hf.materialize_with_hash(
            owner, as_tokens(tokens), cache_salt=cache_salt, publish=True
        )
        sess = HybridSession(
            session_id=session_id,
            owner=owner,
            pages=mat.page_ids,
            tokens=as_tokens(tokens),
            last_hash=mat.last_hash,
        )
        self.sessions[session_id] = sess
        return sess

    def fork(self, parent_id: str, child_id: str, known_suffix: TokenSeq | list[int] = ()) -> HybridSession:
        parent = self.sessions[parent_id]
        owner = NodeId(self._next_owner)
        self._next_owner += 1
        pages = self.hf.fork_alias(parent.pages)
        known = as_tokens(known_suffix)
        last = parent.last_hash
        if known:
            pages, last = self.hf.append_residual(
                owner, pages, known, speculative=True, parent_hash=last
            )
        child = HybridSession(
            session_id=child_id,
            owner=owner,
            pages=pages,
            tokens=parent.tokens + known,
            last_hash=last,
            speculative=True,
        )
        parent.children.append(child_id)
        self.sessions[child_id] = child
        return child

    def commit(self, session_id: str, full_prompt: TokenSeq | list[int]) -> HybridSession:
        sess = self.sessions[session_id]
        prompt = as_tokens(full_prompt)
        # Extend if needed (tail after speculative known suffix).
        if len(prompt) > len(sess.tokens):
            tail = prompt[len(sess.tokens) :]
            sess.pages, sess.last_hash = self.hf.append_residual(
                sess.owner,
                sess.pages,
                tail,
                speculative=False,
                parent_hash=sess.last_hash,
                publish_committed=False,
            )
            sess.tokens = prompt
        elif len(prompt) < len(sess.tokens):
            # Truncate to LCP length in token space; page truncate is approximate
            # for the hybrid facade (full engine uses tree.truncate_residual).
            keep_pages = pages_for_tokens(len(prompt), self.hf.cfg.page_size)
            drop = sess.pages[keep_pages:]
            sess.pages = sess.pages[:keep_pages]
            for pid in drop:
                self.hf.pool.decref(pid)
            sess.tokens = prompt
        self.hf.commit_publish(sess.pages, sess.tokens)
        sess.speculative = False
        return sess

    def close(self, session_id: str) -> None:
        sess = self.sessions.pop(session_id, None)
        if sess is None:
            return
        self.hf.release_pages_reverse(sess.pages)

    @property
    def stats(self) -> HybridStats:
        return self.hf.stats
