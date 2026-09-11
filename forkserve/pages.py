"""Copy-on-write KV pages on top of a PagedAttention block pool (§5.2).

A physical page is (id, ref, ro, owner). A node's logical table is
σ_v = (off_v, ρ_v): an offset into the parent table plus a private residual
list, so fork copies O(1) words rather than O(L/P) page ids.

Invariants
----------
* Liveness: refcount equals the number of non-Dead nodes that map the page.
* Writes never mutate a page with ref > 1 or ro=True; they CoW the tail.
* Abort cost is O(residual pages), independent of trunk length (Lemma 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol

from forkserve.config import ForkServeConfig
from forkserve.types import NodeId, PageId, TokenId, TokenSeq


class KvStore(Protocol):
    """Backend-owned KV tensor handle. Mock stores tokens; vLLM stores blocks."""

    def alloc(self, n_tokens: int) -> object: ...

    def copy_rows(self, src: object, n_valid: int) -> object: ...

    def free(self, handle: object) -> None: ...

    def write(self, handle: object, tokens: TokenSeq, offset: int = 0) -> None: ...


@dataclass(slots=True)
class PhysicalPage:
    id: PageId
    ref: int = 0
    ro: bool = False
    owner: NodeId | None = None
    n_valid: int = 0
    handle: object | None = None
    tenant: str | None = None

    @property
    def shared(self) -> bool:
        return self.ref > 1 or self.ro


@dataclass(slots=True)
class LogicalTable:
    """σ_v: parent alias + private residual pages.

    ``parent`` is the table we alias for ``alias_len`` tokens. Residual pages
    cover tokens [alias_len, alias_len + residual_tokens).
    """

    parent: LogicalTable | None
    alias_len: int
    residual: list[PageId] = field(default_factory=list)
    residual_tokens: int = 0

    def depth(self) -> int:
        d = 0
        cur: LogicalTable | None = self
        while cur is not None:
            d += 1
            cur = cur.parent
        return d


class PagePool:
    """Refcounted CoW allocator. Thread-unsafe; the engine loop owns it."""

    def __init__(self, config: ForkServeConfig, store: KvStore) -> None:
        self.cfg = config
        self.store = store
        self._pages: dict[PageId, PhysicalPage] = {}
        self._free: list[PageId] = []
        self._next = 1
        self.bytes_live = 0

    @property
    def page_size(self) -> int:
        return self.cfg.page_size

    def __len__(self) -> int:
        return sum(1 for p in self._pages.values() if p.ref > 0)

    def footprint_bytes(self) -> float:
        live = sum(1 for p in self._pages.values() if p.ref > 0)
        return live * self.cfg.page_size * self.cfg.bytes_per_token

    def alloc_page(self, owner: NodeId, *, ro: bool = False, tenant: str | None = None) -> PageId:
        if self._free:
            pid = self._free.pop()
            page = self._pages[pid]
            page.ref = 1
            page.ro = ro
            page.owner = owner
            page.n_valid = 0
            page.tenant = tenant
            page.handle = self.store.alloc(self.page_size)
        else:
            pid = PageId(self._next)
            self._next += 1
            page = PhysicalPage(
                id=pid,
                ref=1,
                ro=ro,
                owner=owner,
                handle=self.store.alloc(self.page_size),
                tenant=tenant,
            )
            self._pages[pid] = page
        self.bytes_live += self.page_size * self.cfg.bytes_per_token
        return pid

    def incref(self, pid: PageId) -> None:
        self._pages[pid].ref += 1

    def decref(self, pid: PageId) -> None:
        page = self._pages[pid]
        page.ref -= 1
        if page.ref <= 0:
            if page.handle is not None:
                self.store.free(page.handle)
            page.handle = None
            page.owner = None
            page.n_valid = 0
            page.ro = False
            page.tenant = None
            page.ref = 0
            self._free.append(pid)
            self.bytes_live -= self.page_size * self.cfg.bytes_per_token

    def get(self, pid: PageId) -> PhysicalPage:
        return self._pages[pid]

    def pin_readonly(self, pids: Iterable[PageId]) -> None:
        for pid in pids:
            self._pages[pid].ro = True

    def cow_if_needed(self, pid: PageId, owner: NodeId) -> PageId:
        """Write path: if shared or ro, copy valid tail rows into a fresh page."""
        self.reload([pid])
        page = self._pages[pid]
        if page.ref == 1 and not page.ro:
            page.owner = owner
            return pid
        new_id = self.alloc_page(owner, tenant=page.tenant)
        new_page = self._pages[new_id]
        src = self._resident_handle(page)
        if src is not None:
            new_page.handle = self.store.copy_rows(src, page.n_valid)
        new_page.n_valid = page.n_valid
        self.decref(pid)
        return new_id

    def split_at(self, pid: PageId, keep_valid: int, owner: NodeId) -> PageId:
        """Mid-page LCP split (§6.4): siblings keep the speculative tail."""
        self.reload([pid])
        page = self._pages[pid]
        if keep_valid >= page.n_valid and page.ref == 1 and not page.ro:
            return pid
        new_id = self.alloc_page(owner, tenant=page.tenant)
        new_page = self._pages[new_id]
        src = self._resident_handle(page)
        if src is not None:
            new_page.handle = self.store.copy_rows(src, keep_valid)
        new_page.n_valid = keep_valid
        self.decref(pid)
        return new_id

    def write_tokens(self, pid: PageId, tokens: TokenSeq, offset: int) -> None:
        page = self._pages[pid]
        if page.handle is None:
            raise RuntimeError(f"write to freed page {pid}")
        handle = self._resident_handle(page)
        self.store.write(handle, tokens, offset)
        page.n_valid = max(page.n_valid, offset + len(tokens))

    def offload(self, pids: Iterable[PageId]) -> int:
        """Park residual handles in DRAM. Trunk pages pinned ``ro`` stay in HBM."""
        n = 0
        for pid in pids:
            page = self._pages.get(pid)
            if page is None or page.handle is None or page.ro:
                continue
            if isinstance(page.handle, tuple) and page.handle and page.handle[0] == "dram":
                continue
            page.handle = ("dram", page.handle)
            n += 1
        return n

    def reload(self, pids: Iterable[PageId]) -> int:
        n = 0
        for pid in pids:
            page = self._pages.get(pid)
            if page is None:
                continue
            if isinstance(page.handle, tuple) and page.handle and page.handle[0] == "dram":
                page.handle = page.handle[1]
                n += 1
        return n

    @staticmethod
    def _resident_handle(page: PhysicalPage) -> object:
        handle = page.handle
        if isinstance(handle, tuple) and handle and handle[0] == "dram":
            return handle[1]
        return handle

    def walk_pages(self, table: LogicalTable, length: int) -> list[PageId]:
        """Materialize the page-id spine for attention over ``length`` tokens."""
        chain: list[LogicalTable] = []
        cur: LogicalTable | None = table
        while cur is not None:
            chain.append(cur)
            cur = cur.parent
        chain.reverse()
        out: list[PageId] = []
        covered = 0
        for tbl in chain:
            if covered >= length:
                break
            if tbl.residual:
                take = min(len(tbl.residual), -(-min(tbl.residual_tokens, length - covered) // self.page_size))
                # residual covers tokens starting at tbl.alias_len
                if covered < tbl.alias_len:
                    # Should have been covered by ancestors; skip holes.
                    covered = tbl.alias_len
                need = min(length - covered, tbl.residual_tokens)
                n_pages = (need + self.page_size - 1) // self.page_size
                out.extend(tbl.residual[:n_pages])
                covered += min(need, n_pages * self.page_size)
        return out

    def alias_pages(self, table: LogicalTable) -> list[PageId]:
        """All physical pages reachable from this table (for pin / abort)."""
        seen: list[PageId] = []
        cur: LogicalTable | None = table
        while cur is not None:
            seen.extend(cur.residual)
            cur = cur.parent
        return seen


class TokenKvStore:
    """CPU reference store used by tests and the mock engine."""

    def alloc(self, n_tokens: int) -> object:
        return [0] * n_tokens

    def copy_rows(self, src: object, n_valid: int) -> object:
        buf = list(src)  # type: ignore[arg-type]
        return buf[:n_valid] + [0] * (len(buf) - n_valid)

    def free(self, handle: object) -> None:
        return None

    def write(self, handle: object, tokens: TokenSeq, offset: int = 0) -> None:
        buf: list[TokenId] = handle  # type: ignore[assignment]
        for i, tok in enumerate(tokens):
            buf[offset + i] = tok


def pages_for_tokens(n_tokens: int, page_size: int) -> int:
    if n_tokens <= 0:
        return 0
    return (n_tokens + page_size - 1) // page_size


def cow_memory_bytes(trunk: int, residuals: Iterable[int], bpt: float) -> float:
    """Equation (3): M_CoW = b (L + Σ ℓ_i)."""
    return bpt * (trunk + sum(residuals))


def clone_memory_bytes(trunk: int, residuals: Iterable[int], bpt: float, k: int) -> float:
    """Equation (3): M_clone = b (k L + Σ ℓ_i)."""
    return bpt * (k * trunk + sum(residuals))
