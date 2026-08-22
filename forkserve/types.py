"""Shared types and paper-level enumerations.

Node modes follow §5.1. Token sequences are lists of integer ids so LCP
commit is a pure token compare and never a string heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import NewType, Sequence

TokenId = int
TokenSeq = tuple[TokenId, ...]

SessionId = NewType("SessionId", str)
NodeId = NewType("NodeId", int)
BranchId = NewType("BranchId", str)
PageId = NewType("PageId", int)
WorkerId = NewType("WorkerId", int)


class NodeMode(str, Enum):
    """ρ_u in Definition 1."""

    SPEC = "Spec"
    COMMIT = "Commit"
    IDLE = "Idle"
    DEAD = "Dead"


class JobClass(str, Enum):
    """Two-class continuous batch (§7.1)."""

    COMMITTED = "committed"
    SPECULATIVE = "speculative"


class JoinPolicy(str, Enum):
    """Fan-out join policies from §5.3 / Autellix-style harnesses."""

    ALL = "all"
    FIRST_SUCCESS = "first"
    K_OF_N = "kofn"
    CONCAT = "concat"
    WINNER = "winner"
    SUMMARY = "summary"


class WorkKind(str, Enum):
    KNOWN_SUFFIX = "known_suffix"
    OBS_RESIDUAL = "obs_residual"


class SchemaKind(str, Enum):
    """Observation residual admission gate of Algorithm 1."""

    FREEFORM = "freeform"
    JSON = "json"
    XML = "xml"
    UNIFIED_DIFF = "unified_diff"
    TYPED_RETURN = "typed_return"


class TenantId(str):
    pass


def as_tokens(seq: Sequence[TokenId] | TokenSeq) -> TokenSeq:
    return tuple(int(t) for t in seq)


def lcp_len(a: Sequence[TokenId], b: Sequence[TokenId]) -> int:
    """Longest common prefix length. Tokenizer-side commit (§6.4)."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class Priority(IntEnum):
    """Harness-declared speculate() priority. Known suffixes starve residuals."""

    KNOWN = 100
    HIGH = 80
    NORMAL = 50
    LOW = 20


@dataclass(frozen=True, slots=True)
class TokenSpan:
    """A contiguous token range owned by one physical page."""

    start: int
    end: int  # exclusive

    def __len__(self) -> int:
        return self.end - self.start

    def covers(self, index: int) -> bool:
        return self.start <= index < self.end


@dataclass(slots=True)
class Counters:
    """Per-node traces that feed §9 accounting."""

    p_b: float = 0.0
    admitted_tokens: int = 0
    lcp_length: int = 0
    residual_tokens: int = 0
    cancelled: int = 0
    known_suffix_hit: bool = False
    residual_full_hit: bool = False
    residual_partial_hit: bool = False
    extra: dict[str, float] = field(default_factory=dict)
