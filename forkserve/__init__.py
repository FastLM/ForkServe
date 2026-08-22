"""ForkServe: branch-aware speculative prefilling and CoW KV state.

Public surface is the five harness verbs plus ``Engine`` / ``Orchestrator``.
Speculation never feeds unverified tokens into decode (Theorem 2).
"""

from forkserve.api import Engine
from forkserve.config import ForkServeConfig
from forkserve.orchestrator import Orchestrator
from forkserve.types import (
    BranchId,
    JoinPolicy,
    NodeId,
    NodeMode,
    SessionId,
)

__all__ = [
    "BranchId",
    "Engine",
    "ForkServeConfig",
    "JoinPolicy",
    "NodeId",
    "NodeMode",
    "Orchestrator",
    "SessionId",
]
__version__ = "0.1.0"
