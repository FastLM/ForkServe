"""ForkServe: the branch is the serving object.

``fork`` aliases the parent's KV. Each residual is rejected at the cheapest
admissible stage. Decode attends the committed spine only.

Public surface is the harness verbs plus ``Engine`` / ``Orchestrator``.
Speculation never feeds unverified tokens into decode.
"""

from forkserve.api import Engine
from forkserve.config import ForkServeConfig
from forkserve.hash_forkserve import HashForkServe
from forkserve.orchestrator import Orchestrator
from forkserve.prefill_prune import PrefillPruner, app_config
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
    "HashForkServe",
    "PrefillPruner",
    "app_config",
    "JoinPolicy",
    "NodeId",
    "NodeMode",
    "Orchestrator",
    "SessionId",
]
__version__ = "0.1.0"
