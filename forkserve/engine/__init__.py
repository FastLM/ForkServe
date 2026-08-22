from forkserve.engine.protocol import EngineBackend, PrefillRequest, DecodeRequest
from forkserve.engine.mock import MockBackend

__all__ = ["DecodeRequest", "EngineBackend", "MockBackend", "PrefillRequest"]

try:
    from forkserve.engine.vllm_backend import VllmBackend

    __all__.append("VllmBackend")
except Exception:  # pragma: no cover - optional extra
    VllmBackend = None  # type: ignore[misc, assignment]
