"""Turn peak-KV savings into extra concurrent slots (paper eq. 9).

Θ ∝ C / T_e2e and C ∝ HBM / M. ForkServe lowers M by 20–38% vs APC on the
same ToT tree, so the scheduler can admit more sessions or a larger leftover
speculative token budget from the same GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

from forkserve.pages import clone_memory_bytes, cow_memory_bytes


@dataclass(frozen=True, slots=True)
class SpecPoolPlan:
    m_cow_bytes: float
    m_apc_bytes: float
    kv_saving: float
    slots_apc: int
    slots_fs: int
    extra_slots: int
    extra_batched_tokens: int
    spec_pool_bytes: float


def kv_saving(trunk: int, residuals: list[int], k: int) -> float:
    cow = cow_memory_bytes(trunk, residuals, 1.0)
    clone = clone_memory_bytes(trunk, residuals, 1.0, k)
    if clone <= 0:
        return 0.0
    return max(0.0, 1.0 - cow / clone)


def concurrent_slots(hbm_bytes: float, bytes_per_seq: float) -> int:
    if bytes_per_seq <= 0:
        return 0
    return max(1, int(hbm_bytes // bytes_per_seq))


def extra_batched_tokens(base: int, saving: float, spec_pool_frac: float) -> int:
    """Give the leftover class a share of the memory that CoW freed."""
    if saving <= 0.0 or spec_pool_frac <= 0.0:
        return 0
    # C_fs / C_apc = 1 / (1-s). Extra tokens scale the same way, then take a slice.
    scale = 1.0 / max(1e-6, 1.0 - saving) - 1.0
    return max(0, int(base * scale * spec_pool_frac))


def plan_spec_pool(
    *,
    hbm_bytes: float,
    bytes_per_token: float,
    trunk: int,
    residuals: list[int],
    k: int,
    base_batched_tokens: int,
    spec_pool_frac: float,
) -> SpecPoolPlan:
    cow = cow_memory_bytes(trunk, residuals, bytes_per_token)
    apc = clone_memory_bytes(trunk, residuals, bytes_per_token, k)
    saving = 0.0 if apc <= 0 else max(0.0, 1.0 - cow / apc)
    slots_apc = concurrent_slots(hbm_bytes, apc)
    slots_fs = concurrent_slots(hbm_bytes, cow)
    extra = max(0, slots_fs - slots_apc)
    spec_bytes = max(0.0, (apc - cow) * spec_pool_frac)
    extra_tok = extra_batched_tokens(base_batched_tokens, saving, spec_pool_frac)
    return SpecPoolPlan(
        m_cow_bytes=cow,
        m_apc_bytes=apc,
        kv_saving=saving,
        slots_apc=slots_apc,
        slots_fs=slots_fs,
        extra_slots=extra,
        extra_batched_tokens=extra_tok,
        spec_pool_bytes=spec_bytes,
    )


def throughput_tokens_per_s(
    slots: int,
    decode_tokens_per_request: float,
    e2e_s: float,
) -> float:
    """Eq. 9 style bound: slots in flight, each finishing every e2e_s."""
    if e2e_s <= 0 or slots <= 0:
        return 0.0
    return slots * decode_tokens_per_request / e2e_s
