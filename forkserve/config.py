"""Paper defaults from §6–§9.

λ converts 1 ms of committed TBT regression into the same units as a 4 ms
TTFT win (interactive SLO of JITServe). C_spec = 512 so one sibling system
prompt cannot swallow leftover budget that could cover three wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ForkServeConfig:
    # PagedAttention (§5.2, §9.1)
    page_size: int = 16
    bytes_per_token: float = 320_000.0  # Llama-3.1-70B GQA fp16, paper §3.3
    kv_dtype_bytes: int = 2

    # Speculative planner (§6)
    c_spec: int = 512
    q_min: float = 0.35
    branch_cap_m: int = 6
    grammar_top_m: int = 3
    ngram_order: int = 3
    ngram_prefix: int = 32
    lambda_tbt: float = 4.0  # 1 ms TBT ≡ 4 ms TTFT
    eta_partial: float = 0.5
    prefill_us_per_token: float = 12.0  # chunked prefill model; backend overrides

    # Two-class scheduler (§7.1)
    max_batched_tokens: int = 8192
    chunked_prefill_cap: int = 2048
    tick_ms: float = 8.0
    tbt_slo_ms: float = 50.0
    ttft_slo_ms: float = 200.0

    # Retention / offload (§5.4)
    tau_max_s: float = 8.0
    ttl_alpha: float = 1.2
    ttl_beta: float = 1.0
    hbm_capacity_bytes: float = 80.0 * (1 << 30)
    dram_capacity_bytes: float = 2.0 * (1 << 40)

    # Fairness (§7.4)
    spec_bill_kappa: float = 0.25
    vtc_enabled: bool = True

    # Placement (§7.2)
    residual_steal_enabled: bool = True
    num_workers: int = 1

    # Security (§10)
    isolate_tenants: bool = True
    placeholder_tool_args: bool = True

    # Prior sketch
    prior_decay: float = 0.97
    prior_width: int = 2048
    prior_depth: int = 4

    extra: dict[str, float] = field(default_factory=dict)

    def prefill_ms(self, n_tokens: int) -> float:
        """T_pre(ℓ) fallback when the engine has not reported a profile."""
        if n_tokens <= 0:
            return 0.0
        return (n_tokens * self.prefill_us_per_token) / 1000.0
