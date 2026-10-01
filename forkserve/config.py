"""Serving knobs for CoW pages, admit, two-class batch, and retention.

λ converts 1 ms of committed TBT regression into the same units as a 4 ms
TTFT win. C_spec = 512 so one sibling system prompt cannot swallow leftover
budget that could cover three wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ForkServeConfig:
    # PagedAttention
    page_size: int = 16
    bytes_per_token: float = 320_000.0  # Llama-3.1-70B GQA fp16
    kv_dtype_bytes: int = 2

    # Speculative planner
    c_spec: int = 512
    q_min: float = 0.35
    branch_cap_m: int = 6
    grammar_top_m: int = 3
    ngram_order: int = 3
    ngram_prefix: int = 32
    lambda_tbt: float = 4.0  # 1 ms TBT ≡ 4 ms TTFT
    eta_partial: float = 0.5
    prefill_us_per_token: float = 12.0  # chunked prefill model; backend overrides

    # Two-class scheduler
    max_batched_tokens: int = 8192
    chunked_prefill_cap: int = 2048
    tick_ms: float = 8.0
    tbt_slo_ms: float = 50.0
    ttft_slo_ms: float = 200.0

    # Retention / offload
    tau_max_s: float = 8.0
    ttl_alpha: float = 1.2
    ttl_beta: float = 1.0
    hbm_capacity_bytes: float = 80.0 * (1 << 30)
    dram_capacity_bytes: float = 2.0 * (1 << 40)

    # Fairness
    spec_bill_kappa: float = 0.25
    vtc_enabled: bool = True

    # Placement
    residual_steal_enabled: bool = True
    num_workers: int = 1

    # Security
    isolate_tenants: bool = True
    placeholder_tool_args: bool = True

    # Prior sketch
    prior_decay: float = 0.97
    prior_width: int = 2048
    prior_depth: int = 4

    # ForkServe+ / APP: lazy abort, prune, spec-pool, hash skip, disagg gate.
    lazy_abort: bool = False
    pointer_swap: bool = True
    prune_enabled: bool = False
    # Prefill admission bar (APP). Frozen plug-in: 0.15. Not retuned per decoder.
    prefill_threshold: float = 0.15
    decode_threshold: float = 0.45
    # Alias of prefill_threshold for older callers.
    prune_threshold: float = 0.15
    early_prune_frac: float = 0.20
    spec_pool_frac: float = 0.0
    decode_stop: tuple[str, ...] = ()
    # "gsm" | "math" | "game24" | "code" | "auto" | "". Stops only after the
    # answer is complete (#### number, boxed, =24, or a code boundary).
    answer_stop: str = ""
    # Per-request hints (Game24 gold strings), aligned with the decode batch.
    answer_stop_hints: tuple[str, ...] = ()
    # APP admission rule. ``apply_admit_mode`` is the switch:
    #   score   — winner + every sibling with prefill score >= prefill_threshold
    #   winner  — skip_known_losers (winner only)
    #   top_m   — winner + next prefill_keep_m-1 by G/C
    #   alpha   — winner + next floor(admit_alpha * k)-1 by score
    admit_mode: str = "score"
    admit_alpha: float = 0.5
    # Harness already picked the winner: do not prefill the other residuals.
    skip_known_losers: bool = False
    hash_prune: bool = False
    # Share a page-aligned residual prefix across siblings (prefill and ship once).
    share_prefixes: bool = False
    # Optional count cap on non-winners (top G/C). 0 = no cap.
    gc_admit: bool = False
    prefill_keep_m: int = 0
    disagg_prefill: bool = False
    disagg_transfer_us_per_token: float = 2.0
    # After a prune, pin next-turn known suffixes into the freed KV budget.
    slack_fill: bool = False

    extra: dict[str, float] = field(default_factory=dict)

    def prefill_ms(self, n_tokens: int) -> float:
        """T_pre(ℓ) fallback when the engine has not reported a profile."""
        if n_tokens <= 0:
            return 0.0
        return (n_tokens * self.prefill_us_per_token) / 1000.0
