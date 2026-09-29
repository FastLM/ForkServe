"""Next-turn prefill after prune, with and without slack fill.

Control plane, no GPU. Ten sessions, k=4, L=256, residual=32.
Three siblings are dropped. The freed budget pins an 80-token known
suffix that the next commit actually sends, plus a 32-token observation
that is not known yet. Decode budget is the same on both sides.
"""

from forkserve.config import ForkServeConfig
from forkserve.prefill_prune import PrefillHashIndex, PrefillPruner
from forkserve.prune import plus_config
from forkserve.slack_fill import freed_tokens, miss_tokens, refill

THOUGHTS = (
    "Thought 1: compute carefully and box the answer with ####.",
    "Thought 2: ****loop****loop****loop****loop****loop",
    "Thought 3: undefined nan junk residual that cannot be a proof",
    "Thought 4: try a substitution then combine like terms.",
)


def main() -> None:
    n = 10
    trunk_len = 256
    residual = 32
    suffix_len = 80
    obs_len = 32
    cfg = plus_config(ForkServeConfig(page_size=16, prefill_us_per_token=12.0))
    assert cfg.answer_stop == ""
    bare_miss = 0
    fill_miss = 0
    freed_sum = 0
    pinned_sum = 0
    for s in range(n):
        spine = tuple(range(s * 100_000, s * 100_000 + trunk_len))
        fulls = [
            spine + tuple(range(10_000 + j * 64, 10_000 + j * 64 + residual))
            for j in range(4)
        ]
        cold = PrefillHashIndex(cfg.page_size)
        cold.publish(spine)
        plan = PrefillPruner(cfg, winner=0, hash_index=cold).plan(
            list(THOUGHTS),
            full_prompts=fulls,
            token_counts=[residual] * 4,
        )
        freed = freed_tokens(plan.decisions)
        nxt = tuple(range(50_000 + s * 100, 50_000 + s * 100 + suffix_len))
        obs = tuple(range(80_000 + s * 100, 80_000 + s * 100 + obs_len))
        prompt = spine + nxt + obs
        bare_miss += miss_tokens(cold, prompt)
        filled_index = PrefillHashIndex(cfg.page_size)
        filled_index.publish(spine)
        filled = refill(filled_index, spine, [nxt], freed=freed)
        fill_miss += miss_tokens(filled_index, prompt)
        freed_sum += freed
        pinned_sum += filled.pinned_tokens
    us = cfg.prefill_us_per_token
    print(
        f"sessions={n} answer_stop={cfg.answer_stop!r} decode_budget=same\n"
        f"freed_tokens={freed_sum} pinned_tokens={pinned_sum}\n"
        f"next_prefill_bare={bare_miss} ({bare_miss * us / 1000:.2f} ms)\n"
        f"next_prefill_fill={fill_miss} ({fill_miss * us / 1000:.2f} ms)\n"
        f"cut={(bare_miss - fill_miss) / bare_miss:.1%}"
    )


if __name__ == "__main__":
    main()
