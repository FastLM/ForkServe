"""Pruned residuals fund the next known suffix; decode length stays the budget."""

from forkserve.config import ForkServeConfig
from forkserve.prefill_prune import PrefillHashIndex, PrefillPruner
from forkserve.prune import plus_config
from forkserve.slack_fill import freed_tokens, miss_tokens, refill


def test_plus_does_not_stop_decode_by_default() -> None:
    cfg = plus_config(ForkServeConfig())
    assert cfg.answer_stop == ""
    assert cfg.slack_fill is True


def test_freed_budget_pins_next_suffix_and_cuts_the_miss() -> None:
    cfg = plus_config(ForkServeConfig(page_size=16, bytes_per_token=1.0))
    spine = tuple(range(256))
    residuals = [
        spine + tuple(range(1000, 1032)),
        spine + tuple(range(2000, 2032)),
        spine + tuple(range(3000, 3032)),
        spine + tuple(range(4000, 4032)),
    ]
    texts = [
        "Thought 1: add the numbers and box the answer.",
        "****loop****loop****loop****loop****",
        "undefined nan junk residual",
        "Thought 4: substitute then combine like terms.",
    ]
    index = PrefillHashIndex(cfg.page_size)
    index.publish(spine)
    plan = PrefillPruner(cfg, winner=0, hash_index=index).plan(
        texts,
        full_prompts=residuals,
        token_counts=[32, 32, 32, 32],
    )
    freed = freed_tokens(plan.decisions)
    assert freed >= 32
    nxt = tuple(range(5000, 5080))
    obs = tuple(range(9000, 9032))
    prompt = spine + nxt + obs
    bare = PrefillHashIndex(cfg.page_size)
    bare.publish(spine)
    miss_bare = miss_tokens(bare, prompt)
    filled = refill(index, spine, [nxt], freed=freed)
    miss_fill = miss_tokens(index, prompt)
    assert filled.pinned_tokens == 80
    assert filled.pinned_tokens <= freed
    assert miss_fill == len(obs)
    assert miss_bare - miss_fill == filled.pinned_tokens
