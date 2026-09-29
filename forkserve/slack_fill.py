"""Pin next-turn known suffixes into the KV budget pruning just freed.

A dropped residual occupies no pages. Those tokens are a budget, not a
hole: the next committed prompt is a known suffix on the winner spine
(p = 1), while the pruned sibling will not be decoded. Publishing a
page-aligned prefix of that suffix, and nothing longer than the freed
budget, makes the following prefill a miss tail.
"""

from __future__ import annotations

from dataclasses import dataclass

from forkserve.prefill_prune import PrefillHashIndex
from forkserve.types import TokenSeq, as_tokens


@dataclass(slots=True)
class SlackFill:
    freed_tokens: int = 0
    pinned_tokens: int = 0
    suffixes: int = 0

    @property
    def unused_tokens(self) -> int:
        return max(0, self.freed_tokens - self.pinned_tokens)


def freed_tokens(decisions) -> int:
    """Residual tokens whose pages were not kept."""
    return sum(int(d.residual_tokens) for d in decisions if not d.keep)


def refill(
    index: PrefillHashIndex,
    spine: TokenSeq,
    suffixes: list[TokenSeq],
    *,
    freed: int,
) -> SlackFill:
    """Publish spine ‖ suffix prefixes until ``freed`` tokens are spent."""
    page = max(1, index.page_size)
    budget = max(0, int(freed))
    spine_t = as_tokens(spine)
    pinned = 0
    used = 0
    for raw in suffixes:
        if budget < page:
            break
        suf = as_tokens(raw)
        take = min(len(suf), budget)
        take = (take // page) * page
        if take <= 0:
            continue
        index.publish(spine_t + suf[:take])
        budget -= take
        pinned += take
        used += 1
    return SlackFill(freed_tokens=int(freed), pinned_tokens=pinned, suffixes=used)


def miss_tokens(index: PrefillHashIndex, prompt: TokenSeq) -> int:
    prompt_t = as_tokens(prompt)
    if not prompt_t:
        return 0
    return max(0, len(prompt_t) - index.lookup(prompt_t))
