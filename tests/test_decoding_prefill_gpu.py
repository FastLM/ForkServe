"""Admission and decode-cut rules for the GPU decoding-prune run."""

from forkserve.bench_tasks import gsm8k_thoughts

from experiments.decoding_prefill_gpu import admit, cut_indices, vote


def test_published_strategies_only_shrink_under_app() -> None:
    thoughts = gsm8k_thoughts(4)
    assert admit(thoughts, "base") == (0, 1, 2, 3)
    assert admit(thoughts, "draft") == (0, 1, 2, 3)
    assert admit(thoughts, "app") == (0,)


def test_decoder_cuts_the_lower_scores_and_always_leaves_one() -> None:
    scores = {0: 0.1, 1: 0.4, 2: 0.2, 3: 0.9}
    assert cut_indices("esc", [0, 1, 2, 3], scores) == (0, 1, 2, 3)
    assert cut_indices("specrej", [0, 1, 2, 3], scores) == (1, 3)
    assert cut_indices("dpts", [0, 1, 2, 3], scores) == (1, 3)
    assert cut_indices("specrej", [0], scores) == (0,)
    assert cut_indices("dpts", [1, 3], {1: 0.2, 3: 0.9}) == (3,)


def test_vote_uses_the_majority_answer() -> None:
    assert vote(["#### 12", "the answer is 12", "#### 9"], "12")
    assert not vote(["#### 9", "#### 9"], "12")
