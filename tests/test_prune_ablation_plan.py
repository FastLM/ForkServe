"""Planner checks for the GPU prefill × decoding prune ablation."""

from experiments.prune_ablation_gpu import (
    admitted_indices,
    assign_tokens,
    build_jobs,
    build_thoughts,
    drop_indices,
    esc_should_stop,
    work_key,
)


def _tokens(method: str, policy: str, **kw: object) -> dict:
    thoughts = build_thoughts(int(kw.get("k", 4)), str(kw.get("mix", "hopeless")))
    return assign_tokens(
        method,
        thoughts,
        policy=policy,
        threshold=float(kw.get("threshold", 0.45)),
        budget=int(kw.get("budget", 512)),
        tau=int(kw.get("tau", 256)),
        dpts_step=int(kw.get("dpts_step", 100)),
        alpha=float(kw.get("alpha", 0.5)),
    )


def test_hopeless_draft_sees_the_two_dead_thoughts() -> None:
    thoughts = build_thoughts(4, "hopeless")
    assert admitted_indices("base", thoughts, 0.45) == [0, 1, 2, 3]
    assert admitted_indices("draft", thoughts, 0.45) == [0, 1]
    assert admitted_indices("draft", thoughts, 0.15) == [0, 1]
    assert admitted_indices("app", thoughts, 0.45) == [0]
    assert drop_indices("esc", thoughts, alpha=0.5) == []
    assert drop_indices("specrej", thoughts, alpha=0.5) == [2, 3]
    assert drop_indices("dpts", thoughts, alpha=0.5) == [2, 3]
    assert drop_indices("specrej", thoughts, alpha=0.25) == [2]


def test_prefill_moves_the_cut_before_the_prefix() -> None:
    esc = _tokens("esc", "base")
    esc_draft = _tokens("esc", "draft")
    dpts = _tokens("dpts", "base")
    dpts_draft = _tokens("dpts", "draft")
    dpts_app = _tokens("dpts", "app")
    sr = _tokens("specrej", "base")
    sr_draft = _tokens("specrej", "draft")

    assert sum(esc["assigned"].values()) == 4 * 512
    assert esc_draft["avoided_decode"] == 2 * 512
    assert sum(esc_draft["assigned"].values()) == 2 * 512

    assert dpts["assigned"][2] == 100
    assert dpts["assigned"][0] == 512
    assert sum(dpts["assigned"].values()) == 2 * 100 + 2 * 512
    # Draft never starts the two DPTS would have cut, so it saves the mini-step
    # and the survivors still run to the budget.
    assert dpts_draft["kept"] == 2
    assert dpts_draft["avoided_decode"] == 2 * 100
    assert sum(dpts_draft["assigned"].values()) == 2 * 512
    assert sr_draft["avoided_decode"] == 2 * 256
    assert sr["assigned"][2] == 256
    assert sum(dpts_app["assigned"].values()) == 512
    assert dpts_app["avoided_decode"] == 512 + 2 * 100


def test_clean_mix_only_rank_pruning_still_cuts() -> None:
    thoughts = build_thoughts(4, "clean")
    assert admitted_indices("draft", thoughts, 0.45) == [0, 1, 2, 3]
    assert drop_indices("dpts", thoughts, alpha=0.5) == []
    assert drop_indices("specrej", thoughts, alpha=0.5) == [1, 2]
    clean_dpts = _tokens("dpts", "base", mix="clean")
    assert clean_dpts["drop"] == []
    assert sum(clean_dpts["assigned"].values()) == 4 * 512


def test_alpha_scales_with_fanout() -> None:
    thoughts = build_thoughts(8, "hopeless")
    assert drop_indices("specrej", thoughts, alpha=0.25) == [4, 5]
    assert len(drop_indices("specrej", thoughts, alpha=0.5)) == 4


def test_esc_window_needs_the_same_marker() -> None:
    assert not esc_should_stop(["1"], 2)
    assert not esc_should_stop(["", ""], 2)
    assert not esc_should_stop(["1", "2"], 2)
    assert esc_should_stop(["3", "1", "1"], 2)


def test_mild_each_phase_drops_one_hopeless_thought() -> None:
    from experiments.prune_ablation_gpu import assign_tokens, build_mild_jobs

    thoughts = build_thoughts(4, "hopeless")
    common = dict(policy="base", threshold=0.45, budget=512, tau=64, dpts_step=64, alpha=0.5)
    pre = assign_tokens("mild", thoughts, prefill_drop=1, decode_drop=0, mild_step=64, **common)
    assert pre["admitted"] == [0, 1, 3]
    assert pre["kept"] == 3
    assert pre["avoided_decode"] == 512
    both = assign_tokens("mild", thoughts, prefill_drop=1, decode_drop=1, mild_step=64, **common)
    assert both["admitted"] == [0, 1, 3]
    assert both["assigned"][3] == 64
    assert both["assigned"][0] == 512
    assert both["kept"] == 2
    assert both["avoided_decode"] == 512 + (512 - 64)
    clean = build_thoughts(4, "clean")
    untouched = assign_tokens("mild", clean, prefill_drop=1, decode_drop=1, mild_step=64, **common)
    assert untouched["admitted"] == [0, 1, 2, 3]
    assert untouched["kept"] == 4
    assert untouched["avoided_decode"] == 0
    labels = [job["mild_label"] for job in build_mild_jobs()]
    assert labels.count("full") == 1
    assert "both-1@64" in labels
    assert "clean/prefill-1" in labels


def test_job_grid_is_unique_and_inside_budget() -> None:
    jobs = build_jobs()
    keys = [work_key(job) for job in jobs]
    assert len(keys) == len(set(keys))
    assert len([job for job in jobs if job["tag"] == "paper"]) == 9
    assert {job["k"] for job in jobs if job["tag"] == "branching"} == {2, 4, 8}
    assert any(job["esc_window"] == 2 for job in jobs)
    assert any(job["mix"] == "clean" for job in jobs)
    assert any(job["alpha"] == 0.25 for job in jobs)
    for job in jobs:
        if job["method"] == "specrej":
            assert job["tau"] <= job["budget"]
        if job["method"] == "dpts":
            assert job["dpts_step"] <= job["budget"]
