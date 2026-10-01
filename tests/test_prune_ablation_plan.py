"""Planner checks for the GPU prefill × decoding prune ablation."""

from experiments.prune_ablation_gpu import (
    admitted_indices,
    assign_tokens,
    build_jobs,
    build_plugin_jobs,
    build_prefill_jobs,
    build_thoughts,
    build_thresh_jobs,
    build_wide_jobs,
    drop_indices,
    esc_should_stop,
    parse_ks,
    unique_prefill_bars,
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
    # 0.15 drops the loop (0.02) and keeps illegal text (0.22).
    assert admitted_indices("draft", thoughts, 0.15) == [0, 1, 3]
    assert admitted_indices("app", thoughts, 0.45) == [0, 1]
    assert admitted_indices("app", thoughts, 0.15) == [0, 1, 3]
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
    assert sum(dpts_app["assigned"].values()) == 2 * 512
    assert dpts_app["avoided_decode"] == 2 * 100
    assert dpts_app["admitted"] == dpts_draft["admitted"]


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
    assert drop_indices("specrej", thoughts, alpha=0.25) == [4, 6]
    assert len(drop_indices("specrej", thoughts, alpha=0.5)) == 4
    wide = build_thoughts(16, "hopeless")
    assert admitted_indices("draft", wide, 0.45) == list(range(8))
    assert admitted_indices("app", wide, 0.45) == list(range(8))
    assert drop_indices("dpts", wide, alpha=0.5) == list(range(8, 16))
    assert len(drop_indices("specrej", wide, alpha=0.5)) == 8


def test_app_keeps_by_score_not_a_fraction_of_k() -> None:
    """Default APP admits every live thought at 0.45. Other modes stay optional."""
    for k in (4, 8, 16):
        thoughts = build_thoughts(k, "hopeless")
        live = list(range((k + 1) // 2))
        assert admitted_indices("app", thoughts, 0.45) == live
        assert admitted_indices("draft", thoughts, 0.45) == live
    thoughts = build_thoughts(16, "hopeless")
    assert admitted_indices("app", thoughts, 0.80) == list(range(8))
    # 0.15 keeps live + illegal text (odd dead slots), drops only loops.
    assert admitted_indices("app", thoughts, 0.15) == [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15]
    assert admitted_indices("app", thoughts, 0.45, admit_mode="winner") == [0]
    assert len(admitted_indices("app", thoughts, 0.45, admit_mode="top_m", keep_m=2)) == 2
    assert 0 in admitted_indices("app", thoughts, 0.45, admit_mode="top_m", keep_m=2)
    got = admitted_indices("app", thoughts, 0.45, admit_mode="alpha", admit_alpha=0.25)
    assert len(got) == 4
    assert 0 in got


def test_wide_grid_covers_k8_and_k16() -> None:
    assert parse_ks("8,16") == (8, 16)
    jobs = build_wide_jobs((4, 8, 16))
    keys = [work_key(job) for job in jobs]
    assert len(keys) == len(set(keys))
    assert {job["k"] for job in jobs} == {4, 8, 16}
    assert len(jobs) == 27
    assert all(job["n"] == 16 and job["budget"] == 512 for job in jobs)
    assert all(job["tag"] == "wide" for job in jobs)
    thresh = build_thresh_jobs((4, 8, 16), (0.15, 0.45))
    assert len(thresh) == 27
    assert {job["threshold"] for job in thresh if job["policy"] == "app"} == {0.15, 0.45}
    assert unique_prefill_bars(8, (0.0, 0.05, 0.10, 0.20, 0.25, 0.45, 0.80)) == [0.0, 0.05, 0.25]
    prefill = build_prefill_jobs((4, 8, 16), (0.0, 0.10, 0.30))
    assert len(prefill) == 9
    assert all(job["method"] == "esc" and job["tag"] == "prefill" for job in prefill)
    assert {job["prefill_threshold"] for job in prefill} == {0.0, 0.10, 0.30}
    plugin = build_plugin_jobs((4, 8, 16))
    assert len(plugin) == 18
    assert all(job["tag"] == "plugin" for job in plugin)
    assert {job["prefill_threshold"] for job in plugin} == {0.15}
    assert {job["method"] for job in plugin} == {"esc", "specrej", "dpts"}
    assert {job["policy"] for job in plugin} == {"base", "app"}
    assert all(float(job["decode_threshold"]) == 0.45 for job in plugin)


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
    wide_mild = build_mild_jobs(ks=(8, 16))
    assert {job["k"] for job in wide_mild} == {8, 16}
    assert len(wide_mild) == 24


def test_scale_cuts_loops_and_keeps_illegal_text() -> None:
    from experiments.prune_ablation_gpu import (
        assign_tokens,
        build_scale_jobs,
        loop_indices,
    )

    thoughts = build_thoughts(16, "hopeless", family="contest")
    assert loop_indices(thoughts) == [8, 10, 12, 14]
    common = dict(policy="base", threshold=0.45, budget=1024, tau=128, dpts_step=128, alpha=0.5)
    pre = assign_tokens(
        "mild", thoughts, prefill_drop=4, decode_drop=0, mild_step=128, mild_target="loop", **common
    )
    assert pre["admitted"] == [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15]
    assert pre["kept"] == 12
    both = assign_tokens(
        "mild", thoughts, prefill_drop=2, decode_drop=2, mild_step=128, mild_target="loop", **common
    )
    assert 8 not in both["admitted"] and 10 not in both["admitted"]
    assert both["assigned"][12] == 128 and both["assigned"][14] == 128
    assert both["assigned"][9] == 1024
    dead = assign_tokens(
        "mild", thoughts, prefill_drop=8, decode_drop=0, mild_step=128, mild_target="low_score", **common
    )
    assert dead["admitted"] == [0, 1, 2, 3, 4, 5, 6, 7]
    jobs = build_scale_jobs()
    assert len(jobs) == 14
    assert {job["k"] for job in jobs} == {4, 8, 16}
    assert {job["workload"] for job in jobs} == {"math500"}
    assert all(job["n"] == 32 and job["budget"] == 1024 for job in jobs)
    assert len({work_key(job) for job in jobs}) == len(jobs)


def test_grow_probes_illegal_text_and_can_extend_it() -> None:
    from experiments.prune_ablation_gpu import build_grow_jobs, plan_grow, should_grow

    thoughts = build_thoughts(8, "hopeless", family="contest")
    plan = plan_grow(thoughts, budget=1024, probe=128, threshold=0.45, skip_loops=True)
    assert plan["probe"] == [5, 7]
    assert plan["assigned"][0] == 1024
    assert 4 not in plan["admitted"] and 6 not in plan["admitted"]
    probed = plan_grow(thoughts, budget=1024, probe=128, threshold=0.45, skip_loops=False)
    assert probed["probe"] == [4, 5, 6, 7]
    assert should_grow("Solve the equation, then box the value.")
    assert not should_grow("****loop****loop****loop****loop")
    assert not should_grow("undefined nan junk residual")
    jobs = build_grow_jobs()
    assert len(jobs) == 9
    assert len({work_key(job) for job in jobs}) == len(jobs)
    assert all(job["workload"] == "math500" and job["budget"] == 1024 for job in jobs)


def test_prefill_and_decode_bars_are_independent() -> None:
    thoughts = build_thoughts(4, "hopeless")
    # Prefill 0.15 starts illegal text; decode 0.45 still cuts it later.
    assert admitted_indices("app", thoughts, 0.15) == [0, 1, 3]
    assert drop_indices("dpts", thoughts, alpha=0.5, decode_threshold=0.45) == [2, 3]
    # Raising only the prefill bar does not change the decode cut list.
    assert drop_indices("dpts", thoughts, alpha=0.5, decode_threshold=0.45) == [2, 3]
    tight = assign_tokens(
        "dpts",
        thoughts,
        policy="app",
        threshold=0.15,
        prefill_threshold=0.15,
        decode_threshold=0.45,
        budget=512,
        tau=256,
        dpts_step=100,
        alpha=0.5,
    )
    assert tight["admitted"] == [0, 1, 3]
    assert tight["assigned"][3] == 100
    assert 2 not in tight["admitted"]
    # Decode bar 0.10 would keep illegal text running; prefill bar unchanged.
    loose = assign_tokens(
        "dpts",
        thoughts,
        policy="app",
        threshold=0.15,
        prefill_threshold=0.15,
        decode_threshold=0.10,
        budget=512,
        tau=256,
        dpts_step=100,
        alpha=0.5,
    )
    assert loose["admitted"] == [0, 1, 3]
    assert loose["assigned"][3] == 512
    assert loose["drop"] == [2]


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
