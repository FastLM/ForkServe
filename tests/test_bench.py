from forkserve.bench import (
    default_tps,
    format_table,
    kv_bytes_per_token,
    parse_args,
    run_mock,
)


def test_qwen3_8b_kv_bytes() -> None:
    # 36 layers × 8 KV heads × 128 × K+V × bf16
    assert kv_bytes_per_token(36, 8, 128) == 147_456.0


def test_tp_sweep_matches_visible_gpus() -> None:
    assert default_tps(4, None) == [2, 4]
    assert default_tps(2, None) == [2]
    assert default_tps(1, None) == [1]
    assert default_tps(4, [2]) == [2]
    assert default_tps(2, [2, 4]) == [2]


def test_mock_tot_cow_saves_kv_vs_clone() -> None:
    args = parse_args(
        [
            "--backend",
            "mock",
            "--workloads",
            "tot",
            "--branching",
            "4",
            "--trunk-tokens",
            "200",
            "--decode",
            "4",
            "--out",
            "/tmp/forkserve-bench-mock.json",
        ]
    )
    rows = run_mock(args)
    assert len(rows) == 1
    tot = rows[0]
    assert tot.workload == "tot"
    assert tot.trunk_tokens >= 200
    assert tot.peak_kv_tokens == tot.trunk_tokens + sum(tot.residual_tokens)
    clone_tokens = tot.branching * tot.trunk_tokens + sum(tot.residual_tokens)
    assert tot.peak_kv_tokens < clone_tokens
    assert tot.kv_saving > 0.5
    assert tot.m_cow_mib < tot.m_clone_mib


def test_mock_react_known_suffix_hit() -> None:
    args = parse_args(
        [
            "--backend",
            "mock",
            "--workloads",
            "react",
            "--trunk-tokens",
            "80",
            "--decode",
            "2",
            "--out",
            "/tmp/forkserve-bench-mock-react.json",
        ]
    )
    rows = run_mock(args)
    react = rows[0]
    assert react.workload == "react"
    assert react.known_suffix_hit_rate == 1.0
    assert react.branching == 2


def test_format_table_speedup() -> None:
    rows = [
        {
            "system": "vllm_recompute",
            "tp": 2,
            "workload": "tot",
            "e2e_ms": 200.0,
            "fanout_ms": 150.0,
            "ttft_from_obs_ms": 0.0,
            "peak_kv_tokens": 4000,
            "m_cow_mib": 10.0,
            "m_clone_mib": 40.0,
            "kv_saving": 0.75,
        },
        {
            "system": "forkserve",
            "tp": 2,
            "workload": "tot",
            "e2e_ms": 50.0,
            "fanout_ms": 20.0,
            "ttft_from_obs_ms": 0.0,
            "peak_kv_tokens": 1200,
            "m_cow_mib": 10.0,
            "m_clone_mib": 40.0,
            "kv_saving": 0.75,
        },
    ]
    table = format_table(rows)
    assert "forkserve" in table
    assert "4.00x" in table
