from forkserve.quality import (
    annotate_quality,
    extract_gsm8k_answer,
    game24_correct,
    gsm8k_correct,
    humaneval_pass,
    score_task,
)


def test_gsm8k_accuracy_is_final_number_not_trace() -> None:
    assert extract_gsm8k_answer("48/2=24, 48+24=72\n#### 72") == "72"
    assert gsm8k_correct("long wording then #### 72", "72")
    assert gsm8k_correct("I think the answer is 72.", "72")
    assert not gsm8k_correct("#### 71", "72")
    scored = score_task("gsm8k", ["#### 72", "#### 0"], ["72", "72"])
    assert scored.metric == "accuracy"
    assert scored.score == 0.5


def test_game24_success_checks_value_and_cards() -> None:
    gold = "Use 4, 8, 3, 12 each once"
    assert game24_correct("(4 + 8) * 3 - 12 = 24", gold)
    assert not game24_correct("(4 + 8) * 3 - 11 = 25", gold)
    assert not game24_correct("1 + 2 + 3 + 18 = 24", gold)  # wrong cards
    scored = score_task("game24", ["(4+8)*3-12"], [gold])
    assert scored.metric == "success_rate"
    assert scored.score == 1.0


def test_humaneval_pass_at_1_runs_hidden_tests() -> None:
    prompt = "def add(a, b):\n"
    good = "    return a + b\n"
    bad = "    return a - b\n"
    tests = "assert add(2, 3) == 5\n"
    assert humaneval_pass(good, tests, prompt)
    assert not humaneval_pass(bad, tests, prompt)
    fenced = "```python\n    return a + b\n```"
    assert humaneval_pass(fenced, tests, prompt)
    scored = score_task("humaneval", [good, bad], [tests, tests], [prompt, prompt])
    assert scored.metric == "pass_at_1"
    assert scored.score == 0.5
    official = (
        "\nMETADATA = {'author': 't'}\n\n"
        "def check(candidate):\n"
        "    assert candidate(2, 3) == 5\n"
    )
    chatty = "</think>\n\n```python\n    return a + b\n```\n<|eot_id|>"
    assert humaneval_pass(chatty, official, prompt)
    assert not humaneval_pass("    return a - b\n", official, prompt)


def test_annotate_scores_each_system_and_delta() -> None:
    rows = [
        {
            "system": "vllm_apc",
            "tp": 2,
            "workload": "gsm8k",
            "decode_texts": ["#### 72", "#### 1"],
            "golds": ["72", "72"],
        },
        {
            "system": "forkserve",
            "tp": 2,
            "workload": "gsm8k",
            "decode_texts": ["#### 72", "#### 72"],
            "golds": ["72", "72"],
        },
    ]
    annotate_quality(rows)
    assert rows[0]["task_score"] == 0.5
    assert rows[1]["task_score"] == 1.0
    assert rows[1]["quality_vs"] == "vllm_apc"
    assert rows[1]["quality_delta"] == 0.5
