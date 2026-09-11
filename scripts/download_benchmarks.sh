#!/usr/bin/env bash
# Download GSM8K, HumanEval, and Game-of-24 into ~/benchmarks (or $BENCHMARKS_DIR).
set -euo pipefail
ROOT="${BENCHMARKS_DIR:-${FORKSERVE_BENCHMARKS:-$HOME/benchmarks}}"
mkdir -p "$ROOT/gsm8k" "$ROOT/humaneval" "$ROOT/game24"

curl -fsSL -o "$ROOT/gsm8k/test.jsonl" \
  "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
curl -fsSL -o "$ROOT/gsm8k/train.jsonl" \
  "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"

curl -fsSL -o "$ROOT/humaneval/HumanEval.jsonl.gz" \
  "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
gzip -dfk "$ROOT/humaneval/HumanEval.jsonl.gz"

curl -fsSL -o "$ROOT/game24/24.csv" \
  "https://raw.githubusercontent.com/princeton-nlp/tree-of-thought-llm/master/src/tot/data/24/24.csv"

find "$ROOT" -type f -exec ls -lh {} \;
echo "wrote $ROOT"
