#!/usr/bin/env bash
# Apply the locked coding vector. Default schedule keeps three models on the
# one GPU when their weights fit, then runs Qwen3-14B and Mistral alone.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PY="${PYTHON:-/root/miniconda3/envs/qwen35scan/bin/python}"
mkdir -p results/transfer/logs

run_one() {
  local model=$1 budget=$2
  echo "START ${model} token_budget=${budget}"
  "$PY" -u experiments/cross_model/transfer/run.py --model-key "$model" --token-budget "$budget" \
    > "results/transfer/logs/${model}.log" 2>&1
  local status=$?
  if [[ $status -eq 0 ]]; then
    echo "OK ${model}"
  else
    echo "FAIL ${model} exit=${status}"
  fi
  return $status
}

wave() {
  local pids=() names=()
  while [[ $# -gt 0 ]]; do
    local model=$1 budget=$2
    shift 2
    run_one "$model" "$budget" &
    pids+=("$!")
    names+=("$model")
  done
  local fail=0
  local i
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "WAVE_FAIL ${names[$i]}"
      fail=1
    fi
  done
  return $fail
}

if [[ $# -eq 0 ]]; then
  wave qwen3_4b 32768 qwen3_8b 24576 qwen35_4b 24576 || exit 1
  wave granite_3p3_8b 32768 qwen35_9b 24576 || exit 1
  wave qwen3_14b 49152 || exit 1
  wave mistral_3p2_24b 24576 || exit 1
  echo "DONE all"
else
  for model in "$@"; do
    run_one "$model" 0 || exit 1
  done
  echo "DONE all"
fi
