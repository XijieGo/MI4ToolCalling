#!/usr/bin/env bash
# Evaluate the locked tool-call vector for one model, or for all seven.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"

if [[ $# -eq 0 ]]; then
  models=(qwen3_4b qwen3_8b qwen35_4b granite_3p3_8b qwen35_9b qwen3_14b mistral_3p2_24b)
else
  models=("$@")
fi

for model in "${models[@]}"; do
  echo "START ${model}"
  "$PY" -u experiments/cross_model/tool_call_vector/run.py --model-key "$model"
  echo "OK ${model}"
done
