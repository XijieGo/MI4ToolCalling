#!/usr/bin/env bash
# Orchestrator for all remaining model experiments
# Runs qwen35_9b and mistral_3p2_24b as GPU memory frees up,
# and generates the final cross-scale synthesis report.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-/root/miniconda3/envs/qwen35scan/bin/python}"

echo "Waiting for current tasks to complete..."
while pgrep -f "experiments/qwen35_4b" >/dev/null || pgrep -f "experiments/granite_3p3_8b" >/dev/null; do
  sleep 15
done

echo "=========================================================="
echo "Starting Qwen3.5-9B Mechanism Suite"
echo "=========================================================="
bash scripts/run_qwen35_9b.sh

echo "=========================================================="
echo "Starting Mistral-3.2-24B Mechanism Suite"
echo "=========================================================="
bash scripts/run_mistral_3p2_24b.sh

echo "=========================================================="
echo "Generating Final Cross-Model Mechanistic Synthesis"
echo "=========================================================="
"$PY" experiments/cross_model/summarize_cross_scale.py

echo "=========================================================="
echo "ALL EXPERIMENTS ACROSS ALL 7 MODELS COMPLETED!"
echo "=========================================================="
