#!/usr/bin/env bash
# Run Scaffold Component Ablation for Qwen3-8B (Section 5.1 & Table 4)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-/root/miniconda3/envs/qwen35scan/bin/python}"

echo "=== Running Scaffold Ablation on Qwen3-8B ==="
"$PY" -u experiments/qwen3_8b/scaffold_ablation/run.py \
  --model-path /root/autodl-tmp/Qwen/Qwen3-8B \
  --dataset-root datasets/qwen3_8b/pair \
  --output-root results/qwen3_8b/scaffold_ablation \
  --batch-size 8 \
  --device cuda

echo "=== Scaffold Ablation Complete ==="
