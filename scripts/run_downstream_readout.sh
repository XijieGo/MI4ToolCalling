#!/usr/bin/env bash
# Run Downstream Readout Mechanism on Qwen3-8B (Sec 6.1, 6.2, Figure 3)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-/root/miniconda3/envs/qwen35scan/bin/python}"

echo "=== Running Downstream Readout Analysis on Qwen3-8B ==="
"$PY" -u experiments/qwen3_8b/downstream_readout/run.py \
  --model-path /root/autodl-tmp/Qwen/Qwen3-8B \
  --transcoder-dir /root/autodl-tmp/Transcoder/Qwen3-8B \
  --vector-path results/transfer/qwen3_8b/coding_vector.pt \
  --dataset-root datasets/qwen3_8b/pair \
  --output-root results/qwen3_8b/downstream_readout \
  --batch-size 4 \
  --device cuda

echo "=== Downstream Readout Analysis Complete ==="
