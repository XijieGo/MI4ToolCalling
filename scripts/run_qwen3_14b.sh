#!/usr/bin/env bash
# Runner for all Qwen3-14B mechanism experiments:
#   1. Scaffold Ablation
#   2. Vector Formation & Transcoder Features
#   3. Downstream Readout Mechanism

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-/root/miniconda3/envs/qwen35scan/bin/python}"

echo "=========================================================="
echo "Qwen3-14B Stage 1/3: Scaffold Component Ablation"
echo "=========================================================="
"$PY" -u experiments/qwen3_14b/scaffold_ablation/run.py \
  --model-path /root/autodl-tmp/Qwen/Qwen3-14B \
  --dataset-root datasets/qwen3_14b/pair \
  --output-root results/qwen3_14b/scaffold_ablation \
  --batch-size 8 \
  --device cuda

echo "=========================================================="
echo "Qwen3-14B Stage 2/3: Vector Formation & Transcoder Features"
echo "=========================================================="
"$PY" -u experiments/qwen3_14b/formation_transcoder/run.py \
  --model-path /root/autodl-tmp/Qwen/Qwen3-14B \
  --transcoder-dir /root/autodl-tmp/Transcoder/Qwen3-14B \
  --vector-path results/transfer/qwen3_14b/coding_vector.pt \
  --dataset-root datasets/qwen3_14b/pair \
  --output-root results/qwen3_14b/formation_transcoder \
  --batch-size 4 \
  --device cuda

echo "=========================================================="
echo "Qwen3-14B Stage 3/3: Downstream Readout Mechanism"
echo "=========================================================="
"$PY" -u experiments/qwen3_14b/downstream_readout/run.py \
  --model-path /root/autodl-tmp/Qwen/Qwen3-14B \
  --transcoder-dir /root/autodl-tmp/Transcoder/Qwen3-14B \
  --vector-path results/transfer/qwen3_14b/coding_vector.pt \
  --dataset-root datasets/qwen3_14b/pair \
  --output-root results/qwen3_14b/downstream_readout \
  --batch-size 4 \
  --device cuda

echo "=========================================================="
echo "Qwen3-14B all mechanism experiments completed successfully!"
echo "Results located in: results/qwen3_14b/"
echo "=========================================================="
