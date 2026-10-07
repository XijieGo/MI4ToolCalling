#!/usr/bin/env bash
# Runner for all Qwen3.5-9B mechanism experiments:
#   1. Scaffold Ablation (Table 4)
#   2. Vector Formation & Feature Analysis (Table 5, Figure 2)
#   3. Downstream Readout Mechanism (Section 6, Figure 3)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"

echo "=========================================================="
echo "Qwen3.5-9B Stage 1/3: Scaffold Component Ablation"
echo "=========================================================="
"$PY" -u experiments/qwen35_9b/scaffold_ablation/run.py \
  --dataset-root datasets/qwen35_9b/pair \
  --output-dir results/qwen35_9b/scaffold_ablation \
  --batch-size 8

echo "=========================================================="
echo "Qwen3.5-9B Stage 2/3: Vector Formation & Feature Analysis"
echo "=========================================================="
"$PY" -u experiments/qwen35_9b/formation_transcoder/run.py \
  --vector-path results/transfer/qwen35_9b/coding_vector.pt \
  --dataset-root datasets/qwen35_9b/pair \
  --output-root results/qwen35_9b/formation_transcoder \
  --batch-size 4

echo "=========================================================="
echo "Qwen3.5-9B Stage 3/3: Downstream Readout Mechanism"
echo "=========================================================="
"$PY" -u experiments/qwen35_9b/downstream_readout/run.py \
  --vector-path results/transfer/qwen35_9b/coding_vector.pt \
  --dataset-root datasets/qwen35_9b/pair \
  --output-root results/qwen35_9b/downstream_readout \
  --batch-size 4

echo "=========================================================="
echo "Qwen3.5-9B all mechanism experiments completed successfully!"
echo "Results located in: results/qwen35_9b/"
echo "=========================================================="
