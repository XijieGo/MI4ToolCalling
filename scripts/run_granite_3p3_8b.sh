#!/usr/bin/env bash
# Runner for all Granite-3.3-8B mechanism experiments:
#   1. Scaffold Ablation (Table 4)
#   2. Vector Formation & Write Decomposition (Table 5, Figure 2)
#   3. Downstream Readout Mechanism (Section 6, Figure 3)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-/root/miniconda3/envs/qwen35scan/bin/python}"

echo "=========================================================="
echo "Granite-3.3-8B Stage 1/3: Scaffold Component Ablation"
echo "=========================================================="
"$PY" -u experiments/granite_3p3_8b/scaffold_ablation/run.py \
  --model-path /root/autodl-tmp/Granite/granite-3.3-8b-instruct \
  --dataset-root datasets/granite_3p3_8b/pair \
  --output-dir results/granite_3p3_8b/scaffold_ablation \
  --batch-size 8

echo "=========================================================="
echo "Granite-3.3-8B Stage 2/3: Vector Formation & Write Decomposition"
echo "=========================================================="
"$PY" -u experiments/granite_3p3_8b/formation_transcoder/run.py \
  --model-path /root/autodl-tmp/Granite/granite-3.3-8b-instruct \
  --vector-path results/transfer/granite_3p3_8b/coding_vector.pt \
  --dataset-root datasets/granite_3p3_8b/pair \
  --output-root results/granite_3p3_8b/formation_transcoder \
  --batch-size 8

echo "=========================================================="
echo "Granite-3.3-8B Stage 3/3: Downstream Readout Mechanism"
echo "=========================================================="
"$PY" -u experiments/granite_3p3_8b/downstream_readout/run.py \
  --model-path /root/autodl-tmp/Granite/granite-3.3-8b-instruct \
  --vector-path results/transfer/granite_3p3_8b/coding_vector.pt \
  --dataset-root datasets/granite_3p3_8b/pair \
  --output-root results/granite_3p3_8b/downstream_readout \
  --batch-size 4

echo "=========================================================="
echo "Granite-3.3-8B all mechanism experiments completed successfully!"
echo "Results located in: results/granite_3p3_8b/"
echo "=========================================================="
