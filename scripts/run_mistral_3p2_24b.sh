#!/usr/bin/env bash
# Runner for all Mistral-3.2-24B mechanism experiments:
#   1. Scaffold Ablation (Table 4)
#   2. Vector Formation & Write Decomposition (Table 5, Figure 2)
#   3. Downstream Readout Mechanism (Section 6, Figure 3)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-/root/miniconda3/envs/qwen35scan/bin/python}"

echo "=========================================================="
echo "Mistral-3.2-24B Stage 1/3: Scaffold Component Ablation"
echo "=========================================================="
"$PY" -u experiments/mistral_3p2_24b/scaffold_ablation/run.py \
  --model-path /root/autodl-tmp/Mistral-Small-3.2-24B-Instruct-2506 \
  --dataset-root datasets/mistral_3p2_24b/pair \
  --output-dir results/mistral_3p2_24b/scaffold_ablation \
  --batch-size 4

echo "=========================================================="
echo "Mistral-3.2-24B Stage 2/3: Vector Formation & Write Decomposition"
echo "=========================================================="
"$PY" -u experiments/mistral_3p2_24b/formation_transcoder/run.py \
  --model-path /root/autodl-tmp/Mistral-Small-3.2-24B-Instruct-2506 \
  --vector-path results/transfer/mistral_3p2_24b/coding_vector.pt \
  --dataset-root datasets/mistral_3p2_24b/pair \
  --output-root results/mistral_3p2_24b/formation_transcoder \
  --batch-size 4

echo "=========================================================="
echo "Mistral-3.2-24B Stage 3/3: Downstream Readout Mechanism"
echo "=========================================================="
"$PY" -u experiments/mistral_3p2_24b/downstream_readout/run.py \
  --model-path /root/autodl-tmp/Mistral-Small-3.2-24B-Instruct-2506 \
  --vector-path results/transfer/mistral_3p2_24b/coding_vector.pt \
  --dataset-root datasets/mistral_3p2_24b/pair \
  --output-root results/mistral_3p2_24b/downstream_readout \
  --batch-size 2

echo "=========================================================="
echo "Mistral-3.2-24B all mechanism experiments completed successfully!"
echo "Results located in: results/mistral_3p2_24b/"
echo "=========================================================="
