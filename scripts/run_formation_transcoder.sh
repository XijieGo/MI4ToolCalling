#!/usr/bin/env bash
# Run Vector Formation Trajectory & Transcoder Feature Analysis (Sec 5.2, 5.3, Figure 2, Table 5)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-/root/miniconda3/envs/qwen35scan/bin/python}"

echo "=== Running Vector Formation & Transcoder Feature Analysis on Qwen3-8B ==="
"$PY" -u experiments/qwen3_8b/formation_transcoder/run.py \
  --model-path /root/autodl-tmp/Qwen/Qwen3-8B \
  --transcoder-dir /root/autodl-tmp/Transcoder/Qwen3-8B \
  --vector-path results/transfer/qwen3_8b/coding_vector.pt \
  --dataset-root datasets/qwen3_8b/pair \
  --output-root results/qwen3_8b/formation_transcoder \
  --batch-size 8 \
  --device cuda

echo "=== Formation & Transcoder Analysis Complete ==="
