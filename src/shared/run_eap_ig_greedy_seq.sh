#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNNER="$PROJECT_ROOT/src/shared/run_eap_ig_8b.py"
SOURCE_DIR="$PROJECT_ROOT/results/8B/eap_ig"
BASE_OUT="$PROJECT_ROOT/results/8B/eap_ig_greedy_runs"
SCORES="$SOURCE_DIR/eap_ig_scores.pt"
DATASET="$SOURCE_DIR/eap_dataset_100.csv"

mkdir -p "$BASE_OUT"

for k in 10000 20000 50000; do
  OUT="$BASE_OUT/top${k}"
  mkdir -p "$OUT"
  cp -f "$SCORES" "$OUT/eap_ig_scores.pt"
  cp -f "$DATASET" "$OUT/eap_dataset_100.csv"

  echo "[greedy] start top-${k}"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -u "$RUNNER" \
      --evaluate-only \
      --selection-method greedy \
      --n-samples 100 \
      --batch-size 1 \
      --ig-steps 5 \
      --topk-list "$k" \
      --primary-topk "$k" \
      --remove-topk-list "$k" \
      --out-dir "$OUT" \
      --skip-visualization 2>&1 | tee "$OUT/run.log"
  echo "[greedy] done top-${k}"
done
