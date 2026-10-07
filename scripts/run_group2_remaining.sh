#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"
LOG_DIR="$REPO_ROOT/results/formation_readout/logs"
mkdir -p "$LOG_DIR"

echo "=== Launching Group 2 Remaining (qwen35_4b, qwen35_9b, mistral_3p2_24b) ==="
date

CUDA_VISIBLE_DEVICES=5 "$PYTHON" "$REPO_ROOT/experiments/cross_model/measure_mlp_attn_and_max_attn.py" --model-key qwen35_4b --device cuda --max-pairs 50 > "$LOG_DIR/qwen35_4b.log" 2>&1 &
PID_Q35_4=$!

CUDA_VISIBLE_DEVICES=6 "$PYTHON" "$REPO_ROOT/experiments/cross_model/measure_mlp_attn_and_max_attn.py" --model-key qwen35_9b --device cuda --max-pairs 50 > "$LOG_DIR/qwen35_9b.log" 2>&1 &
PID_Q35_9=$!

CUDA_VISIBLE_DEVICES=7 "$PYTHON" "$REPO_ROOT/experiments/cross_model/measure_mlp_attn_and_max_attn.py" --model-key mistral_3p2_24b --device cuda --max-pairs 50 > "$LOG_DIR/mistral_3p2_24b.log" 2>&1 &
PID_M24=$!

echo "Remaining PIDs: qwen35_4b=$PID_Q35_4, qwen35_9b=$PID_Q35_9, mistral_3p2_24b=$PID_M24"

wait $PID_Q35_4
wait $PID_Q35_9
wait $PID_M24

echo "=== Group 2 Remaining all finished ==="
date
