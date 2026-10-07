#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"
LOG_DIR="$REPO_ROOT/results/transfer/logs"
mkdir -p "$LOG_DIR"

echo "=== Launching tau2 arm on GPUs 0, 2, 4, 5, 6 ==="
date

CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$REPO_ROOT/experiments/cross_model/transfer/run.py" --model-key qwen3_4b --arms tau2 --device cuda > "$LOG_DIR/qwen3_4b_tau2.log" 2>&1 &
PID_0=$!
CUDA_VISIBLE_DEVICES=2 "$PYTHON" "$REPO_ROOT/experiments/cross_model/transfer/run.py" --model-key qwen3_14b --arms tau2 --device cuda > "$LOG_DIR/qwen3_14b_tau2.log" 2>&1 &
PID_2=$!
CUDA_VISIBLE_DEVICES=4 "$PYTHON" "$REPO_ROOT/experiments/cross_model/transfer/run.py" --model-key qwen35_9b --arms tau2 --token-budget 16384 --device cuda > "$LOG_DIR/qwen35_9b_tau2.log" 2>&1 &
PID_4=$!
CUDA_VISIBLE_DEVICES=5 "$PYTHON" "$REPO_ROOT/experiments/cross_model/transfer/run.py" --model-key granite_3p3_8b --arms tau2 --device cuda > "$LOG_DIR/granite_3p3_8b_tau2.log" 2>&1 &
PID_5=$!
CUDA_VISIBLE_DEVICES=6 "$PYTHON" "$REPO_ROOT/experiments/cross_model/transfer/run.py" --model-key mistral_3p2_24b --arms tau2 --device cuda > "$LOG_DIR/mistral_3p2_24b_tau2.log" 2>&1 &
PID_6=$!

echo "PIDs: qwen3_4b=$PID_0, qwen3_14b=$PID_2, qwen35_9b=$PID_4, granite_3p3_8b=$PID_5, mistral_3p2_24b=$PID_6"

wait $PID_0
wait $PID_2
wait $PID_4
wait $PID_5
wait $PID_6

echo "=== tau2 arms finished ==="
date
