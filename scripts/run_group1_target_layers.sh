#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"
LOG_DIR="$REPO_ROOT/results/transfer/logs_target_layers"
mkdir -p "$LOG_DIR"

echo "=== Resuming Group 1 (tau2) on GPUs 0, 1, 2 ==="
date

# GPU 0: granite_3p3_8b (L35)
(
    echo "--- Starting granite_3p3_8b tau2 (L35) ---"
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$REPO_ROOT/experiments/cross_model/transfer/run.py" --model-key granite_3p3_8b --arms tau2 --device cuda
    echo "--- granite_3p3_8b L35 tau2 finished ---"
) >> "$LOG_DIR/granite_3p3_8b_L35.log" 2>&1 &
PID_0=$!

# GPU 1: qwen35_4b (L31)
(
    echo "--- Starting qwen35_4b tau2 (L31) ---"
    CUDA_VISIBLE_DEVICES=1 "$PYTHON" "$REPO_ROOT/experiments/cross_model/transfer/run.py" --model-key qwen35_4b --token-budget 16384 --arms tau2 --device cuda
    echo "--- qwen35_4b L31 tau2 finished ---"
) >> "$LOG_DIR/qwen35_4b_L31.log" 2>&1 &
PID_1=$!

# GPU 2: qwen35_9b (L31)
(
    echo "--- Starting qwen35_9b tau2 (L31) ---"
    CUDA_VISIBLE_DEVICES=2 "$PYTHON" "$REPO_ROOT/experiments/cross_model/transfer/run.py" --model-key qwen35_9b --token-budget 16384 --arms tau2 --device cuda
    echo "--- qwen35_9b L31 tau2 finished ---"
) >> "$LOG_DIR/qwen35_9b_L31.log" 2>&1 &
PID_2=$!

echo "Group 1 PIDs: granite_3p3_8b=$PID_0, qwen35_4b=$PID_1, qwen35_9b=$PID_2"

wait $PID_0
wait $PID_1
wait $PID_2

echo "=== Group 1 all finished ==="
date
