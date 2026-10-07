#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"
LOG_DIR="$REPO_ROOT/results/tool_call_vector/logs"
mkdir -p "$LOG_DIR"

echo "=== Launching 7 tool_call_vector runs on GPUs 0-6 ==="
date

CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$REPO_ROOT/experiments/cross_model/tool_call_vector/run.py" --model-key qwen3_4b > "$LOG_DIR/qwen3_4b.log" 2>&1 &
PID_0=$!
CUDA_VISIBLE_DEVICES=1 "$PYTHON" "$REPO_ROOT/experiments/cross_model/tool_call_vector/run.py" --model-key qwen3_8b > "$LOG_DIR/qwen3_8b.log" 2>&1 &
PID_1=$!
CUDA_VISIBLE_DEVICES=2 "$PYTHON" "$REPO_ROOT/experiments/cross_model/tool_call_vector/run.py" --model-key qwen3_14b > "$LOG_DIR/qwen3_14b.log" 2>&1 &
PID_2=$!
CUDA_VISIBLE_DEVICES=3 "$PYTHON" "$REPO_ROOT/experiments/cross_model/tool_call_vector/run.py" --model-key qwen35_4b > "$LOG_DIR/qwen35_4b.log" 2>&1 &
PID_3=$!
CUDA_VISIBLE_DEVICES=4 "$PYTHON" "$REPO_ROOT/experiments/cross_model/tool_call_vector/run.py" --model-key qwen35_9b > "$LOG_DIR/qwen35_9b.log" 2>&1 &
PID_4=$!
CUDA_VISIBLE_DEVICES=5 "$PYTHON" "$REPO_ROOT/experiments/cross_model/tool_call_vector/run.py" --model-key granite_3p3_8b > "$LOG_DIR/granite_3p3_8b.log" 2>&1 &
PID_5=$!
CUDA_VISIBLE_DEVICES=6 "$PYTHON" "$REPO_ROOT/experiments/cross_model/tool_call_vector/run.py" --model-key mistral_3p2_24b > "$LOG_DIR/mistral_3p2_24b.log" 2>&1 &
PID_6=$!

echo "PIDs: qwen3_4b=$PID_0, qwen3_8b=$PID_1, qwen3_14b=$PID_2, qwen35_4b=$PID_3, qwen35_9b=$PID_4, granite_3p3_8b=$PID_5, mistral_3p2_24b=$PID_6"

wait $PID_0
wait $PID_1
wait $PID_2
wait $PID_3
wait $PID_4
wait $PID_5
wait $PID_6

echo "=== All 7 tool_call_vector runs finished ==="
date
