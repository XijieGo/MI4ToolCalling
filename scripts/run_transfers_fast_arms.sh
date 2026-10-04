#!/bin/bash
set -e

REPO_ROOT="/home/xijie/MI4Toolcalling"
PYTHON="/home/xijie/llm/venvs/qwen35-transcoder/bin/python"
LOG_DIR="$REPO_ROOT/results/transfer/logs"
mkdir -p "$LOG_DIR"

echo "=== Launching multi_domain and verb_free arms on GPUs ==="
date

CUDA_VISIBLE_DEVICES=0 $PYTHON "$REPO_ROOT/experiments/cross_model/transfer/run.py" --model-key qwen3_4b --arms multi_domain,verb_free --device cuda > "$LOG_DIR/qwen3_4b_fast.log" 2>&1 &
PID_0=$!
CUDA_VISIBLE_DEVICES=2 $PYTHON "$REPO_ROOT/experiments/cross_model/transfer/run.py" --model-key qwen3_14b --arms multi_domain,verb_free --device cuda > "$LOG_DIR/qwen3_14b_fast.log" 2>&1 &
PID_2=$!
CUDA_VISIBLE_DEVICES=4 $PYTHON "$REPO_ROOT/experiments/cross_model/transfer/run.py" --model-key qwen35_9b --arms multi_domain,verb_free --device cuda > "$LOG_DIR/qwen35_9b_fast.log" 2>&1 &
PID_4=$!
CUDA_VISIBLE_DEVICES=5 $PYTHON "$REPO_ROOT/experiments/cross_model/transfer/run.py" --model-key granite_3p3_8b --arms multi_domain,verb_free --device cuda > "$LOG_DIR/granite_3p3_8b_fast.log" 2>&1 &
PID_5=$!

echo "PIDs: qwen3_4b=$PID_0, qwen3_14b=$PID_2, qwen35_9b=$PID_4, granite_3p3_8b=$PID_5"

wait $PID_0
wait $PID_2
wait $PID_4
wait $PID_5

echo "=== Fast arms finished ==="
date
