#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"
LOG_DIR="$REPO_ROOT/results/formation_readout/logs"
mkdir -p "$LOG_DIR"

echo "=== Launching Group 2 (MLP/Attn and Max Attn) on GPUs 4, 5, 6, 7 ==="
date

# GPU 4: qwen3_4b and qwen3_8b
CUDA_VISIBLE_DEVICES=4 "$PYTHON" "$REPO_ROOT/experiments/cross_model/measure_mlp_attn_and_max_attn.py" --model-key qwen3_4b --device cuda --max-pairs 50 > "$LOG_DIR/qwen3_4b.log" 2>&1 &
PID_Q4=$!

CUDA_VISIBLE_DEVICES=4 "$PYTHON" "$REPO_ROOT/experiments/cross_model/measure_mlp_attn_and_max_attn.py" --model-key qwen3_8b --device cuda --max-pairs 50 > "$LOG_DIR/qwen3_8b.log" 2>&1 &
PID_Q8=$!

# GPU 5: qwen3_14b and qwen35_4b
CUDA_VISIBLE_DEVICES=5 "$PYTHON" "$REPO_ROOT/experiments/cross_model/measure_mlp_attn_and_max_attn.py" --model-key qwen3_14b --device cuda --max-pairs 50 > "$LOG_DIR/qwen3_14b.log" 2>&1 &
PID_Q14=$!

CUDA_VISIBLE_DEVICES=5 "$PYTHON" "$REPO_ROOT/experiments/cross_model/measure_mlp_attn_and_max_attn.py" --model-key qwen35_4b --device cuda --max-pairs 50 > "$LOG_DIR/qwen35_4b.log" 2>&1 &
PID_Q35_4=$!

# GPU 6: qwen35_9b and granite_3p3_8b
CUDA_VISIBLE_DEVICES=6 "$PYTHON" "$REPO_ROOT/experiments/cross_model/measure_mlp_attn_and_max_attn.py" --model-key qwen35_9b --device cuda --max-pairs 50 > "$LOG_DIR/qwen35_9b.log" 2>&1 &
PID_Q35_9=$!

CUDA_VISIBLE_DEVICES=6 "$PYTHON" "$REPO_ROOT/experiments/cross_model/measure_mlp_attn_and_max_attn.py" --model-key granite_3p3_8b --device cuda --max-pairs 50 > "$LOG_DIR/granite_3p3_8b.log" 2>&1 &
PID_G8=$!

# GPU 7: mistral_3p2_24b
CUDA_VISIBLE_DEVICES=7 "$PYTHON" "$REPO_ROOT/experiments/cross_model/measure_mlp_attn_and_max_attn.py" --model-key mistral_3p2_24b --device cuda --max-pairs 50 > "$LOG_DIR/mistral_3p2_24b.log" 2>&1 &
PID_M24=$!

echo "Group 2 PIDs: qwen3_4b=$PID_Q4, qwen3_8b=$PID_Q8, qwen3_14b=$PID_Q14, qwen35_4b=$PID_Q35_4, qwen35_9b=$PID_Q35_9, granite_3p3_8b=$PID_G8, mistral_3p2_24b=$PID_M24"

wait $PID_Q4
wait $PID_Q8
wait $PID_Q14
wait $PID_Q35_4
wait $PID_Q35_9
wait $PID_G8
wait $PID_M24

echo "=== Group 2 all finished ==="
date
