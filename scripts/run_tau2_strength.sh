#!/usr/bin/env bash
# Tau2 at alpha 1.5 and 3 on saved coding vectors. Up to 3 models at once.
# Holds alpha_sweep.lock so the transfer pool does not start 14B or Mistral
# on the same GPU.
set -uo pipefail
trap '' HUP
shopt -u huponexit

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="experiments/cross_model/transfer${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-python}"
mkdir -p results/transfer/logs
LOCK=results/transfer/alpha_sweep.lock
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

# model, token budget, minimum free MiB before launch
QUEUE=(
  "qwen3_4b 32768 12000"
  "qwen3_8b 24576 20000"
  "granite_3p3_8b 32768 20000"
  "qwen35_9b 24576 28000"
)
MAX_SLOTS=3
declare -A RUNNING=()
LOG=results/transfer/tau2_strength.log

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' '
}

has_alphas() {
  local model=$1
  "$PY" - "$model" <<'PY'
import json, sys
from pathlib import Path
path = Path(f"results/transfer/{sys.argv[1]}/tau2_alphas.json")
if not path.exists():
    raise SystemExit(1)
keys = set(json.loads(path.read_text(encoding="utf-8")).get("alphas", {}))
raise SystemExit(0 if {"1.5", "3.0"} <= keys else 1)
PY
}

adopt() {
  local pid cmd model
  for pid in /proc/[0-9]*; do
    pid="${pid#/proc/}"
    [[ -r "/proc/${pid}/cmdline" ]] || continue
    cmd=$(tr '\0' ' ' < "/proc/${pid}/cmdline" || true)
    [[ "$cmd" == *"transfer/tau2_alphas.py --models "* ]] || continue
    model=$(sed -n 's/.*--models \([^ ]*\).*/\1/p' <<<"$cmd")
    [[ -n "$model" && "$model" != *","* ]] || continue
    RUNNING["$model"]=$pid
    log "ADOPT ${model} pid=${pid}"
  done
}

start_model() {
  local model=$1 budget=$2
  log "START ${model} alphas=1.5,3 token_budget=${budget} free_mib=$(free_mib)"
  "$PY" -u experiments/cross_model/transfer/tau2_alphas.py \
    --models "$model" --alphas 1.5,3 --random-alphas "" --token-budget "$budget" \
    > "results/transfer/logs/tau2_strength_${model}.log" 2>&1 &
  RUNNING["$model"]=$!
  log "PID ${model} ${RUNNING[$model]}"
}

reap() {
  local model pid
  for model in "${!RUNNING[@]}"; do
    pid=${RUNNING[$model]}
    if kill -0 "$pid" 2>/dev/null; then
      continue
    fi
    if grep -q "^MODEL_DONE ${model}$" "results/transfer/logs/tau2_strength_${model}.log" 2>/dev/null; then
      log "OK ${model}"
    else
      log "FAIL ${model} pid=${pid}"
    fi
    unset "RUNNING[$model]"
  done
}

adopt
kept=()
for item in "${QUEUE[@]}"; do
  model=${item%% *}
  if [[ -n "${RUNNING[$model]:-}" ]] || has_alphas "$model"; then
    log "SKIP ${model}"
    continue
  fi
  kept+=("$item")
done
if ((${#kept[@]})); then
  QUEUE=("${kept[@]}")
else
  QUEUE=()
fi

while true; do
  reap
  while ((${#RUNNING[@]} < MAX_SLOTS)) && ((${#QUEUE[@]} > 0)); do
    item=${QUEUE[0]}
    read -r model budget need <<<"$item"
    free=$(free_mib)
    if (( free < need )); then
      log "WAIT ${model} free_mib=${free} need=${need} running=${#RUNNING[@]}"
      break
    fi
    QUEUE=("${QUEUE[@]:1}")
    start_model "$model" "$budget"
  done
  if ((${#RUNNING[@]} == 0)) && ((${#QUEUE[@]} == 0)); then
    log "DONE strength"
    exit 0
  fi
  names=""
  if ((${#RUNNING[@]})); then
    names=$(printf '%s ' "${!RUNNING[@]}")
  fi
  log "STATUS running=[${names}] queue=${#QUEUE[@]} free_mib=$(free_mib)"
  sleep 30
done
