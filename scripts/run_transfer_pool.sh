#!/usr/bin/env bash
# Keep up to 3 transfer models running. When one exits, start the next model
# that fits in the free GPU memory. 14B and Mistral wait until the card is
# almost empty so they do not evict a tau2 run.
set -uo pipefail
trap '' HUP
shopt -u huponexit

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PY="${PYTHON:-python}"
mkdir -p results/transfer/logs

# model, token budget, minimum free MiB before launch
QUEUE=(
  "granite_3p3_8b 32768 22000"
  "qwen35_9b 24576 32000"
  "qwen3_14b 49152 40000"
  "mistral_3p2_24b 24576 78000"
)
MAX_SLOTS=3
declare -A RUNNING=()

log() { echo "$(date '+%F %T') $*"; }

free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' '
}

finished() {
  local model=$1
  local summary="results/transfer/${model}/summary.json"
  [[ -f "$summary" ]] || return 1
  "$PY" - "$summary" <<'PY'
import json, sys
report = json.loads(open(sys.argv[1], encoding="utf-8").read())
raise SystemExit(0 if isinstance(report.get("tau2"), dict) else 1)
PY
}

adopt() {
  local pid cmd model
  for pid in /proc/[0-9]*; do
    pid="${pid#/proc/}"
    [[ -r "/proc/${pid}/cmdline" ]] || continue
    cmd=$(tr '\0' ' ' < "/proc/${pid}/cmdline" || true)
    [[ "$cmd" == *"experiments/cross_model/transfer/run.py --model-key "* ]] || continue
    model=$(sed -n 's/.*--model-key \([^ ]*\).*/\1/p' <<<"$cmd")
    [[ -n "$model" ]] || continue
    RUNNING["$model"]=$pid
    log "ADOPT ${model} pid=${pid}"
  done
}

start_model() {
  local model=$1 budget=$2
  log "START ${model} token_budget=${budget} free_mib=$(free_mib)"
  "$PY" -u experiments/cross_model/transfer/run.py \
    --model-key "$model" --token-budget "$budget" \
    > "results/transfer/logs/${model}.log" 2>&1 &
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
    if grep -q '^MODEL_DONE ' "results/transfer/logs/${model}.log" 2>/dev/null || finished "$model"; then
      log "OK ${model}"
    else
      log "FAIL ${model} pid=${pid}"
    fi
    unset "RUNNING[$model]"
  done
}

drop_finished_from_queue() {
  local item model kept=()
  for item in "${QUEUE[@]}"; do
    model=${item%% *}
    if [[ -n "${RUNNING[$model]:-}" ]] || finished "$model"; then
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
}

adopt
drop_finished_from_queue

while true; do
  reap
  while ((${#RUNNING[@]} < MAX_SLOTS)) && ((${#QUEUE[@]} > 0)); do
    item=${QUEUE[0]}
    read -r model budget need <<<"$item"
    free=$(free_mib)
    if [[ -f results/transfer/alpha_sweep.lock ]] && [[ "$model" == "qwen3_14b" || "$model" == "mistral_3p2_24b" ]]; then
      log "WAIT ${model} alpha_sweep_running free_mib=${free} running=${#RUNNING[@]}"
      break
    fi
    if (( free < need )); then
      log "WAIT ${model} free_mib=${free} need=${need} running=${#RUNNING[@]}"
      break
    fi
    QUEUE=("${QUEUE[@]:1}")
    start_model "$model" "$budget"
    # One launch per cycle. The next model must see the memory this one
    # actually allocated, otherwise 14B and Mistral start together.
    break
  done
  if ((${#RUNNING[@]} == 0)) && ((${#QUEUE[@]} == 0)); then
    log "DONE all"
    exit 0
  fi
  names=""
  if ((${#RUNNING[@]})); then
    names=$(printf '%s ' "${!RUNNING[@]}")
  fi
  log "STATUS running=[${names}] queue=${#QUEUE[@]} free_mib=$(free_mib)"
  sleep 30
done
