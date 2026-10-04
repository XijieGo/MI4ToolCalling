#!/usr/bin/env bash
# Master runner for all Qwen3-8B mechanism experiments on the final 500-pair dataset:
#   1. Scaffold Ablation (Sec 5.1 & Table 4)
#   2. Vector Formation & Transcoder Features (Sec 5.2, 5.3, Figure 2, Table 5)
#   3. Downstream Readout Mechanism (Sec 6.1, 6.2, Figure 3)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=========================================================="
echo "Stage 1/3: Scaffold Component Ablation (Table 4)"
echo "=========================================================="
bash scripts/run_scaffold_ablation.sh

echo "=========================================================="
echo "Stage 2/3: Vector Formation & Transcoder Features (Figure 2, Table 5)"
echo "=========================================================="
bash scripts/run_formation_transcoder.sh

echo "=========================================================="
echo "Stage 3/3: Downstream Readout Mechanism (Figure 3)"
echo "=========================================================="
bash scripts/run_downstream_readout.sh

echo "=========================================================="
echo "All Qwen3-8B mechanism experiments completed successfully!"
echo "Results located in: results/qwen3_8b/"
echo "=========================================================="
