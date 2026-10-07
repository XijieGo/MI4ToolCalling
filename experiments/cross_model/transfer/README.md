# Tool-call vector transfer

Follow the [project README](../../../README.md) for installation, input downloads and path configuration. Run the commands below from the repository root.

`run.py` applies the coding vector at the layer and hook defined in `../tool_call_vector/run.py`. It evaluates cross-domain prompt pairs, verb-free requests, and native multi-turn tau2 trajectories. Cross-domain transfer preserves the direction and calibrates its norm to the target domain; verb-free and tau2 interventions use fixed gains.

```bash
bash scripts/run_tool_call_vector.sh qwen3_8b
bash scripts/run_transfer.sh qwen3_8b

# Select individual evaluation arms
python experiments/cross_model/transfer/run.py \
  --model-key qwen3_8b --arms multi_domain,verb_free \
  --token-budget 4096
```

Tau2 call-arm interventions subtract the vector; text-arm interventions add it. The control uses a same-norm Gaussian direction orthogonal to the coding vector, with seed `20260726`. Telecom and Retail system prompts and tool schemas are provided in `templates/`, with the original tau2-bench license. The frozen candidate turns come from the Hugging Face dataset.

`tau2_alphas.py` evaluates additional gains. `tau2_parallel.py` runs and merges independent call/text shards using the same layer, vector and template provenance.

Outputs are written to `results/transfer/<model-key>/summary.json` and `results/transfer/summary.md`.
