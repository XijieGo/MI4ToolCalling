# Cross-model experiments

Follow the [project README](../../README.md) for installation, input downloads and path configuration. Run the commands below from the repository root.

The primary entry points are:

- [Tool-call vector](tool_call_vector/README.md): training-split vector estimation and held-out sufficiency / necessity.
- [Transfer](transfer/README.md): cross-domain, verb-free and tau2 evaluations with the fitted coding vector.
- `recompute_mechanisms.py`: fixed-layer scaffold, formation and readout measurements.
- `measure_mlp_attn_and_max_attn.py`: component-write and attention measurements.
- `run_native_localization.py`: prediction-position and token-alignment localization controls.
- `summarize_core_table.py` and `summarize_cross_scale.py`: cross-model tables from generated measurements.

Native Mistral prompts use their stored input IDs; text-native families encode their frozen chat renderings with `add_special_tokens=False`.

The model-specific formation runners write layer trajectories and component measurements. `recompute_mechanisms.py` computes feature contributions from the released `.pt` Transcoders. For Qwen3 `.safetensors` Transcoders, provide the per-model feature runner's JSON output with `--feature-summary` when assembling cross-model measurements.

```bash
python experiments/cross_model/recompute_mechanisms.py \
  --model-key qwen35_4b --stage formation --run-dir runs/qwen35_4b
```

The runner initializes the fixed layer configuration in the specified run directory and checks it on subsequent invocations.

```bash
python experiments/cross_model/run_native_localization.py \
  --model-key granite_3p3_8b --dry-run

python experiments/cross_model/run_native_localization.py \
  --model-key mistral_3p2_24b \
  --model-path external/models/Mistral-Small-3.2-24B-Instruct-2506 \
  --layers 25 --direction-layer 25 \
  --output-root runs/mistral_pair
```

`prediction` uses the selected pairs at the final input position. `strict` selects pairs with equal token lengths and exactly one changed token. Each output records its input split, alignment policy and hook convention.
