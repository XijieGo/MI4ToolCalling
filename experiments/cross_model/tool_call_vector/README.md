# Tool-call vector

Follow the [project README](../../../README.md) for installation, input downloads and path configuration. Run the commands below from the repository root.

The vector is the training-split mean of clean minus corrupt at the final input token. Sufficiency adds it to analysis prompts; necessity subtracts it from execution prompts.

| Model key | Layer | Hook | Training / held-out pairs |
|---|---:|---|---:|
| `qwen3_4b` | 26 | pre | 300 / 200 |
| `qwen3_8b` | 24 | pre | 300 / 200 |
| `qwen3_14b` | 34 | pre | 300 / 200 |
| `qwen35_4b` | 31 | pre | 300 / 200 |
| `qwen35_9b` | 31 | pre | 300 / 200 |
| `granite_3p3_8b` | 35 | pre | 300 / 200 |
| `mistral_3p2_24b` | 25 | pre | 300 / 200 |

Layers use zero-based decoder indices. `pre` denotes the block-input residual. Inputs are loaded from `datasets/<model-key>/pair/`.

```bash
# One model
bash scripts/run_tool_call_vector.sh qwen3_8b

# All seven models, sequentially
bash scripts/run_tool_call_vector.sh

# Set pair counts and the activation budget
python experiments/cross_model/tool_call_vector/run.py \
  --model-key qwen3_8b --max-train-pairs 4 --max-heldout-pairs 4 \
  --token-budget 4096 --output-root runs/vector_example
```

Full runs write to `results/tool_call_vector/<model-key>/`, with a combined table under `results/tool_call_vector/`.
