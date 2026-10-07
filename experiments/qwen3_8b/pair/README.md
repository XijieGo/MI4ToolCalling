# Qwen3-8B localization control

Follow the [project README](../../../README.md) for installation and input downloads. Run the commands below from the repository root.

`run.py` fits a mean clean-minus-corrupt direction on `datasets/qwen3_8b/controlled/train` and evaluates held-out patches on the code-domain control set. The primary seven-model vector experiments use the [cross-model vector runner](../../cross_model/tool_call_vector/README.md).

```bash
python experiments/qwen3_8b/pair/run.py \
  --model-path external/models/Qwen3-8B \
  --layers 24 \
  --direction-layer 24 \
  --output-root runs/qwen3_8b_pair
```

Use `--model-path` to select an existing local model. Outputs are written to the directory specified by `--output-root`.
