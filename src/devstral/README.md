# Devstral-Small-2-24B-Instruct-2512 Behavior Scan

This directory contains a fresh Devstral-specific pipeline for the task in `tasks/TASK_TOOLCALL_GENERALIZATION_DEVSTRAL_SMALL_2_24B.md`.

## Files

- `parse_qwen_dataset.py`: parse Qwen3 raw prompts into canonical pairs and re-render them with Devstral's native chat template.
- `run_behavior_scan.py`: load the converted prompts, run batched forward passes, and measure whether the first generated token is `[TOOL_CALLS]`.

## Expected Outputs

- `results/Devstral-Small-2-24B-Instruct-2512/converted_dataset/`
- `results/Devstral-Small-2-24B-Instruct-2512/behavior_scan/`
- `results/Devstral-Small-2-24B-Instruct-2512/logs/`

## Suggested Commands

```bash
export DEVSTRAL_2_24B_PATH=/abs/path/to/Devstral-Small-2-24B-Instruct-2512

python src/devstral/parse_qwen_dataset.py
python src/devstral/run_behavior_scan.py \
  --batch-size 16
```
