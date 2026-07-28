# Granite 3.3 8B Tool-Call Generalization

This directory contains the Granite-specific first-stage behavior screen required by `tasks/TASK_TOOLCALL_GENERALIZATION_GRANITE_3_3_8B.md`.

## Scripts

- `parse_qwen_dataset.py`: parse the root-level 1500 Qwen clean/corrupt pairs into canonical fields, then re-render each side with Granite's native `chat_template`.
- `run_behavior_scan.py`: run full next-token forward passes on the converted prompts and record whether the first generated token is `<|tool_call|>`.
- `granite_toolcall_common.py`: shared parsing and path helpers.

## Expected outputs

- Converted prompts: `results/granite-3.3-8b-instruct/converted_dataset/`
- Scan artifacts: `results/granite-3.3-8b-instruct/behavior_scan/`
- Logs: `results/granite-3.3-8b-instruct/logs/`

## Typical commands

```bash
conda run -n base python code/granite-3.3-8b-instruct/parse_qwen_dataset.py
conda run -n base python code/granite-3.3-8b-instruct/run_behavior_scan.py --batch-size 16
```
