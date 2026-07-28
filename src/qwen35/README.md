# Qwen3.5-9B Dataset Conversion

This directory contains the Qwen3.5-specific dataset conversion step for the existing Qwen3 clean/corrupt tool-call pairs.

## Script

- `parse_qwen_dataset.py`: parse the existing Qwen3 raw prompts, preserve each pair's user content and tool schema, and rerender both sides with Qwen3.5-9B's native `chat_template`.

## Outputs

- Converted prompts: `results/Qwen3.5-9B/converted_dataset/`
- Key files:
  - `canonical_pairs.jsonl`
  - `manifest.jsonl`
  - `conversion_summary.json`

## Notes

- The source dataset under `datasets/` is not modified.
- By default the script uses `enable_thinking=False` so the rendered assistant prefix stays closer to the original first-token decision setup.

## Typical command

```bash
conda run -n base python code/Qwen3.5-9B/parse_qwen_dataset.py
```
