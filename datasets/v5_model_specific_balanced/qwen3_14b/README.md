# Qwen3-14B v5 dataset

This dataset is target-model-specific and behavior-screened.

- Pairs: `500` (`200` train, `300` held-out).
- Native first-token call marker: `<tool_call>` (ID `151657`).
- Admission rule: clean marker is top-1 and corrupt marker is not top-1.
- Verb counts are balanced independently on clean and corrupt sides in every split.
- `manifest.jsonl` contains prompt hashes and the complete screening metrics for every selected pair.
