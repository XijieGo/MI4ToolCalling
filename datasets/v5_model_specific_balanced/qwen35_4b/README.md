# Qwen3.5-4B v5 dataset

This is a Qwen3.5-4B-specific, behavior-screened TAU2 telecom dataset.

- Pairs: `500` (`200` train, `300` held-out).
- Native first-token call marker: `<tool_call>` (ID `248058`).
- Each paired prompt changes only the final action/discussion verb.
- Admission rule: clean marker is top-1 and corrupt marker is not top-1.
- Verb counts are balanced independently on clean and corrupt sides in every split.
