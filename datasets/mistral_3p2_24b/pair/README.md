# Mistral-Small-3.2-24B paired prompts

This model-specific TAU2 dataset is stored as native Mistral token IDs.

- Pairs: `500` (`200` train, `300` held-out).
- Native first-token call marker: `[TOOL_CALLS]` (ID `9`).
- Each paired prompt changes only the final action/discussion verb.
- Every prompt JSON contains the original messages, tool schema, and exact native `input_ids`.
- Use the stored IDs directly or re-render from `messages` and `tools` with `apply_chat_template(tokenize=True)`.
- Do not encode a decoded plain-text rendering: this backend does not round-trip special tokens losslessly.
