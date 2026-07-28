# v5 model-specific balanced datasets

Each model directory contains 500 paired prompts: 200 train and 300 held-out.
The clean and corrupt verb margins are balanced independently within each
split. Every selected pair was screened on its target model: the clean prompt
has the native tool marker at top-1 and the corrupt prompt does not.

This release has two independent validations:

- `audit.json` verifies all split sizes, verb balances, source uniqueness, and
  prompt hashes.
- `release_validation.json` points to the full live replay audit. It checks
  every clean and corrupt prompt through the experiment runner's exact V0
  input path in its canonical batch size and in batch size 1. All 14,000
  target-model top-1 replays passed with zero mismatches.

| Model | Directory |
| --- | --- |
| Qwen3-4B | `qwen3_4b/` |
| Qwen3-8B | `qwen3_8b/` |
| Qwen3-14B | `qwen3_14b/` |
| Qwen3.5-4B | `qwen35_4b/` |
| Qwen3.5-9B | `qwen35_9b/` |
| Mistral-Small-3.2-24B-Instruct | `mistral_3p2_24b/` |
| Granite-3.3-8B-Instruct | `granite_3p3_8b/` |

Read each directory's `README.md`, `summary.json`, and `manifest.jsonl`
before running an experiment. Use only this release root, not the superseded
pre-live-replay directories retained under `datasets/archive/`.

The Mistral directory deliberately stores prompt JSON payloads with native
`input_ids`, messages, and tool schema. Use those IDs directly, or re-render
from messages and tools through `apply_chat_template(tokenize=True)`. Do not
encode a decoded plain-text prompt for that model because its native special
tokens do not round-trip through normal text encoding.
