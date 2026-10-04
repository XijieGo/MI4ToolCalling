# Screened native pairs

Each model has 500 paired prompts: 200 train and 300 held-out. Clean and corrupt verb margins are balanced within each split. Every selected pair was screened on its target model: the clean prompt has the native tool marker at top-1 and the corrupt prompt does not.

| Model | Directory |
| --- | --- |
| Qwen3-4B | `datasets/qwen3_4b/pair/` |
| Qwen3-8B | `datasets/qwen3_8b/native/` |
| Qwen3-14B | `datasets/qwen3_14b/pair/` |
| Qwen3.5-4B | `datasets/qwen35_4b/pair/` |
| Qwen3.5-9B | `datasets/qwen35_9b/pair/` |
| Mistral-Small-3.2-24B-Instruct | `datasets/mistral_3p2_24b/pair/` |
| Granite-3.3-8B-Instruct | `datasets/granite_3p3_8b/pair/` |

Qwen3-8B's `datasets/qwen3_8b/pair/` directory is a separate 300/200 rerun, not this screened split.

`audit.json` records split sizes, verb balances, and prompt hashes. `release_validation.json` records the live replay: 14,000 target-model top-1 checks, zero mismatches. The replay log itself is not in this repo.

Mistral stores native `input_ids`. Use those IDs. Do not encode a decoded plain-text prompt.
