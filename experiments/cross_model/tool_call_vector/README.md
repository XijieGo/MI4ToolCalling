# Locked tool-call vector

One layer and one hook per model. The measured table is `results/tool_call_vector/summary.md`.

| model | layer | hook | data |
|---|---:|---|---|
| qwen3_4b | 24 | pre | `datasets/qwen3_4b/pair` (200/300) |
| qwen3_8b | 24 | pre | `datasets/qwen3_8b/pair` (300/200) |
| qwen3_14b | 31 | pre | `datasets/qwen3_14b/pair` (200/300) |
| qwen35_4b | 30 | post | `trash/selected_500/qwen35_4b` (300/200) |
| qwen35_9b | 30 | post | `trash/selected_500/qwen35_9b` (300/200) |
| granite_3p3_8b | 25 | pre | `datasets/granite_3p3_8b/pair` (200/300) |
| mistral_3p2_24b | 25 | pre | `datasets/mistral_3p2_24b/pair` (200/300) |

`pre` is the decoder-block input. `post` is the decoder-block output. The vector is the train-split mean of clean minus corrupt at the last real token.

```bash
# one model, or every locked model
bash scripts/run_tool_call_vector.sh qwen3_8b
bash scripts/run_tool_call_vector.sh
```

A run writes `results/tool_call_vector/<model_key>/`.
