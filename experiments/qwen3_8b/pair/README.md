# Qwen3-8B pair localization

Fits a mean clean-minus-corrupt direction on `datasets/qwen3_8b/controlled/train` and evaluates held-out patches. This is the 1,200/300 code-domain set, not the 300/200 rerun in `datasets/qwen3_8b/pair`.

```bash
python experiments/qwen3_8b/pair/run.py \
  --model-path "$MI4TC_QWEN3_8B_PATH" \
  --layers 24 \
  --direction-layer 24 \
  --output-root runs/qwen3_8b_pair
```
