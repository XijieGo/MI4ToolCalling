# Cross-model runs

The locked tool-call vector is `tool_call_vector/`. Each model has one layer and one hook. Run it with `scripts/run_tool_call_vector.sh`. The measured table is `results/tool_call_vector/summary.md`.

`transfer/` applies that locked coding vector to multi-domain, verb-free, and tau2. It does not refit the direction on those sets. Run it with `scripts/run_transfer.sh`. The table is `results/transfer/summary.md`.

`run_native_localization.py` is the earlier single-site patch runner. It fits a direction on a model's screened 200 train pairs and evaluates the 300 held-out pairs. It does not drop a pair because a new baseline disagrees with the frozen selection.

Granite reads the stored rendered text with `add_special_tokens=False` and expects `<|tool_call|>` to be token `49154`. Mistral reads stored `input_ids` and uses `[TOOL_CALLS]` token `9`. Do not decode and re-encode Mistral prompts.

```bash
python experiments/cross_model/run_native_localization.py \
  --model-key granite_3p3_8b --dry-run

python experiments/cross_model/run_native_localization.py \
  --model-key mistral_3p2_24b \
  --model-path "$MI4TC_MISTRAL_PATH" \
  --layers 25 --direction-layer 25 \
  --output-root runs/mistral_pair
```

Those layer numbers match `tool_call_vector/`. Use the shell script above when the goal is the locked sufficiency and necessity measurement.

`prediction` uses every selected pair. `strict` keeps only pairs with equal token lengths and one changed token.

`inspect_audits.py` and `summarize_transfer.py` read the summary files in this tree and under the other model experiment directories. They do not load a model.
