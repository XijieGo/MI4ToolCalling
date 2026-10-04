# Locked-vector transfer

Fits each model's coding vector once, at the layer and hook in `tool_call_vector/run.py`, then applies it unchanged.

- Multi-domain keeps the coding direction and rescales its L2 norm to the target domain contrast. Qwen3-4B, Qwen3-8B, and Qwen3-14B all read `datasets/qwen3_8b/multi_domain`: 400 train pairs set the norm and 100 test pairs are scored. Other models use their own 100-pair test set for both the norm and the score.
- Verb-free baselines all 600 stored requests, then keeps up to 10 fresh tool-call top-1 rows per domain and pattern. Removal is alpha 1 and 1.5. Mistral uses stored `input_ids`.
- Tau2 uses the raw coding vector at alpha 1. The call arm is subtracted and the text arm is added. Qwen models render Telecom. Granite and Mistral render Retail, with the historical message adapters.

The control is a same-norm Gaussian made orthogonal to the coding vector, seed 20260726. Pairs are not dropped when a fresh baseline disagrees with an older screen.

```bash
bash scripts/run_transfer.sh
```

Outputs land in `results/transfer/<model>/summary.json` and `results/transfer/summary.md`.
