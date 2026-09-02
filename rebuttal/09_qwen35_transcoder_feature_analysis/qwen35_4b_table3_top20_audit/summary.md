# Qwen3.5-4B historical Table 3 K-protocol audit

This is the historical cross-scale protocol, not the paper's all-feature activation-side K summary.

- Selection: train-200 pairs; validation: heldout-300 pairs
- Layers: `[28, 29]`
- Window note: Available-checkpoint audit over L28/L29; not the exact four-layer pre-commit window. For the strict pre-decision subset with L29 as decision layer, use --layers 28.
- Candidate masks: corrupt `delta_activation < 0 and beta_mu < 0`; clean `delta_activation > 0 and beta_mu > 0`
- Selection: `top-20 abs(kappa) per layer prefilter, then global top-20 per aligned category`

| evaluation | K_corrupt | K_clean | ratio |
|---|---:|---:|---:|
| train selection | 0.274755 | 0.467595 | 0.5876 |
| heldout recomputed top-20 | 0.246986 | 0.418520 | 0.5901 |

Frozen train-selected features evaluated on heldout:

| category | train-selected abs mass on heldout | still-aligned heldout mass |
|---|---:|---:|
| corrupt | 0.237757 | 0.237757 |
| clean | 0.403588 | 0.403588 |
