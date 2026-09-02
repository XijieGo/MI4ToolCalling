# Qwen3.5-4B historical Table 3 K-protocol audit

This is the historical cross-scale protocol, not the paper's all-feature activation-side K summary.

- Selection: train-200 pairs; validation: heldout-300 pairs
- Layers: `[28]`
- Window note: Strict pre-decision subset using the available L28 checkpoint.
- Candidate masks: corrupt `delta_activation < 0 and beta_mu < 0`; clean `delta_activation > 0 and beta_mu > 0`
- Selection: `top-20 abs(kappa) per layer prefilter, then global top-20 per aligned category`

| evaluation | K_corrupt | K_clean | ratio |
|---|---:|---:|---:|
| train selection | 0.247440 | 0.390483 | 0.6337 |
| heldout recomputed top-20 | 0.213400 | 0.348284 | 0.6127 |

Frozen train-selected features evaluated on heldout:

| category | train-selected abs mass on heldout | still-aligned heldout mass |
|---|---:|---:|
| corrupt | 0.207881 | 0.207881 |
| clean | 0.335366 | 0.335366 |
