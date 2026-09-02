# Qwen3.5-4B activation-frequency/selectivity sensitivity

This is a separate robustness audit; it is not substituted for the paper K or historical Table 3 definition.

- Layers: `[28, 29]`; selection: train-200; validation: heldout-300
- Legacy rule: relevant active rate >= 0.05; opposite active rate <= max(0.10, 0.25 x relevant); the active-plus-mean mode applies the analogous mean-activation rule.
- All cells were evaluated on a fixed grid; no heldout cell was selected post hoc.

## Legacy frequency/selectivity constraints + current C >= 0.80

| k | mode/status | train Kc/Ke | heldout frozen Kc/Ke | heldout ratio | heldout Kc−Ke | S/E train relevant active min |
|---:|---|---:|---:|---:|---:|---:|
| 1 | active_plus_mean/complete | 0.01848/0.02899 | 0.01358/0.02637 | 0.5152 | -0.01278 | 0.850/0.985 |
| 2 | active_plus_mean/complete | 0.03063/0.05710 | 0.02363/0.04830 | 0.4891 | -0.02468 | 0.850/0.985 |
| 5 | active_plus_mean/complete | 0.06232/0.11689 | 0.05193/0.10063 | 0.5161 | -0.04869 | 0.850/0.975 |
| 10 | active_plus_mean/complete | 0.10924/0.20688 | 0.09570/0.18497 | 0.5174 | -0.08927 | 0.850/0.900 |
| 16 | active_plus_mean/complete | 0.15307/0.30438 | 0.12364/0.25924 | 0.4769 | -0.13560 | 0.835/0.900 |
| 20 | active_plus_mean/complete | 0.17991/0.36284 | 0.14121/0.30653 | 0.4607 | -0.16532 | 0.835/0.900 |
| 32 | active_plus_mean/complete | 0.24676/0.52215 | 0.19354/0.45237 | 0.4278 | -0.25883 | 0.835/0.815 |
| 50 | active_plus_mean/complete | 0.31346/0.72993 | 0.24722/0.63509 | 0.3893 | -0.38787 | 0.825/0.815 |
| 100 | active_plus_mean/insufficient_candidates | — | — | — | — | — |

The full frequency/consistency grid is in `activation_frequency_grid.csv`; `summary.json` records every parameter cell.
