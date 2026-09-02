# Qwen3.5-4B activation-frequency/selectivity sensitivity

This is a separate robustness audit; it is not substituted for the paper K or historical Table 3 definition.

- Layers: `[28]`; selection: train-200; validation: heldout-300
- Legacy rule: relevant active rate >= 0.05; opposite active rate <= max(0.10, 0.25 x relevant); the active-plus-mean mode applies the analogous mean-activation rule.
- All cells were evaluated on a fixed grid; no heldout cell was selected post hoc.

## Legacy frequency/selectivity constraints + current C >= 0.80

| k | mode/status | train Kc/Ke | heldout frozen Kc/Ke | heldout ratio | heldout Kc−Ke | S/E train relevant active min |
|---:|---|---:|---:|---:|---:|---:|
| 1 | active_plus_mean/complete | 0.01848/0.02811 | 0.01358/0.02194 | 0.6192 | -0.00835 | 0.850/1.000 |
| 2 | active_plus_mean/complete | 0.03063/0.04877 | 0.02363/0.03947 | 0.5985 | -0.01585 | 0.850/0.995 |
| 5 | active_plus_mean/complete | 0.06232/0.10517 | 0.05193/0.08961 | 0.5796 | -0.03767 | 0.850/0.900 |
| 10 | active_plus_mean/complete | 0.10581/0.18543 | 0.08666/0.16319 | 0.5310 | -0.07653 | 0.835/0.900 |
| 16 | active_plus_mean/complete | 0.14680/0.26607 | 0.11321/0.23667 | 0.4783 | -0.12346 | 0.835/0.815 |
| 20 | active_plus_mean/complete | 0.17183/0.31526 | 0.13223/0.27673 | 0.4778 | -0.14450 | 0.835/0.815 |
| 32 | active_plus_mean/complete | 0.22637/0.44666 | 0.17348/0.41614 | 0.4169 | -0.24267 | 0.835/0.815 |
| 50 | active_plus_mean/complete | 0.27481/0.60419 | 0.20892/0.56386 | 0.3705 | -0.35493 | 0.800/0.805 |
| 100 | active_plus_mean/insufficient_candidates | — | — | — | — | — |

The full frequency/consistency grid is in `activation_frequency_grid.csv`; `summary.json` records every parameter cell.
