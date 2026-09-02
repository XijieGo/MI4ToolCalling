# Qwen3.5-4B top-k / contribution-consistency sensitivity

This is a pre-specified grid; no heldout cell was selected post hoc.

- Layers: `[28]`; selection: train-200; validation: heldout-300
- Consistency is the fraction of train pairs whose feature contribution has the selected positive sign; it is not raw activation rate.
- Historical strict setting: `C >= 0.80`; random-control pool: `C >= 0.75`.

## C >= 0.80 curve

| k | status | train Kc/Ke | heldout frozen Kc/Ke | heldout ratio | heldout Kc−Ke |
|---:|---|---:|---:|---:|---:|
| 1 | complete | 0.02638/0.02929 | 0.02384/0.02101 | 1.1344 | +0.00282 |
| 2 | complete | 0.04496/0.05741 | 0.03722/0.04295 | 0.8667 | -0.00573 |
| 5 | complete | 0.09240/0.12566 | 0.07672/0.10203 | 0.7519 | -0.02531 |
| 10 | complete | 0.15048/0.22058 | 0.12395/0.18033 | 0.6874 | -0.05638 |
| 16 | complete | 0.21018/0.32645 | 0.18041/0.28031 | 0.6436 | -0.09990 |
| 20 | complete | 0.24681/0.39048 | 0.21016/0.33537 | 0.6267 | -0.12521 |
| 32 | complete | 0.33652/0.55801 | 0.27297/0.48546 | 0.5623 | -0.21249 |
| 50 | complete | 0.44341/0.77355 | 0.34832/0.67997 | 0.5123 | -0.33165 |
| 100 | complete | 0.63148/1.23274 | 0.48171/1.11200 | 0.4332 | -0.63029 |
| 200 | insufficient_candidates | — | — | — | — |
| 500 | insufficient_candidates | — | — | — | — |
| 1000 | insufficient_candidates | — | — | — | — |

The complete grid is in `k_consistency_grid.csv`; selected feature diagnostics, including train/heldout consistency and raw active rates, are in `selected_feature_rows.csv`.

## Monotonicity

In this run, both train and heldout absolute-mass curves increased across the scanned k values, while the heldout corrupt/clean ratio decreased across every complete k curve. The ratio is a quotient and this direction is empirical, not a general algebraic guarantee.

```json
{
  "0.00": {
    "train_corrupt_mass_monotonic": true,
    "train_clean_mass_monotonic": true,
    "heldout_corrupt_mass_monotonic": true,
    "heldout_clean_mass_monotonic": true,
    "heldout_ratio_nondecreasing": false,
    "heldout_ratio_nonincreasing": true
  },
  "0.50": {
    "train_corrupt_mass_monotonic": true,
    "train_clean_mass_monotonic": true,
    "heldout_corrupt_mass_monotonic": true,
    "heldout_clean_mass_monotonic": true,
    "heldout_ratio_nondecreasing": false,
    "heldout_ratio_nonincreasing": true
  },
  "0.60": {
    "train_corrupt_mass_monotonic": true,
    "train_clean_mass_monotonic": true,
    "heldout_corrupt_mass_monotonic": true,
    "heldout_clean_mass_monotonic": true,
    "heldout_ratio_nondecreasing": false,
    "heldout_ratio_nonincreasing": true
  },
  "0.70": {
    "train_corrupt_mass_monotonic": true,
    "train_clean_mass_monotonic": true,
    "heldout_corrupt_mass_monotonic": true,
    "heldout_clean_mass_monotonic": true,
    "heldout_ratio_nondecreasing": false,
    "heldout_ratio_nonincreasing": true
  },
  "0.75": {
    "train_corrupt_mass_monotonic": true,
    "train_clean_mass_monotonic": true,
    "heldout_corrupt_mass_monotonic": true,
    "heldout_clean_mass_monotonic": true,
    "heldout_ratio_nondecreasing": false,
    "heldout_ratio_nonincreasing": true
  },
  "0.80": {
    "train_corrupt_mass_monotonic": true,
    "train_clean_mass_monotonic": true,
    "heldout_corrupt_mass_monotonic": true,
    "heldout_clean_mass_monotonic": true,
    "heldout_ratio_nondecreasing": false,
    "heldout_ratio_nonincreasing": true
  },
  "0.85": {
    "train_corrupt_mass_monotonic": true,
    "train_clean_mass_monotonic": true,
    "heldout_corrupt_mass_monotonic": true,
    "heldout_clean_mass_monotonic": true,
    "heldout_ratio_nondecreasing": false,
    "heldout_ratio_nonincreasing": true
  },
  "0.90": {
    "train_corrupt_mass_monotonic": true,
    "train_clean_mass_monotonic": true,
    "heldout_corrupt_mass_monotonic": true,
    "heldout_clean_mass_monotonic": true,
    "heldout_ratio_nondecreasing": false,
    "heldout_ratio_nonincreasing": true
  },
  "0.95": {
    "train_corrupt_mass_monotonic": true,
    "train_clean_mass_monotonic": true,
    "heldout_corrupt_mass_monotonic": true,
    "heldout_clean_mass_monotonic": true,
    "heldout_ratio_nondecreasing": false,
    "heldout_ratio_nonincreasing": true
  }
}
```
