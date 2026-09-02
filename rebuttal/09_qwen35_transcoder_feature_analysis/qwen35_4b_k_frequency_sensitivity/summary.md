# Qwen3.5-4B top-k / contribution-consistency sensitivity

This is a pre-specified grid; no heldout cell was selected post hoc.

- Layers: `[28, 29]`; selection: train-200; validation: heldout-300
- Consistency is the fraction of train pairs whose feature contribution has the selected positive sign; it is not raw activation rate.
- Historical strict setting: `C >= 0.80`; random-control pool: `C >= 0.75`.

## C >= 0.80 curve

| k | status | train Kc/Ke | heldout frozen Kc/Ke | heldout ratio | heldout Kc−Ke |
|---:|---|---:|---:|---:|---:|
| 1 | complete | 0.03291/0.04439 | 0.02958/0.03201 | 0.9242 | -0.00243 |
| 2 | complete | 0.05929/0.07503 | 0.05342/0.06040 | 0.8845 | -0.00698 |
| 5 | complete | 0.11241/0.16142 | 0.09575/0.12971 | 0.7382 | -0.03396 |
| 10 | complete | 0.17401/0.27701 | 0.14807/0.22773 | 0.6502 | -0.07966 |
| 16 | complete | 0.23639/0.39132 | 0.20008/0.32256 | 0.6203 | -0.12248 |
| 20 | complete | 0.27476/0.46284 | 0.23776/0.39392 | 0.6036 | -0.15616 |
| 32 | complete | 0.37520/0.66155 | 0.31214/0.56098 | 0.5564 | -0.24884 |
| 50 | complete | 0.49565/0.91789 | 0.40128/0.76371 | 0.5254 | -0.36243 |
| 100 | complete | 0.72561/1.50334 | 0.57423/1.28404 | 0.4472 | -0.70982 |
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
