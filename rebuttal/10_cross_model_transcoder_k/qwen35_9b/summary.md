# Qwen3.5-9B: Transcoder K_c/K_e

Train-200 fits the direction and selects Table-3-style top-20 features; heldout-300 validates the frozen selection.
Available Transcoder layers: `[26, 27, 28, 29]`. Historical reference layer: `L28`; reference layer is included in the available Transcoder coverage.

## Paper-style all-feature K

| split | K_corrupt | K_clean | Kc/Kclean | aligned S | aligned E | S/E |
|---|---:|---:|---:|---:|---:|---:|
| train | 19.66766 | 31.96800 | 0.615 | 10.04091 | 14.17892 | 0.708 |
| heldout | 17.99780 | 30.48919 | 0.590 | 9.28931 | 13.40993 | 0.693 |

## Table-3-style aligned global top-k

| k | status | train Kc/Ke | heldout frozen Kc/Ke | heldout ratio | heldout Kc−Ke | heldout re-ranked ratio |
|---:|---|---:|---:|---:|---:|---:|
| 20 | complete | 3.02130/1.98526 | 2.87807/1.92690 | 1.494 | +0.95117 | 1.492 |

The all-feature K and aligned top-k estimates are different estimands; neither has a consistency/frequency filter in this primary run.
