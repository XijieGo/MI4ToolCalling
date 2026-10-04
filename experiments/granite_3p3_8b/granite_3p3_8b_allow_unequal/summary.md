# Granite-3.3-8B-Instruct: Transcoder K_c/K_e

Train-200 fits the direction and selects Table-3-style top-20 features; heldout-300 validates the frozen selection. This is a diagnostic final-position run that retains 40 train and 60 heldout token-length mismatches.
Available Transcoder layers: `[29, 30, 31, 32]`. Historical reference layer: `L36`; reference layer is outside the available Transcoder coverage.

## Paper-style all-feature K

| split | K_corrupt | K_clean | Kc/Kclean | aligned S | aligned E | S/E |
|---|---:|---:|---:|---:|---:|---:|
| train | 15.84149 | 25.34882 | 0.625 | 13.01769 | 19.25983 | 0.676 |
| heldout | 15.78146 | 25.65589 | 0.615 | 12.98209 | 19.42020 | 0.668 |

## Table-3-style aligned global top-k

| k | status | train Kc/Ke | heldout frozen Kc/Ke | heldout ratio | heldout Kc−Ke | heldout re-ranked ratio |
|---:|---|---:|---:|---:|---:|---:|
| 20 | complete | 5.75101/6.90043 | 5.78417/6.92710 | 0.835 | -1.14293 | 0.834 |

The all-feature K and aligned top-k estimates are different estimands; neither has a consistency/frequency filter in this primary run.
