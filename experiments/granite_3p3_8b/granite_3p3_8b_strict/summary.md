# Granite-3.3-8B-Instruct: Transcoder K_c/K_e

Train-160 fits the direction and selects Table-3-style top-20 features; heldout-240 validates the frozen selection. The strict token-alignment policy excludes 40 train and 60 heldout pairs whose Granite token lengths differ.
Available Transcoder layers: `[29, 30, 31, 32]`. Historical reference layer: `L36`; reference layer is outside the available Transcoder coverage.

## Paper-style all-feature K

| split | K_corrupt | K_clean | Kc/Kclean | aligned S | aligned E | S/E |
|---|---:|---:|---:|---:|---:|---:|
| train | 16.53452 | 25.02917 | 0.661 | 13.38696 | 18.63256 | 0.718 |
| heldout | 16.58952 | 24.23891 | 0.684 | 13.41611 | 18.15823 | 0.739 |

## Table-3-style aligned global top-k

| k | status | train Kc/Ke | heldout frozen Kc/Ke | heldout ratio | heldout Kc−Ke | heldout re-ranked ratio |
|---:|---|---:|---:|---:|---:|---:|
| 20 | complete | 6.00338/6.46501 | 6.00854/6.56914 | 0.915 | -0.56060 | 0.917 |

The all-feature K and aligned top-k estimates are different estimands; neither has a consistency/frequency filter in this primary run.
