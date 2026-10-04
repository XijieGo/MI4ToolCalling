# Qwen3.5-4B: Transcoder K_c/K_e

Train-200 fits the direction and selects Table-3-style top-20 features; heldout-300 validates the frozen selection.
Available Transcoder layers: `[26, 27, 28, 29]`. Historical reference layer: `L29`; reference layer is included in the available Transcoder coverage.

## Paper-style all-feature K

| split | K_corrupt | K_clean | Kc/Kclean | aligned S | aligned E | S/E |
|---|---:|---:|---:|---:|---:|---:|
| train | 4.90674 | 13.53100 | 0.363 | 2.46555 | 7.07089 | 0.349 |
| heldout | 4.60022 | 15.04655 | 0.306 | 2.37177 | 7.82384 | 0.303 |

## Table-3-style aligned global top-k

| k | status | train Kc/Ke | heldout frozen Kc/Ke | heldout ratio | heldout Kc−Ke | heldout re-ranked ratio |
|---:|---|---:|---:|---:|---:|---:|
| 20 | complete | 0.29887/0.47687 | 0.25073/0.38898 | 0.645 | -0.13825 | 0.599 |

The all-feature K and aligned top-k estimates are different estimands; neither has a consistency/frequency filter in this primary run.
