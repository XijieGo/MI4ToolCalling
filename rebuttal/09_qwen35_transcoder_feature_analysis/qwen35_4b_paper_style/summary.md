# Qwen3.5-4B paper-style Transcoder analysis

Train-200 fits the direction and selects features; heldout-300 is used for validation and causal ablation.

The paper's K totals group by activation side only. The aligned S/E quadrants are reported separately and are not substituted for K.

## Held-out paper K summary

| layer | K_corrupt | K_clean | Kc/Kclean | Kc top-20 | Kclean top-20 | aligned S | aligned E |
|---:|---:|---:|---:|---:|---:|---:|---:|
| L28 | 2.1929 | 7.1857 | 0.305 | 0.2557 | 0.4145 | 1.1284 | 3.6849 |
| L29 | 0.8460 | 4.1706 | 0.203 | 0.1990 | 0.4308 | 0.4449 | 2.1446 |

## Held-out corrupt-prompt feature ablation

| group | features | mean gate Δ | mean margin Δ | mean log-odds Δ | mean tool-logit Δ | recovery |
|---|---:|---:|---:|---:|---:|---:|
| paper_suppressor_top | 5 | +0.4519 | -0.6796 | -0.9735 | -1.2454 | 0.00% |
| aligned_driver_top_control | 5 | -0.0446 | -0.0273 | -0.0328 | -0.0377 | 0.00% |
| random_layer_matched_control | 5 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0.00% |
