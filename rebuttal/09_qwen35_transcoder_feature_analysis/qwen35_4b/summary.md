# Qwen3.5-4B Transcoder feature analysis

The 200 train pairs fit the L29 clean-minus-corrupt direction and select features; the 300 held-out pairs are validation only.

- Transcoders: L28 checkpoint step `70000`, L29 checkpoint step `65000`.
- Fitted direction: L29 decoder-block output; norm `24.8594`.
- Tool token: `<tool_call>` (ID `248058`).

## Held-out quadrant summary

| layer | suppressor mass | driver mass | S/E | S-E | quadrant residual |
|---:|---:|---:|---:|---:|---:|
| L28 | 1.1284 | 3.6849 | 0.306 | -2.5565 | -1.49e-07 |
| L29 | 0.4449 | 2.1446 | 0.207 | -1.6997 | -1.94e-07 |

## Interpretation

The suppressor-over-driver pattern is not uniform across the two analyzed layers; the layer-level table should be read rather than collapsed into a blanket replication claim.

## Held-out causal feature swaps

| group | side | direction | mean gate Δ | mean tool-logit Δ | top-1 | recovery | drop |
|---|---|---|---:|---:|---:|---:|---:|
| suppressor_top | clean | to_corrupt | -0.1115 | +0.0050 | 99.7% | 0.0% | 0.0% |
| suppressor_top | corrupt | to_clean | +0.1061 | -0.0104 | 0.0% | 0.0% | 0.0% |
| driver_top | clean | to_corrupt | -0.1421 | -0.0821 | 99.7% | 0.0% | 0.0% |
| driver_top | corrupt | to_clean | +0.1424 | +0.0885 | 0.0% | 0.0% | 0.0% |
| random_layer_matched | clean | to_corrupt | +0.0000 | +0.0000 | 99.7% | 0.0% | 0.0% |
| random_layer_matched | corrupt | to_clean | +0.0000 | +0.0000 | 0.0% | 0.0% | 0.0% |

The causal table is the direct replication check: suppressor swaps should move clean prompts downward and corrupt prompts upward if the Qwen3 suppression account transfers.
