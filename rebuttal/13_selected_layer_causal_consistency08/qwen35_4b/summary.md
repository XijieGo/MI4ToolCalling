# Qwen3.5-4B L27 causal k sweep

Heldout pairs: 300; reference layer: L29; checkpoint: `/root/autodl-tmp/Transcoder/Qwen3.5-4B/layer27/checkpoint_step_0061035.pt`.
Suppressor groups are selected on train with delta<0 and beta<0, then frozen on heldout.
Raw-logit midpoint flip is separate from top-1 and margin-boundary flip.

| group | k | mode | side | mean Δ tool logit | mean Δ margin | gap progress | gate Δ | top-1 flip | raw-logit midpoint | margin flip |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| suppressor_top20 | 1 | ablate | corrupt | -0.0527 | -0.0010 | -0.011 | +0.166 | 0.0% | 0.0% | 0.0% |
| suppressor_top20 | 1 | swap | corrupt | +0.0052 | +0.0144 | +0.001 | +0.017 | 0.0% | 0.0% | 0.0% |
| suppressor_top20 | 5 | ablate | corrupt | -0.3306 | +0.0527 | -0.068 | +0.522 | 0.0% | 0.0% | 0.0% |
| suppressor_top20 | 5 | swap | corrupt | -0.0094 | +0.0415 | -0.003 | +0.083 | 0.0% | 0.0% | 0.0% |
| suppressor_top20 | 10 | ablate | corrupt | -0.3285 | +0.1377 | -0.068 | +0.608 | 0.0% | 0.0% | 0.0% |
| suppressor_top20 | 10 | swap | corrupt | +0.0254 | +0.0988 | +0.004 | +0.125 | 0.0% | 0.0% | 0.0% |
| suppressor_top20 | 20 | ablate | corrupt | -0.5877 | +0.1798 | -0.118 | +0.832 | 0.0% | 0.0% | 0.0% |
| suppressor_top20 | 20 | swap | corrupt | -0.0675 | +0.0279 | -0.013 | +0.217 | 0.0% | 0.0% | 0.0% |
| driver_top20 | 1 | ablate | clean | -0.0163 | +0.0108 | +0.004 | -0.013 | 0.0% | 0.0% | 0.0% |
| driver_top20 | 1 | swap | clean | -0.0075 | +0.0163 | +0.002 | -0.008 | 0.0% | 0.0% | 0.0% |
| driver_top20 | 5 | ablate | clean | -0.0129 | +0.0333 | +0.003 | -0.061 | 0.0% | 0.0% | 0.0% |
| driver_top20 | 5 | swap | clean | -0.0117 | +0.0300 | +0.003 | -0.052 | 0.0% | 0.0% | 0.0% |
| driver_top20 | 10 | ablate | clean | -0.0429 | +0.0071 | +0.009 | -0.099 | 0.3% | 0.0% | 0.0% |
| driver_top20 | 10 | swap | clean | -0.0258 | +0.0225 | +0.005 | -0.085 | 0.0% | 0.0% | 0.0% |
| driver_top20 | 20 | ablate | clean | -0.0317 | +0.0542 | +0.007 | -0.141 | 0.0% | 0.0% | 0.0% |
| driver_top20 | 20 | swap | clean | -0.0308 | +0.0387 | +0.006 | -0.119 | 0.0% | 0.0% | 0.0% |
| random_layer_matched | 1 | ablate | corrupt | -0.0010 | +0.0090 | -0.000 | -0.026 | 0.0% | 0.0% | 0.0% |
| random_layer_matched | 5 | ablate | corrupt | -0.0050 | +0.0150 | -0.000 | -0.004 | 0.0% | 0.0% | 0.0% |
| random_layer_matched | 10 | ablate | corrupt | +0.0240 | +0.0435 | +0.005 | +0.017 | 0.0% | 0.0% | 0.0% |
| random_layer_matched | 20 | ablate | corrupt | -0.0269 | +0.0335 | -0.005 | +0.069 | 0.0% | 0.0% | 0.0% |
