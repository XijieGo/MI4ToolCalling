# Qwen3.5-9B L28 causal k sweep

Heldout pairs: 300; reference layer: L28; checkpoint: `/root/autodl-tmp/Transcoder/Qwen3.5-9B/layer28/checkpoint_step_0061035.pt`.
Suppressor groups are selected on train with delta<0 and beta<0, then frozen on heldout.
Raw-logit midpoint flip is separate from top-1 and margin-boundary flip.

| group | k | mode | side | mean Δ tool logit | mean Δ margin | gap progress | gate Δ | top-1 flip | raw-logit midpoint | margin flip |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| suppressor_top20 | 1 | ablate | corrupt | -0.0146 | -0.1613 | -0.007 | +1.379 | 0.0% | 0.0% | 0.0% |
| suppressor_top20 | 1 | swap | corrupt | -0.0142 | -0.1558 | -0.006 | +1.292 | 0.0% | 0.0% | 0.0% |
| suppressor_top20 | 5 | ablate | corrupt | -0.0896 | -0.1542 | -0.050 | +1.730 | 0.0% | 0.0% | 0.0% |
| suppressor_top20 | 5 | swap | corrupt | -0.0521 | -0.1542 | -0.026 | +1.557 | 0.0% | 0.0% | 0.0% |
| suppressor_top20 | 10 | ablate | corrupt | -0.0625 | -0.1983 | -0.033 | +1.900 | 0.0% | 0.0% | 0.0% |
| suppressor_top20 | 10 | swap | corrupt | -0.0258 | -0.1888 | -0.014 | +1.712 | 0.0% | 0.0% | 0.0% |
| suppressor_top20 | 20 | ablate | corrupt | -0.0262 | -0.1800 | -0.012 | +2.152 | 0.0% | 0.0% | 0.0% |
| suppressor_top20 | 20 | swap | corrupt | -0.0121 | -0.1829 | -0.004 | +1.868 | 0.0% | 0.0% | 0.0% |
| driver_top20 | 1 | ablate | clean | +0.0850 | +0.0923 | -0.048 | -0.435 | 0.0% | 0.0% | 0.0% |
| driver_top20 | 1 | swap | clean | +0.0788 | +0.0833 | -0.045 | -0.403 | 0.0% | 0.0% | 0.0% |
| driver_top20 | 5 | ablate | clean | +0.2475 | +0.3517 | -0.137 | -0.957 | 0.0% | 0.0% | 0.0% |
| driver_top20 | 5 | swap | clean | +0.1950 | +0.2867 | -0.109 | -0.809 | 0.0% | 0.0% | 0.0% |
| driver_top20 | 10 | ablate | clean | +0.2517 | +0.3700 | -0.140 | -1.217 | 0.0% | 0.0% | 0.0% |
| driver_top20 | 10 | swap | clean | +0.2008 | +0.3065 | -0.112 | -0.982 | 0.0% | 0.0% | 0.0% |
| driver_top20 | 20 | ablate | clean | +0.3183 | +0.4067 | -0.177 | -1.513 | 0.0% | 0.0% | 0.0% |
| driver_top20 | 20 | swap | clean | +0.2462 | +0.3308 | -0.135 | -1.227 | 0.0% | 0.0% | 0.0% |
| random_layer_matched | 1 | ablate | corrupt | -0.0233 | -0.0229 | -0.010 | -0.109 | 0.0% | 0.0% | 0.0% |
| random_layer_matched | 5 | ablate | corrupt | -0.0100 | -0.0167 | -0.004 | -0.079 | 0.0% | 0.0% | 0.0% |
| random_layer_matched | 10 | ablate | corrupt | -0.0071 | -0.0196 | -0.003 | -0.050 | 0.0% | 0.0% | 0.0% |
| random_layer_matched | 20 | ablate | corrupt | -0.0108 | +0.0142 | -0.004 | +0.018 | 0.0% | 0.0% | 0.0% |
