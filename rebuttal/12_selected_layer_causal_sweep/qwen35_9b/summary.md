# Qwen3.5-9B L28 causal k sweep

Heldout pairs: 300; reference layer: L28; checkpoint: `/root/autodl-tmp/Transcoder/Qwen3.5-9B/layer28/checkpoint_step_0061035.pt`.
Suppressor groups are selected on train with delta<0 and beta<0, then frozen on heldout.
Raw-logit midpoint flip is separate from top-1 and margin-boundary flip.

| group | k | mode | side | mean Δ tool logit | mean Δ margin | gap progress | gate Δ | top-1 flip | raw-logit midpoint | margin flip |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| suppressor_top20 | 1 | ablate | corrupt | -0.0146 | -0.1613 | -0.007 | +1.379 | 0.0% | 0.0% | 0.0% |
| suppressor_top20 | 1 | swap | corrupt | -0.0142 | -0.1558 | -0.006 | +1.292 | 0.0% | 0.0% | 0.0% |
| suppressor_top20 | 5 | ablate | corrupt | -0.0729 | -0.1517 | -0.041 | +1.721 | 0.0% | 0.0% | 0.0% |
| suppressor_top20 | 5 | swap | corrupt | -0.0321 | -0.1467 | -0.018 | +1.553 | 0.0% | 0.0% | 0.0% |
| suppressor_top20 | 10 | ablate | corrupt | -0.0608 | -0.1363 | -0.032 | +1.928 | 0.0% | 0.0% | 0.0% |
| suppressor_top20 | 10 | swap | corrupt | -0.0262 | -0.1275 | -0.011 | +1.746 | 0.0% | 0.0% | 0.0% |
| suppressor_top20 | 20 | ablate | corrupt | -0.0246 | -0.1675 | -0.011 | +2.232 | 0.0% | 0.0% | 0.0% |
| suppressor_top20 | 20 | swap | corrupt | -0.0071 | -0.1638 | -0.002 | +1.929 | 0.0% | 0.0% | 0.0% |
| driver_top20 | 1 | ablate | clean | +0.0850 | +0.0923 | -0.048 | -0.435 | 0.0% | 0.0% | 0.0% |
| driver_top20 | 1 | swap | clean | +0.0788 | +0.0833 | -0.045 | -0.403 | 0.0% | 0.0% | 0.0% |
| driver_top20 | 5 | ablate | clean | +0.2475 | +0.3517 | -0.137 | -0.957 | 0.0% | 0.0% | 0.0% |
| driver_top20 | 5 | swap | clean | +0.1950 | +0.2867 | -0.109 | -0.809 | 0.0% | 0.0% | 0.0% |
| driver_top20 | 10 | ablate | clean | +0.2508 | +0.3319 | -0.141 | -1.200 | 0.0% | 0.0% | 0.0% |
| driver_top20 | 10 | swap | clean | +0.1879 | +0.2627 | -0.106 | -0.988 | 0.0% | 0.0% | 0.0% |
| driver_top20 | 20 | ablate | clean | +0.3008 | +0.3646 | -0.167 | -1.537 | 0.0% | 0.0% | 0.0% |
| driver_top20 | 20 | swap | clean | +0.2346 | +0.2913 | -0.130 | -1.255 | 0.0% | 0.0% | 0.0% |
| random_layer_matched | 1 | ablate | corrupt | -0.0208 | -0.0254 | -0.009 | -0.111 | 0.0% | 0.0% | 0.0% |
| random_layer_matched | 5 | ablate | corrupt | -0.0208 | -0.0246 | -0.008 | -0.111 | 0.0% | 0.0% | 0.0% |
| random_layer_matched | 10 | ablate | corrupt | -0.0200 | -0.0246 | -0.009 | -0.111 | 0.0% | 0.0% | 0.0% |
| random_layer_matched | 20 | ablate | corrupt | -0.0200 | -0.0246 | -0.009 | -0.111 | 0.0% | 0.0% | 0.0% |
