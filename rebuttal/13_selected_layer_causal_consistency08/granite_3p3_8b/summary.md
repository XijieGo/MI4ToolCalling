# Granite-3.3-8B-Instruct L31 causal k sweep

Heldout pairs: 240; reference layer: L36; checkpoint: `/root/autodl-tmp/Transcoder/granite-3.3-8b-instruct/layer31/checkpoint_step_0061035.pt`.
Suppressor groups are selected on train with delta<0 and beta<0, then frozen on heldout.
Raw-logit midpoint flip is separate from top-1 and margin-boundary flip.

| group | k | mode | side | mean Δ tool logit | mean Δ margin | gap progress | gate Δ | top-1 flip | raw-logit midpoint | margin flip |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| suppressor_top20 | 1 | ablate | corrupt | -0.0437 | -0.0453 | -0.012 | +0.675 | 0.0% | 0.0% | 0.0% |
| suppressor_top20 | 1 | swap | corrupt | -0.0370 | -0.0365 | -0.011 | +0.613 | 0.0% | 0.0% | 0.0% |
| suppressor_top20 | 5 | ablate | corrupt | +0.1260 | +0.0844 | +0.032 | +0.979 | 0.0% | 0.0% | 0.0% |
| suppressor_top20 | 5 | swap | corrupt | +0.0706 | +0.0336 | +0.018 | +0.781 | 0.0% | 0.0% | 0.0% |
| suppressor_top20 | 10 | ablate | corrupt | +0.0154 | +0.0409 | +0.002 | +1.713 | 0.0% | 0.0% | 0.0% |
| suppressor_top20 | 10 | swap | corrupt | -0.0065 | -0.0143 | -0.002 | +1.212 | 0.0% | 0.0% | 0.0% |
| suppressor_top20 | 20 | ablate | corrupt | +0.1852 | +0.3638 | +0.046 | +2.729 | 0.0% | 0.0% | 0.0% |
| suppressor_top20 | 20 | swap | corrupt | +0.0979 | +0.1849 | +0.023 | +1.845 | 0.0% | 0.0% | 0.0% |
| driver_top20 | 1 | ablate | clean | -0.0036 | -0.0044 | +0.001 | -0.078 | 0.0% | 0.0% | 0.0% |
| driver_top20 | 1 | swap | clean | -0.0063 | -0.0068 | +0.002 | -0.075 | 0.0% | 0.0% | 0.0% |
| driver_top20 | 5 | ablate | clean | +0.1125 | +0.0880 | -0.029 | -0.740 | 0.0% | 0.0% | 0.0% |
| driver_top20 | 5 | swap | clean | +0.1255 | +0.1271 | -0.033 | -0.587 | 0.0% | 0.0% | 0.0% |
| driver_top20 | 10 | ablate | clean | +0.3000 | +0.2859 | -0.084 | -0.757 | 0.0% | 0.0% | 0.0% |
| driver_top20 | 10 | swap | clean | +0.2557 | +0.2638 | -0.070 | -0.576 | 0.0% | 0.0% | 0.0% |
| driver_top20 | 20 | ablate | clean | +0.0276 | -0.2826 | -0.011 | -1.186 | 0.0% | 0.0% | 0.0% |
| driver_top20 | 20 | swap | clean | +0.0990 | -0.0586 | -0.028 | -0.854 | 0.0% | 0.0% | 0.0% |
| random_layer_matched | 1 | ablate | corrupt | -0.0169 | -0.0174 | -0.006 | -0.028 | 0.0% | 0.0% | 0.0% |
| random_layer_matched | 5 | ablate | corrupt | +0.0216 | +0.0242 | +0.004 | +0.071 | 0.0% | 0.0% | 0.0% |
| random_layer_matched | 10 | ablate | corrupt | +0.0044 | +0.0044 | -0.001 | +0.063 | 0.0% | 0.0% | 0.0% |
| random_layer_matched | 20 | ablate | corrupt | -0.0578 | -0.1031 | -0.017 | -0.042 | 0.0% | 0.0% | 0.0% |
