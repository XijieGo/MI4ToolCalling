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
| suppressor_top20 | 10 | ablate | corrupt | +0.0195 | -0.0617 | +0.005 | +1.262 | 0.0% | 0.0% | 0.0% |
| suppressor_top20 | 10 | swap | corrupt | -0.0104 | -0.0766 | -0.003 | +0.973 | 0.0% | 0.0% | 0.0% |
| suppressor_top20 | 20 | ablate | corrupt | +0.0935 | +0.2206 | +0.022 | +2.503 | 0.0% | 0.0% | 0.0% |
| suppressor_top20 | 20 | swap | corrupt | +0.0417 | +0.0969 | +0.009 | +1.724 | 0.0% | 0.0% | 0.0% |
| driver_top20 | 1 | ablate | clean | -0.0036 | -0.0044 | +0.001 | -0.078 | 0.0% | 0.0% | 0.0% |
| driver_top20 | 1 | swap | clean | -0.0063 | -0.0068 | +0.002 | -0.075 | 0.0% | 0.0% | 0.0% |
| driver_top20 | 5 | ablate | clean | +0.0052 | -0.0852 | -0.002 | -0.778 | 0.0% | 0.0% | 0.0% |
| driver_top20 | 5 | swap | clean | +0.0172 | -0.0453 | -0.005 | -0.635 | 0.0% | 0.0% | 0.0% |
| driver_top20 | 10 | ablate | clean | +0.2354 | +0.2133 | -0.068 | -0.918 | 0.0% | 0.0% | 0.0% |
| driver_top20 | 10 | swap | clean | +0.1995 | +0.2188 | -0.057 | -0.727 | 0.0% | 0.0% | 0.0% |
| driver_top20 | 20 | ablate | clean | +0.1219 | -0.0654 | -0.037 | -1.172 | 0.0% | 0.0% | 0.0% |
| driver_top20 | 20 | swap | clean | +0.1313 | +0.0383 | -0.038 | -0.956 | 0.0% | 0.0% | 0.0% |
| random_layer_matched | 1 | ablate | corrupt | -0.0021 | -0.0010 | -0.002 | +0.005 | 0.0% | 0.0% | 0.0% |
| random_layer_matched | 5 | ablate | corrupt | -0.0021 | -0.0010 | -0.002 | +0.005 | 0.0% | 0.0% | 0.0% |
| random_layer_matched | 10 | ablate | corrupt | -0.0021 | -0.0010 | -0.002 | +0.005 | 0.0% | 0.0% | 0.0% |
| random_layer_matched | 20 | ablate | corrupt | -0.0021 | -0.0010 | -0.002 | +0.005 | 0.0% | 0.0% | 0.0% |
