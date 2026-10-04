# Granite-3.3-8B-Instruct L31 selected Transcoder top-20 causal audit

Heldout pairs: 240; reference layer: L36; checkpoint: `external Transcoder checkpoint`.
The raw-logit midpoint flip is separate from top-1: it asks whether the intervened tool-call logit crosses the paired clean/corrupt tool-logit midpoint.

| group | side | direction | mean tool-logit Δ | mean margin Δ | gap progress | top-1 flip | margin flip | raw-logit midpoint flip | reached paired target logit |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| suppressor_top20 | corrupt | ablate | +0.0935 | +0.2206 | +0.022 | 0.0% | 0.0% | 0.0% | 0.0% |
| driver_top20 | clean | ablate | +0.1219 | -0.0654 | -0.037 | 0.0% | 0.0% | 0.0% | 0.0% |
| random_layer_matched | corrupt | ablate | -0.0021 | -0.0010 | -0.002 | 0.0% | 0.0% | 0.0% | 0.0% |
