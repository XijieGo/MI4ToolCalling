# Qwen3.5-9B: Transcoder S/E causal dominance

Formation window: L26,L27,L28,L29; decision layer: L28; held-out pairs: 300.
S is selected from corrupt-higher / write-away aligned features; E is clean-higher / write-toward aligned.

| group | target side | transfer | mean Δm | mean Δ tool logit | mean Δ gate | top-1 flip | margin flip | logit-midpoint flip |
|---|---|---|---:|---:|---:|---:|---:|---:|
| S_top20 | clean | corrupt_to_clean | -0.067292 | +0.011667 | -2.640773 | 0.00% | 0.00% | 0.00% |
| S_top20 | corrupt | clean_to_corrupt | -0.026250 | +0.048750 | +2.385776 | 0.00% | 0.00% | 0.33% |
| E_top20 | clean | corrupt_to_clean | +0.141458 | +0.118333 | -2.926345 | 0.00% | 0.00% | 0.00% |
| E_top20 | corrupt | clean_to_corrupt | +0.192083 | +0.165000 | +2.175480 | 0.00% | 0.00% | 0.67% |

S:E = (0.067292 + 0.026250) / (0.141458 + 0.192083) = 0.280450

All feature choices are frozen from train; the four causal cells use held-out paired activation values.
