# Qwen3-4B: Transcoder S/E causal dominance

Formation window: L22,L23,L24,L25; decision layer: L26; held-out pairs: 300.
S is selected from corrupt-higher / write-away aligned features; E is clean-higher / write-toward aligned.

| group | target side | transfer | mean Δm | mean Δ tool logit | mean Δ gate | top-1 flip |
|---|---|---|---:|---:|---:|---:|
| S_top20 | clean | corrupt_to_clean | -0.632083 | +0.110000 | -6.171646 | 1.00% |
| S_top20 | corrupt | clean_to_corrupt | -0.873333 | -0.874583 | +5.276554 | 0.00% |
| E_top20 | clean | corrupt_to_clean | -0.665000 | +0.027083 | -3.713834 | 2.00% |
| E_top20 | corrupt | clean_to_corrupt | +3.687917 | +1.736250 | +2.923399 | 0.00% |

S:E = (0.632083 + 0.873333) / (0.665000 + 3.687917) = 0.345841

All feature choices are frozen from train; the four causal cells use held-out paired activation values.
