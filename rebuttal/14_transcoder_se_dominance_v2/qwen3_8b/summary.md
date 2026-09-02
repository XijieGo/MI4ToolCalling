# Qwen3-8B: Transcoder S/E causal dominance

Formation window: L20,L21,L22,L23; decision layer: L24; held-out pairs: 300.
S is selected from corrupt-higher / write-away aligned features; E is clean-higher / write-toward aligned.

| group | target side | transfer | mean Δm | mean Δ tool logit | mean Δ gate | top-1 flip | margin flip | logit-midpoint flip |
|---|---|---|---:|---:|---:|---:|---:|---:|
| S_top20 | clean | corrupt_to_clean | -1.640417 | -0.992083 | -12.063785 | 0.00% | 0.00% | 0.00% |
| S_top20 | corrupt | clean_to_corrupt | +2.515833 | +1.058333 | +14.753749 | 1.00% | 1.00% | 0.00% |
| E_top20 | clean | corrupt_to_clean | -0.187917 | -0.113750 | -5.592553 | 0.00% | 0.00% | 0.00% |
| E_top20 | corrupt | clean_to_corrupt | +1.580000 | +0.452083 | +5.656316 | 0.00% | 0.00% | 0.00% |

S:E = (1.640417 + 2.515833) / (0.187917 + 1.580000) = 2.350931

All feature choices are frozen from train; the four causal cells use held-out paired activation values.
