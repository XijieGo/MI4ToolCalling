# Granite-3.3-8B-Instruct: Transcoder S/E causal dominance

Formation window: L29,L30,L31,L32; decision layer: L36; held-out pairs: 300.
S is selected from corrupt-higher / write-away aligned features; E is clean-higher / write-toward aligned.

| group | target side | transfer | mean Δm | mean Δ tool logit | mean Δ gate | top-1 flip | margin flip | logit-midpoint flip |
|---|---|---|---:|---:|---:|---:|---:|---:|
| S_top20 | clean | corrupt_to_clean | +0.690833 | +0.287917 | -2.093503 | 0.00% | 0.00% | 0.00% |
| S_top20 | corrupt | clean_to_corrupt | +0.028750 | -0.025833 | +2.070965 | 0.00% | 0.00% | 0.00% |
| E_top20 | clean | corrupt_to_clean | +0.670833 | +0.117917 | -3.483852 | 0.00% | 0.00% | 0.00% |
| E_top20 | corrupt | clean_to_corrupt | +0.837708 | +0.700625 | +4.540357 | 0.00% | 0.00% | 0.67% |

S:E = (0.690833 + 0.028750) / (0.670833 + 0.837708) = 0.477006

All feature choices are frozen from train; the four causal cells use held-out paired activation values.
