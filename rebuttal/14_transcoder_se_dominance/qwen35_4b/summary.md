# Qwen3.5-4B: Transcoder S/E causal dominance

Formation window: L26,L27,L28,L29; decision layer: L29; held-out pairs: 300.
S is selected from corrupt-higher / write-away aligned features; E is clean-higher / write-toward aligned.

| group | target side | transfer | mean Δm | mean Δ tool logit | mean Δ gate | top-1 flip |
|---|---|---|---:|---:|---:|---:|
| S_top20 | clean | corrupt_to_clean | +0.077917 | +0.068333 | -0.412296 | 0.00% |
| S_top20 | corrupt | clean_to_corrupt | +0.159375 | -0.001042 | +0.296823 | 0.00% |
| E_top20 | clean | corrupt_to_clean | -0.002083 | -0.168750 | -0.419530 | 0.00% |
| E_top20 | corrupt | clean_to_corrupt | +0.172500 | +0.220833 | +0.393017 | 0.00% |

S:E = (0.077917 + 0.159375) / (0.002083 + 0.172500) = 1.359189

All feature choices are frozen from train; the four causal cells use held-out paired activation values.
