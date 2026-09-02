# Qwen3-14B: Transcoder S/E causal dominance

Formation window: L29,L30,L31,L32; decision layer: L33; held-out pairs: 300.
S is selected from corrupt-higher / write-away aligned features; E is clean-higher / write-toward aligned.

| group | target side | transfer | mean Δm | mean Δ tool logit | mean Δ gate | top-1 flip | margin flip | logit-midpoint flip |
|---|---|---|---:|---:|---:|---:|---:|---:|
| S_top20 | clean | corrupt_to_clean | -0.481250 | -0.110417 | -30.613744 | 0.33% | 0.00% | 0.00% |
| S_top20 | corrupt | clean_to_corrupt | +0.867344 | +0.550677 | +23.763866 | 0.00% | 0.00% | 0.00% |
| E_top20 | clean | corrupt_to_clean | -2.824167 | -1.549167 | -28.615958 | 1.67% | 1.67% | 0.00% |
| E_top20 | corrupt | clean_to_corrupt | +8.214844 | +5.486927 | +38.198822 | 12.67% | 12.67% | 20.00% |

S:E = (0.481250 + 0.867344) / (2.824167 + 8.214844) = 0.122166

All feature choices are frozen from train; the four causal cells use held-out paired activation values.
