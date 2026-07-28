# L5--L20 verb-to-prediction head attribution

For each head, `attention` is the final-prompt-position mass on the changed verb token(s). `mu linear write` is `<W_O z, mu_hat>` at the same prediction position. Both are evaluated on the held-out v2_1500 test pairs.

## Top linear transport heads

| rank | head | Δ attention (clean-corrupt) | Δ μ write | paired Pearson r | transport score |
|---:|---|---:|---:|---:|---:|
| 1 | L20H29 | -0.02745 | +0.23482 | -0.643 | 0.00414338 |
| 2 | L19H29 | -0.03135 | +0.14386 | -0.741 | 0.00334293 |
| 3 | L20H14 | -0.01532 | +0.18342 | -0.855 | 0.00240372 |
| 4 | L19H31 | -0.00721 | +0.71650 | -0.445 | 0.00229819 |
| 5 | L18H19 | -0.00888 | +0.20315 | -0.899 | 0.00162183 |
| 6 | L20H21 | -0.03531 | +0.04457 | -0.751 | 0.00118258 |
| 7 | L19H10 | +0.00602 | +0.36654 | +0.449 | 0.0009892 |
| 8 | L16H7 | -0.03267 | +0.03307 | -0.813 | 0.00087807 |
| 9 | L20H17 | -0.00505 | +0.15301 | -0.756 | 0.000584797 |
| 10 | L20H13 | -0.00529 | -0.13861 | +0.745 | 0.000546112 |
| 11 | L19H13 | -0.01568 | +0.06161 | -0.540 | 0.0005213 |
| 12 | L20H16 | -0.00402 | +0.11920 | -0.912 | 0.000437083 |

## Causal check: clean head-z patched into corrupt run

| head | Δ L24 μ coordinate | Δ tool logit | strict recovery |
|---|---:|---:|---:|
| L19H31 | +0.0708 | +0.5375 | 31.33% |
| L20H14 | +0.0510 | +0.1529 | 13.00% |
| L20H29 | +0.0363 | +0.3904 | 22.00% |
| L19H10 | +0.0340 | +0.3250 | 10.00% |
| L20H17 | +0.0291 | +0.2862 | 18.33% |
| L20H13 | -0.0277 | -0.2000 | 0.33% |
| L20H16 | +0.0222 | -0.2558 | 0.67% |
| L19H29 | +0.0188 | +0.2713 | 8.67% |
| L18H19 | +0.0162 | +0.0033 | 1.67% |
| L16H7 | +0.0106 | +0.0296 | 3.67% |
| L19H13 | +0.0061 | -0.0288 | 2.67% |
| L20H21 | +0.0016 | +0.0113 | 1.33% |

The causal check validates head outputs at the prediction position. It is deliberately narrower than a full edge-patching proof, so the rebuttal should describe it as a tested bridge rather than a complete upstream circuit.
