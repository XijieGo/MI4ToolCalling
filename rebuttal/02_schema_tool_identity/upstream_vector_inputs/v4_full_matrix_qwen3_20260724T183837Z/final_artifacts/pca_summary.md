# PCA Common-direction Summary

The primary common-direction measure is the rank-1 explained energy of an uncentered SVD over the four L2-normalized domain vectors. It directly measures whether one direction reconstructs all four vectors; the conventional centered-PCA value is included separately.

| model | rank-1 common energy | centered PC1 variance | mean pairwise cosine | min pairwise cosine | min cosine to common |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 65.1% | 55.1% | 0.522 | 0.326 | 0.603 |
| Qwen3-14B | 57.4% | 57.9% | 0.394 | 0.087 | 0.357 |
| Qwen3-4B | 65.6% | 52.6% | 0.532 | 0.392 | 0.635 |
| Qwen3-8B | 65.6% | 60.6% | 0.522 | 0.289 | 0.551 |
