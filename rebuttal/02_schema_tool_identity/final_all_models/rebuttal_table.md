| Manipulation | Qwen3-4B | Qwen3-8B | Qwen3-14B | Qwen3.5-4B | Qwen3.5-9B | Mistral | Granite |
|---|---|---|---|---|---|---|---|
| Renamed | 0.981/1.000/1.000 | 0.994/1.000/1.000 | 0.985/1.000/1.000 | 0.998/0.347/0.975 | 0.970/0.804/0.657 | 0.985/1.000/0.997 | 0.976/1.000/0.937 |
| Removed | 0.888/1.000/1.000 | 0.920/1.000/1.000 | 0.712/1.000/1.000 | 0.910/0.168/0.924 | 0.890/0.679/1.000 | 0.896/1.000/1.000 | 0.878/1.000/1.000 |
| Mismatched | 0.655/0.993/NA | 0.622/1.000/1.000 | 0.544/0.997/1.000 | 0.932/0.067/0.973 | 0.887/0.183/1.000 | 0.630/1.000/1.000 | 0.780/0.830/1.000 |

Each cell is cosine / strict flip / strict drop. Directions are fit on 136 disjoint training pairs after a 64-pair train-only layer selection; every metric uses all 300 v5 held-out pairs.
Strict flip is conditioned on each variant's baseline-corrupt non-tool cases; strict drop is conditioned on its baseline-clean tool cases. `NA` means that condition had a zero denominator; exact counts and denominators are in `table_metrics.csv`.
