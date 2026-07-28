# Boundary-verb L24 projection sweep

Every variant substitutes only the leading request verb in a held-out v2 test task body. The L24 direction is frozen from all 1,200 v2 train pairs.

| verb | class | top-1 tool-call | mean L24 coordinate | SE | mean rank |
|---|---|---:|---:|---:|---:|
| discuss | anchor_analysis | 0.0% | -0.222 | 0.005 | 13.5 |
| assess | boundary_analysis | 0.0% | -0.080 | 0.006 | 7.4 |
| analyze | boundary_analysis | 6.3% | -0.030 | 0.006 | 5.4 |
| explore | anchor_analysis | 4.7% | 0.005 | 0.007 | 6.7 |
| examine | boundary_analysis | 3.3% | 0.012 | 0.006 | 5.4 |
| review | anchor_analysis | 9.3% | 0.068 | 0.004 | 4.5 |
| inspect | boundary_analysis | 54.3% | 0.076 | 0.006 | 3.5 |
| study | anchor_analysis | 21.7% | 0.080 | 0.006 | 4.1 |
| summarize | boundary_analysis | 18.7% | 0.113 | 0.006 | 3.3 |
| compare | boundary_analysis | 60.3% | 0.124 | 0.005 | 1.7 |
| complete | anchor_execution | 100.0% | 0.794 | 0.004 | 1.0 |
| implement | boundary_execution | 100.0% | 0.830 | 0.004 | 1.0 |
| modify | boundary_execution | 100.0% | 0.867 | 0.004 | 1.0 |
| build | anchor_execution | 100.0% | 0.912 | 0.004 | 1.0 |
| create | boundary_execution | 100.0% | 0.941 | 0.003 | 1.0 |
| update | boundary_execution | 100.0% | 0.950 | 0.003 | 1.0 |
| write | anchor_execution | 100.0% | 0.951 | 0.003 | 1.0 |
| generate | boundary_execution | 100.0% | 0.954 | 0.003 | 1.0 |
| add | anchor_execution | 100.0% | 0.992 | 0.003 | 1.0 |
| save | anchor_execution | 100.0% | 1.110 | 0.002 | 1.0 |

Across the 20 verbs, Spearman ρ between top-1 call rate and mean L24 coordinate is 0.923 (two-sided p=7.09e-09).
The raw per-item values are retained in `projection_per_sample.csv`.
