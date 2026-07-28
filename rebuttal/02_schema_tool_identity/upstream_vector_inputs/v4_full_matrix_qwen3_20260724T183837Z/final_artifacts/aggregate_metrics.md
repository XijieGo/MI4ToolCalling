# Aggregate Matrix Metrics

Off-diagonal rows summarize the 12 genuine cross-domain directions; strict flip/drop use the behavioral top-1 decision.

| model | condition | scope | mean Suff. | mean Necc. | mean flip | mean drop | both 100% |
|---|---|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | target_norm_aligned | all | 0.874 | 1.343 | 90.6% | 92.5% | 43.8% |
| Qwen3-1.7B | target_norm_aligned | diagonal | 1.046 | 1.034 | 99.0% | 94.4% | 50.0% |
| Qwen3-1.7B | target_norm_aligned | off_diagonal | 0.816 | 1.445 | 87.8% | 91.9% | 41.7% |
| Qwen3-1.7B | raw_1p5 | all | 0.716 | 3.270 | 83.3% | 83.8% | 56.2% |
| Qwen3-1.7B | raw_1p5 | diagonal | 0.965 | 2.692 | 100.0% | 100.0% | 100.0% |
| Qwen3-1.7B | raw_1p5 | off_diagonal | 0.633 | 3.462 | 77.8% | 78.3% | 41.7% |
| Qwen3-14B | target_norm_aligned | all | 0.611 | 0.257 | 56.2% | 18.8% | 0.0% |
| Qwen3-14B | target_norm_aligned | diagonal | 0.648 | 0.137 | 63.0% | 7.0% | 0.0% |
| Qwen3-14B | target_norm_aligned | off_diagonal | 0.598 | 0.296 | 53.9% | 22.8% | 0.0% |
| Qwen3-14B | raw_1p5 | all | 0.764 | 0.468 | 74.3% | 30.1% | 6.2% |
| Qwen3-14B | raw_1p5 | diagonal | 0.800 | 0.258 | 85.0% | 13.5% | 0.0% |
| Qwen3-14B | raw_1p5 | off_diagonal | 0.753 | 0.538 | 70.8% | 35.6% | 8.3% |
| Qwen3-4B | target_norm_aligned | all | 0.879 | 0.746 | 84.6% | 98.1% | 37.5% |
| Qwen3-4B | target_norm_aligned | diagonal | 0.965 | 0.704 | 95.0% | 95.5% | 50.0% |
| Qwen3-4B | target_norm_aligned | off_diagonal | 0.851 | 0.760 | 81.2% | 98.9% | 33.3% |
| Qwen3-4B | raw_1p5 | all | 0.942 | 1.367 | 95.4% | 100.0% | 68.8% |
| Qwen3-4B | raw_1p5 | diagonal | 1.066 | 1.296 | 100.0% | 100.0% | 100.0% |
| Qwen3-4B | raw_1p5 | off_diagonal | 0.901 | 1.390 | 93.8% | 100.0% | 58.3% |
| Qwen3-8B | target_norm_aligned | all | 0.862 | 0.918 | 97.8% | 96.4% | 43.8% |
| Qwen3-8B | target_norm_aligned | diagonal | 0.957 | 0.839 | 99.8% | 97.5% | 50.0% |
| Qwen3-8B | target_norm_aligned | off_diagonal | 0.831 | 0.944 | 97.2% | 96.1% | 41.7% |
| Qwen3-8B | raw_1p5 | all | 0.959 | 1.541 | 99.9% | 100.0% | 93.8% |
| Qwen3-8B | raw_1p5 | diagonal | 1.086 | 1.372 | 100.0% | 100.0% | 100.0% |
| Qwen3-8B | raw_1p5 | off_diagonal | 0.916 | 1.598 | 99.8% | 100.0% | 91.7% |
