# Transcoder S/E dominance across available models

| Model | S→clean Δm | E→clean Δm | S→corrupt Δm | E→corrupt Δm | S:E |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B | -0.632 | -0.665 | -0.873 | +3.688 | 0.346 |
| Qwen3-8B | -1.640 | -0.188 | +2.516 | +1.580 | 2.351 |
| Qwen3-14B | -0.481 | -2.824 | +0.867 | +8.215 | 0.122 |
| Qwen3.5-4B | +0.078 | -0.002 | +0.159 | +0.172 | 1.359 |
| Qwen3.5-9B | -0.067 | +0.141 | -0.026 | +0.192 | 0.280 |
| Granite-3.3-8B-Instruct | +0.691 | +0.671 | +0.029 | +0.838 | 0.477 |

Mistral-Small-3.2-24B is not included because no matching Transcoder checkpoint is present under /root/autodl-tmp/Transcoder.
