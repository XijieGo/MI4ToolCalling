# Qwen3.5-9B tau2-bench Telecom Traces (Judged)

## Dataset

Execution traces from Qwen3.5-9B on [tau2-bench](https://github.com/sierra-research/tau2-bench) telecom domain.

- **Model**: Qwen3.5-9B with thinking mode (temperature=0.6, top_p=0.95)
- **Domain**: Telecom customer service (MMS, mobile data, service issues)
- **Traces**: 1,625 total
- **Oracle reward**: 442 pass (27.2%), 1,183 fail
- **Judge scores**: 1,321 traces scored by GPT-5.4 (v2 rubric, 3 dimensions)
- **max_steps**: 50

## Format

Each line is a JSON object with:
- `task_id`: tau2-bench task identifier with sub-issues
- `task_ticket`: Customer's issue description
- `messages`: Full conversation (system, user, assistant, tool messages)
- `tool_calls`: List of tool calls with names and arguments
- `reward`: Oracle reward (0.0 or 1.0) from tau2-bench environment
- `termination_reason`: USER_STOP (resolved) or MAX_STEPS (timed out)
- `judge`: GPT-5.4 process quality scores (v2 rubric):
  - `action_quality` (0-5): Correct tools for correct problems
  - `execution_efficiency` (0-5): Minimal wasted steps
  - `task_completion` (0-5): Progress toward resolution
  - `overall_quality` (0-5): Holistic quality
  - `process_quality`: Mean of 3 dimensions
  - `*_reasoning`: Chain-of-thought per dimension

## SFT Filtering

Recommended filter for SFT: `judge.task_completion >= 5` (495 traces, 80% precision, 94.5% recall)

## Usage

```python
from huggingface_hub import hf_hub_download
import json

path = hf_hub_download("KermitCO/qwen3.5-9B-tau2bench-telecom-traces", 
                        "tau2_baseline_judged.jsonl", repo_type="dataset")
traces = [json.loads(l) for l in open(path)]
```
