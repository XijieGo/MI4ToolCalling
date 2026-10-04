# mi4tc

Shared pieces used by more than one experiment:

- `paths.py` — repo root and external model/checkpoint paths
- `io.py` — JSONL, JSON, CSV, SHA-256
- `pairs.py` — paired-prompt loading and native token-id checks
- `model.py` — Hugging Face adapters, including Mistral native IDs
- `directions.py`, `metrics.py`, `features.py`, `readout.py`

Model libraries are imported only when a runner loads a model.
