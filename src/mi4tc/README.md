# Shared experiment utilities

The `mi4tc` package provides shared utilities for the experiment runners. Follow the [project README](../../README.md) for installation and path configuration.

| Module | Purpose |
|---|---|
| `paths.py` | Repository paths and model / checkpoint configuration |
| `io.py` | JSONL, JSON, CSV and SHA-256 utilities |
| `pairs.py` | Paired-prompt loading and native token-ID checks |
| `model.py`, `native_mistral.py` | Model adapters and native prompt encoding |
| `directions.py`, `metrics.py` | Vector estimation and intervention metrics |
| `features.py`, `transcoder.py` | Feature accounting and Transcoder inference |
| `readout.py` | Downstream readout utilities |
| `legacy_qwen.py` | TransformerLens loading for MLP patch comparisons |
