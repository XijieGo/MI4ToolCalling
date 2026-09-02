# Implicit-intent artifact bundle

The standalone rerun starts from the frozen, release-contained 600-item
collection:

```text
implicit_intent_oversampled_600.jsonl
    -> native rendering and baseline screen
    -> up to 10 baseline-positive rows per domain/pattern cell
    -> frozen coding-direction removal and random-direction control
```

Use:

```bash
python rebuttal/06_implicit_intent_tool_calls/artifacts/run_cross_model_removal.py \
  --models qwen3_8b
```

The default output is `artifacts/reruns/cross_model_removal/`, so it never
overwrites the frozen evidence in `cross_model_removal_20260726/`. Model paths
come from `src/artifact_paths.py`; vectors and native rendering references are
resolved by `release_paths.py` inside this directory.

Frozen files and their roles:

- `implicit_intent_oversampled_600.jsonl`: canonical release input.
- `implicit_intent_oversampled_600_screened.jsonl`: the original Qwen3-8B
  baseline screen.
- `removal_arm_161.jsonl`: the original Qwen3-8B selected arm and its legacy
  causal record.
- `cross_model_removal_20260726/`: native-rendered, per-model 600-item
  baseline records, selected arms, intervention records, and summaries.

`build_implicit_intent_set.py` is retained only as historical construction
provenance: its raw D3-D5 source pool was not part of this release. It is not
needed to reproduce the frozen experiment. The smaller Qwen3-8B
`screen_implicit_intent.py`, `select_final.py`, `run_removal_causal.py`, and
`merge_final.py` are portable alternate stages; each writes only to `reruns/`
by default.
