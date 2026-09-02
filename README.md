# MI4 Tool-Calling

This is the standalone migration bundle for the paper and its rebuttal. It
contains the paper source and figures, frozen data, intervention vectors,
per-example records, final summaries, and the code that produced them. The
only intentionally external dependencies are model weights and Transcoder
weights.

## Standalone migration contract

After copying all working-tree files, every item in the experiment registry can
resolve its registered code, data, vectors, and evidence from this directory
alone. The paper source-map entries preserve the core code/data lineage; the
active rebuttal entries additionally preserve their frozen per-example
evidence. A Git-based transfer must materialize Git-LFS payloads rather than
leave pointer files. Nothing in the supported migration surface reads the
retired synchronization workspace.

Run the no-model structural check after transfer:

```bash
python scripts/verify_standalone.py
```

It validates the experiment registry, v2/v5 cardinalities, rebuttal evidence,
source syntax, LFS payloads, and the absence of retired-workspace paths in
canonical sources. `python scripts/verify_release.py` retains the original
release-level checks and frozen-vector hashes.

For a Git-based migration, do not rely only on the committed release: the
current Qwen3.5/Transcoder rebuttal code and evidence are present in this
working tree and must be added before creating a new commit. A direct
directory transfer already includes them.

## Start here

- [experiments/README.md](experiments/README.md) is the canonical map from a
  scientific question to frozen data, code, and evidence.
- [rebuttal/README.md](rebuttal/README.md) gives the reviewer-item-level
  evidence chain.
- [src/README.md](src/README.md) distinguishes active code from retained
  historical exploration.
- [paper/neurips_2026.tex](paper/neurips_2026.tex) is the paper entry point;
  its source, bibliography, figures, and rebuttal materials are included in
  this checkout.

## Layout

```text
datasets/      frozen v2/v4/v5 data, provenance, and tau2 trajectories
experiments/   canonical experiment registry and migration map
paper/         NeurIPS source snapshot, figures, and rebuttal materials
rebuttal/      frozen rebuttal records, vectors, summaries, and tables
src/           data construction, intervention, measurement, and rendering
scripts/       no-model structural validation
```

Paths for external weights are centralized in `src/artifact_paths.py`. By
default they resolve under `external/models/` and `external/transcoders/` in
this repository; set `MODEL_ROOT`, `TRANSCODER_ROOT`, or a model-specific
environment variable on the destination server. Large immutable vectors and
tau2 JSONL files are configured for Git LFS in `.gitattributes`.
