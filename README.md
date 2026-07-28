# MI4 Tool-Calling: anonymous rebuttal evidence

This anonymous repository contains the construction inputs, frozen datasets,
intermediate computations, intervention vectors, final summaries, and code
needed to inspect and rerun every quantitative result reported in the
rebuttal.

## Start here

Open [rebuttal/README.md](rebuttal/README.md). It maps every reviewer item to
its intermediate records, calculation, final result, and corresponding code.

To verify the release layout without loading a model, run:

```bash
python scripts/verify_release.py
```

The check validates the indexed evidence, all seven model-specific manifests,
the frozen τ² vectors, and the absence of references to the private
synchronization workspace.

## Layout

```text
rebuttal/   evidence for every rebuttal table and quantitative claim
datasets/   frozen source data, manifests, and τ² trajectories
src/        construction, intervention, aggregation, and rendering code
scripts/    release validation
```

Model and Transcoder weights are external dependencies. They are intentionally
not included; all other inputs, vectors, intermediate records, and summaries
listed in `rebuttal/README.md` are included in this repository.

Large immutable vectors and τ² JSONL files are configured for Git LFS in
`.gitattributes`.
