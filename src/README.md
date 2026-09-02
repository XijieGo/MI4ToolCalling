# Source layout

Only scripts named by [the experiment registry](../experiments/registry.json)
are supported migration entrypoints or paper source-map code. This keeps a
clear boundary between the current paper/rebuttal pipeline and retained
exploratory work.

| Directory | Role | Status |
| --- | --- | --- |
| `paper_repro/` | v2 and v5 dataset construction/audits | Canonical data lineage |
| `shared/` | shared localization, patching, readout, and generalization utilities | Canonical paper support |
| `qwen3_8b/` | core Qwen3-8B mechanism and LxEy controls | Canonical paper/rebuttal |
| `multidomain/` | D1/D3/D4/D5 scaffold, schema, and transfer experiments | Canonical rebuttal |
| `natural_trajectory/` | tau2 selection and bidirectional intervention | Canonical rebuttal |
| `rebuttal/` | schema identity, rank/probability, and table renderers | Canonical rebuttal |
| `qwen35/` | Qwen3.5 and Transcoder audits; see its README | Canonical with rebuttal 09-14 evidence |
| `granite/`, `mistral/`, `devstral/` | model-specific adapters and historical conversion support | Supporting code |
| `toolcall_circuit/` | reusable legacy helpers plus exploratory scripts | Retained for compatibility; not a canonical entrypoint set |

`artifact_paths.py` is the single source for external model and Transcoder
locations. New runners should import it (or their local `path_defaults.py`)
rather than embed a server-specific absolute path.

The registry uses `paper_source_map` for the core paper's preserved code/data
lineage and `standalone_rebuttal` for the fully frozen current rebuttal
experiments. Historical scripts remain available for provenance, but are not
part of the migration contract unless the registry explicitly names them.
