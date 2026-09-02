# Qwen3.5 and Transcoder rebuttal code

This directory contains the current model-specific Transcoder analysis rather
than a generic scratch area. `path_defaults.py` is the only local path layer:
it derives repository data/output paths and imports external weight locations
from `src/artifact_paths.py`.

| Script family | Result directory | Purpose |
| --- | --- | --- |
| `run_transcoder_feature_analysis.py`, `run_paper_transcoder_analysis.py` | `rebuttal/09_qwen35_transcoder_feature_analysis/` | Qwen3.5-4B feature decomposition, paper-style K summary, and causal screen |
| `audit_*` | `rebuttal/09_qwen35_transcoder_feature_analysis/` | sensitivity and historical Table-3 protocol audits |
| `run_cross_model_transcoder_k.py` | `rebuttal/10_cross_model_transcoder_k/` | released K summaries/caches for Qwen3.5 and Granite |
| `run_selected_layer_transcoder_causal.py` | `rebuttal/11_selected_layer_transcoder_causal_ablation/` | selected top-20 zero-ablation test |
| `run_selected_layer_transcoder_causal_sweep.py` | `rebuttal/12_selected_layer_causal_sweep/` | k and intervention-mode sweep |
| `run_transcoder_se_dominance.py` | `rebuttal/14_transcoder_se_dominance_v2/` | bidirectional suppressor/execution-feature test |

All of these use the seven frozen v5 200-train/300-heldout datasets. Model
and Transcoder checkpoints remain external. For example, set
`QWEN35_4B_PATH` and `QWEN35_4B_TRANSCODER_PATH` on a destination server; no
runner needs an old server-specific path.

Older conversion and behavior scripts (`parse_qwen_dataset.py`,
`build_500_pair_subset.py`, `run_behavior_scan.py`, and
`run_mechanism_generalization.py`) are preserved as dataset/provenance support
and are not the canonical Transcoder rerun path.
