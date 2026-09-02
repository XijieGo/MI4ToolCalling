# Canonical experiments

This directory is the short operational map for a migrated checkout. The
machine-readable source of truth is [registry.json](registry.json); it is also
checked by `python scripts/verify_standalone.py`.

The registry has two tiers: `paper_source_map` preserves the core paper's
current code/data lineage, while `standalone_rebuttal` adds frozen inputs,
vectors, per-example records, and summaries needed to inspect or rerun the
current response without the retired workspace.

| Experiment | Scientific role | Frozen inputs | Canonical code | Frozen evidence |
| --- | --- | --- | --- | --- |
| Core v2 localization | A paired verb change controls tool-call versus analysis behavior; localize and causally test the decision direction at Qwen3-8B L24. | `datasets/train`, `datasets/test`, `datasets/provenance/v2_1500` | `shared/run_residual_patch_sweep.py`, `qwen3_8b/phase7_l24_*` | v2 manifests and linked rebuttal vectors |
| Formation/readout mechanism | Test upstream MLP suppression, gate behavior, attention/readout, and late vocabulary writing. | v2 train/test | `qwen3_8b/phase8_*`, `phase9_*`, `shared/measure_section6_readout_attention.py` | linked LxEy controls in rebuttal 07 |
| Cross-model generalization | Re-evaluate the same decision logic on each model's native 500-pair release. | `datasets/v5_model_specific_balanced` | `shared/run_core_generalization.py` | v5 manifests and rebuttal tables |
| Schema/tool identity | Separate request intent from schema identity and test direction interventions. | v4/v5 bundles and frozen vectors | `rebuttal/run_cjrq_v5_tool_identity.py`, `multidomain/run_tool_identity_ablation.py` | rebuttal 02 |
| Scaffold and request ladder | Attribute behavior to R/T/F scaffold components and graded request cues. | v4/v5 bundles | `multidomain/run_scaffold_component_ablation.py`, `run_request_ladder_ablation.py` | rebuttal 03 |
| tau2 natural trajectories | Apply frozen coding directions to naturally occurring tool-use trajectories. | `datasets/external` and coding vectors | `natural_trajectory/run_tau2_bidirectional_200_intervention.py` | rebuttal 05 |
| Implicit intent | Test removal on frozen no-explicit-verb requests with native model renderers. | `rebuttal/06_.../artifacts/implicit_intent_oversampled_600.jsonl` | `artifacts/run_cross_model_removal.py` | rebuttal 06 |
| LxEy controls | Audit heads, neighbor layers, boundary verbs, and rank/probability. | v2/v4/v5 bundles | `qwen3_8b/rebuttal_lxey_*`, `rebuttal/run_lxey_v5_*` | rebuttal 04 and 07 |
| Qwen-family transfer | Test common directions across model/domain combinations. | v4/v5 bundles | `multidomain/run_common_direction_causal_ablation.py` | rebuttal 08 |
| Transcoder audits | Feature decomposition, K, selected-feature causal tests, sweeps, and S/E dominance. | v5 bundles and released caches | `qwen35/run_*`, `qwen35/audit_*` | rebuttal 09-14 |

`paper/` contains the migrated paper snapshot: its entry point, included TeX
sources, bibliography, figures, and rebuttal materials are checked by the
standalone validators. `docs/` and a monolithic `scripts/run_paper.py` remain
intentionally excluded from the migration surface.
