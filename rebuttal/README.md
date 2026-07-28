# Rebuttal evidence index

This directory is the evidence bundle for the rebuttal. All paths below are
relative to the repository root.

For each result, follow the same chain:

```text
frozen inputs / provenance → per-prompt or per-turn records → summary → code
```

Run `python scripts/verify_release.py` to check that the indexed files are
present and that no external synchronization-workspace path has entered the
public release.

## Numerical tables

| Rebuttal item | Intermediate records and calculation | Final result | Code |
| --- | --- | --- | --- |
| CJrQ §1: renamed, removed, and mismatched schemas | For every model, `02_schema_tool_identity/final_all_models/<model>/sample_metrics.csv` stores native baseline decisions and `intervention_long.csv` stores the ±μΔ decisions for each held-out pair and schema. Strict flips/drops are counts over the corresponding baseline stratum. | `<model>/table_metrics.json`; combined `table_metrics.csv` and `table_audit.json`. | `src/rebuttal/run_cjrq_v5_tool_identity.py`; `src/rebuttal/render_cjrq_v5_table.py` |
| CJrQ §1: affordance-reversal 2×2 | `02_schema_tool_identity/affordance_reversal_qwen3_8b_v4_20260725/affordance_reversal_2x2_samples.csv` has all 400 held-out rows (100 per verb/schema cell), with first-token logit, probability, rank, and top-1 decision. | `affordance_reversal_2x2.csv`; frozen vectors in `native_vectors/`. | `src/multidomain/run_tool_identity_ablation.py` |
| CJrQ §2; LxEy §1; wPFH §2: τ²-bench | Native contexts and selected 200-turn arms are in `datasets/external/tau2_*`. Each model's `suppression_per_sample.jsonl` and `induction_per_sample.jsonl` stores baseline and intervened first-token outcomes for mean and random directions. | `05_tau2_bench_200/final_bidirectional_results/tau2_bidirectional_200_interventions_20260727_canonical_batch1/run_summary.json`. | `src/natural_trajectory/run_tau2_bidirectional_200_intervention.py` |
| CJrQ §3: 600 implicit-intent requests | Construction/rejection records are in `06_implicit_intent_tool_calls/artifacts/implicit_intent_*.jsonl`. For each model, `rendered_600.jsonl` → `baseline_600.jsonl` → `removal_arm.jsonl` records construction, baseline eligibility, and each intervention. | Per-model `summary.json`; `cross_model_summary.json`. | `build_implicit_intent_set.py`, `run_cross_model_removal.py`, and `summarize_removal.py` in the same artifact directory. |
| LxEy §2: request ladder | `03_scaffold_ablation_and_request_ladder/final_v5_qwen3_8b/request_ladder/sample_metrics.csv` has one row per held-out prompt and ladder level. Top-1 rate and mean probability are grouped by level. | `ladder_summary.csv`; `summary/request_ladder_table.md`. | `src/multidomain/run_request_ladder_ablation.py` |
| wPFH §4: R/T/F scaffold factorial | `03_scaffold_ablation_and_request_ladder/final_v5_qwen3_8b/components/sample_metrics.csv` records each scaffold × prompt × request condition; `intervention_long.csv` retains the matched vector interventions. Neutral and analysis probabilities are averaged within scaffold. | `components/scaffold_component_results.json`; `summary/scaffold_components_table.md`. | `src/multidomain/run_scaffold_component_ablation.py`; `src/multidomain/summarize_scaffold_ablation.py` |
| LxEy §3: three-head table | `07_lxey_mechanistic_controls/lxey_other_concerns_v2_1500/head_attribution/top_head_per_sample_attribution.csv` stores the 300-pair attention and μΔ-write measurements; `head_sweep_cache.pt` preserves the raw head-state cache. Cross-domain patches are stored row-wise in `lxey_crossdomain_heads_v4/D{3,4,5}/causal_patch_per_sample.csv`. | `head_linear_attribution_summary.csv`; `causal_head_z_patch_summary.csv`; `lxey_crossdomain_heads_v4/summary.md`. | `src/qwen3_8b/rebuttal_lxey_head_attribution.py`; `src/qwen3_8b/rebuttal_lxey_crossdomain_head_roles.py` |
| LxEy §4: rank and probability | Each model's `04_rank_probability/final_heldout_all_models/<model>/sample_metrics.csv` stores the final marker logit, probability, rank, and top-1 outcome for all 300 held-out suppressed prompts. Top-k rates and medians are derived directly from these rows. | Per-model `summary.json`; root `table_metrics.csv` and `rebuttal_table.md`. | `src/rebuttal/run_lxey_v5_rank_probability.py` |
| LxEy §5: L22–L25 sweep | `07_lxey_mechanistic_controls/lxey_v5_neighbor_layers_20260728T040846Z/per_sample_metrics.csv` contains all four layers × four conditions × 300 held-out pairs. Each μΔ is fit on the separate 200-pair training split; vectors are in `vectors/`. | `summary.json`; `summary.md`. | `src/rebuttal/run_lxey_v5_neighbor_layers.py` |

## Other quantitative claims

| Claim | Evidence | Code |
| --- | --- | --- |
| LxEy §6: 11/300 baseline, 297/300 recovery, and three failures | `07_lxey_mechanistic_controls/lxey_other_concerns_v2_1500/failure_audit/heldout_mu_delta_per_sample.csv`, `failure_cases.csv`, and `summary.md`. | `src/qwen3_8b/rebuttal_lxey_other_concerns.py` |
| LxEy §6: 20-verb boundary sweep | `07_lxey_mechanistic_controls/lxey_other_concerns_v2_1500/boundary_verbs/projection_per_sample.csv` and `summary.md`. | `src/qwen3_8b/rebuttal_lxey_other_concerns.py` |
| CJrQ §3: thinking-trace illustration | Prompt, intervention settings, ranks, and continuations are retained in `06_implicit_intent_tool_calls/thinking_trace_support/reviewer_cjrq_vector_thinking_post_probe_long_20260725/{results.json,summary.md}`. This is an illustration, not an aggregate table. | Supporting probes: `thinking_trace_support/cross_domain_pilot_20260724/*.py` |
| wPFH cross-domain Qwen-family support | Model/domain membership, vectors, denominators, and matrix entries are in `08_wpfh_crossdomain_qwen_family/`; use `qwen3_{4b,8b,14b}/matrix_long.csv` or `qwen35_{4b,9b}/transfer/matrix_long.csv` for each model's exact matrix. | `src/multidomain/run_common_direction_causal_ablation.py`; `src/multidomain/build_wpfh_qwen_table.py` |

## Inputs and reruns

- Frozen v2 lineage and manifests: `datasets/provenance/`.
- Seven model-specific 500-pair releases: `datasets/v5_model_specific_balanced/`.
- τ² raw trajectories and selected collections: `datasets/external/`.
- Model and Transcoder weights are external dependencies; all other inputs,
  vectors, intermediate records, and summaries above are in this repository.
