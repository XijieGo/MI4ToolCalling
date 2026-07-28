# Held-out L24 +μΔ failure audit

- Split: frozen v2_1500 test manifest, N=300; the vector is fit only on v2_1500 train.
- Baseline `<tool_call>` top-1: 3.67% (11/300).
- After `+μΔ`: 99.00% (297/300).
- Strict non-tool→tool recoveries: 95.33% (286/300).
- Unrecovered baseline-non-tool cases: 3.

The CSV files retain every held-out item and the exact contingency summaries. With a small failure count, the per-category Fisher tests are descriptive rather than evidence for a broad population claim.

## Failure IDs

- `mbpp_python_96`: source=mbpp, language=python, corrupt verb=discuss, tokens=200, patched rank=2, coordinate=0.608.
- `apps_python_366`: source=apps, language=python, corrupt verb=discuss, tokens=212, patched rank=2, coordinate=0.634.
- `apps_python_469`: source=apps, language=python, corrupt verb=discuss, tokens=210, patched rank=2, coordinate=0.599.
