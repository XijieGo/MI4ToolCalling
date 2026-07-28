# wPFH cross-domain table, Qwen-family candidate

Cells are strict flip / strict drop (%). Exact count denominators are in `qwen_table_values.csv`.

| Domain | Direction | Qwen3-4B | Qwen3-8B | Qwen3-14B | Qwen3.5-4B | Qwen3.5-9B | Mistral | Granite |
|---|---|---|---|---|---|---|---|---|
| Retrieval | Code | 76/100 | 100/98 | 95/83 | 100/100 | 100/100 | xx/xx | xx/xx |
| Retrieval | Random | 0/0 | 0/0 | 0/1 | 43.1/7 | 32.8/6 | xx/xx | xx/xx |
| SQL | Code | 58/100 | 100/100 | 42/0 | 75/41.7 | 59/100 | xx/xx | xx/xx |
| SQL | Random | 0/0 | 0/0 | 0/0 | 0/0 | 0/42.5 | xx/xx | xx/xx |
| Email | Code | 100/100 | 100/100 | 32/0 | 100/92 | 100/97 | xx/xx | xx/xx |
| Email | Random | 0/0 | 0/0 | 0/0 | 5.7/0 | 32/0 | xx/xx | xx/xx |

All five Qwen-family artifacts passed completion, held-out cardinality, norm-match, and strict-rate denominator audits. Qwen3 Code values additionally exactly reproduce their prior frozen matrices.
