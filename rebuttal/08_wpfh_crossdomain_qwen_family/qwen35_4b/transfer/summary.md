# Qwen3.5-4B: frozen Coding cross-domain transfer

Each row is evaluated on the fixed 100-pair held-out split. Code is the D1 mean-difference direction at the previously frozen Coding layer, rescaled to the target native norm. Random is a seeded equal-norm Gaussian direction.

| domain | direction | strict flip / strict drop | eligible corrupt / clean |
|---|---|---:|---:|
| Retrieval | Code | 100/100 | 51/86 |
| Retrieval | Random | 43/7 | 51/86 |
| SQL | Code | 75/42 | 100/60 |
| SQL | Random | 0/0 | 100/60 |
| Email | Code | 100/92 | 53/100 |
| Email | Random | 6/0 | 53/100 |
