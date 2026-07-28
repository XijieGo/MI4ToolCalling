# Qwen3.5-9B: frozen Coding cross-domain transfer

Each row is evaluated on the fixed 100-pair held-out split. Code is the D1 mean-difference direction at the previously frozen Coding layer, rescaled to the target native norm. Random is a seeded equal-norm Gaussian direction.

| domain | direction | strict flip / strict drop | eligible corrupt / clean |
|---|---|---:|---:|
| Retrieval | Code | 100/100 | 64/100 |
| Retrieval | Random | 33/6 | 64/100 |
| SQL | Code | 59/100 | 100/80 |
| SQL | Random | 0/42 | 100/80 |
| Email | Code | 100/97 | 97/100 |
| Email | Random | 32/0 | 97/100 |
