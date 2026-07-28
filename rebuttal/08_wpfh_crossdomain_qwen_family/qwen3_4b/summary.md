# Qwen3-4B: frozen Coding transfer with Random control

Code re-evaluates the frozen D1 vector from the prior target-norm matrix. Random is one seeded Gaussian unit direction, rescaled separately to each target native norm.

| domain | direction | strict flip / strict drop | eligible corrupt / clean |
|---|---|---:|---:|
| Retrieval | Code | 76/100 | 100/100 |
| Retrieval | Random | 0/0 | 100/100 |
| SQL | Code | 58/100 | 100/100 |
| SQL | Random | 0/0 | 100/100 |
| Email | Code | 100/100 | 100/100 |
| Email | Random | 0/0 | 100/100 |
