# Full Cross-domain Causal-transfer Matrices

Each model has two 4×4 directed tables. The intervention layer is L24 `hook_resid_pre` at the final prompt token.
`target_norm_aligned`: source direction rescaled to the target-native L2 norm. `raw_1p5`: native source vector ×1.5.

## Qwen3-1.7B — Target-norm aligned (α=1)

Each cell: `S` normalized sufficiency / `N` normalized necessity; `F/D` strict flip/drop.

| source \ target | D1 | D3 | D4 | D5 |
|---|---:|---:|---:|---:|
| D1 | S=1.130<br>N=1.311<br>F/D=96/87% | S=0.705<br>N=2.109<br>F/D=100/100% | S=0.527<br>N=1.010<br>F/D=100/100% | S=0.593<br>N=0.982<br>F/D=100/100% |
| D3 | S=0.983<br>N=1.334<br>F/D=66/83% | S=1.144<br>N=1.140<br>F/D=100/91% | S=0.346<br>N=0.917<br>F/D=64/88% | S=0.537<br>N=0.671<br>F/D=78/56% |
| D4 | S=1.529<br>N=1.634<br>F/D=95/88% | S=0.523<br>N=2.182<br>F/D=85/100% | S=0.972<br>N=0.904<br>F/D=100/100% | S=0.786<br>N=0.640<br>F/D=65/100% |
| D5 | S=1.449<br>N=1.965<br>F/D=100/88% | S=0.965<br>N=2.620<br>F/D=100/100% | S=0.850<br>N=1.282<br>F/D=100/100% | S=0.935<br>N=0.782<br>F/D=100/100% |

## Qwen3-1.7B — Raw source vector (α=1.5)

Each cell: `S` normalized sufficiency / `N` normalized necessity; `F/D` strict flip/drop.

| source \ target | D1 | D3 | D4 | D5 |
|---|---:|---:|---:|---:|
| D1 | S=1.190<br>N=2.977<br>F/D=100/100% | S=0.673<br>N=3.186<br>F/D=100/100% | S=0.560<br>N=0.282<br>F/D=93/85% | S=0.602<br>N=0.181<br>F/D=81/0% |
| D3 | S=0.964<br>N=3.896<br>F/D=92/100% | S=1.132<br>N=2.768<br>F/D=100/100% | S=0.363<br>N=0.749<br>F/D=62/55% | S=0.545<br>N=0.458<br>F/D=73/0% |
| D4 | S=0.901<br>N=5.816<br>F/D=100/100% | S=-0.091<br>N=7.209<br>F/D=0/100% | S=0.702<br>N=2.750<br>F/D=100/100% | S=0.647<br>N=1.687<br>F/D=32/100% |
| D5 | S=1.200<br>N=6.392<br>F/D=100/100% | S=0.549<br>N=7.949<br>F/D=100/100% | S=0.688<br>N=3.738<br>F/D=100/100% | S=0.837<br>N=2.274<br>F/D=100/100% |

## Qwen3-14B — Target-norm aligned (α=1)

Each cell: `S` normalized sufficiency / `N` normalized necessity; `F/D` strict flip/drop.

| source \ target | D1 | D3 | D4 | D5 |
|---|---:|---:|---:|---:|
| D1 | S=0.620<br>N=0.256<br>F/D=86/5% | S=0.787<br>N=0.806<br>F/D=95/83% | S=0.677<br>N=0.023<br>F/D=42/0% | S=0.624<br>N=0.058<br>F/D=32/0% |
| D3 | S=0.355<br>N=0.212<br>F/D=63/2% | S=0.693<br>N=0.245<br>F/D=69/21% | S=0.479<br>N=0.058<br>F/D=13/0% | S=0.508<br>N=0.095<br>F/D=4/0% |
| D4 | S=0.446<br>N=0.157<br>F/D=65/2% | S=0.635<br>N=0.809<br>F/D=93/83% | S=0.569<br>N=0.021<br>F/D=29/2% | S=0.507<br>N=0.004<br>F/D=5/0% |
| D5 | S=0.665<br>N=0.267<br>F/D=91/6% | S=0.806<br>N=1.009<br>F/D=99/87% | S=0.691<br>N=0.059<br>F/D=45/10% | S=0.710<br>N=0.027<br>F/D=68/0% |

## Qwen3-14B — Raw source vector (α=1.5)

Each cell: `S` normalized sufficiency / `N` normalized necessity; `F/D` strict flip/drop.

| source \ target | D1 | D3 | D4 | D5 |
|---|---:|---:|---:|---:|
| D1 | S=0.838<br>N=0.398<br>F/D=100/12% | S=0.825<br>N=0.943<br>F/D=96/90% | S=0.798<br>N=0.110<br>F/D=75/11% | S=0.620<br>N=0.056<br>F/D=30/0% |
| D3 | S=0.597<br>N=0.445<br>F/D=81/7% | S=0.761<br>N=0.305<br>F/D=76/17% | S=0.716<br>N=0.162<br>F/D=45/10% | S=0.613<br>N=0.117<br>F/D=22/0% |
| D4 | S=0.660<br>N=0.319<br>F/D=93/6% | S=0.707<br>N=1.216<br>F/D=100/99% | S=0.747<br>N=0.169<br>F/D=64/25% | S=0.517<br>N=0.016<br>F/D=7/0% |
| D5 | S=1.047<br>N=0.502<br>F/D=100/15% | S=1.049<br>N=1.999<br>F/D=100/100% | S=0.882<br>N=0.569<br>F/D=100/89% | S=0.855<br>N=0.159<br>F/D=100/0% |

## Qwen3-4B — Target-norm aligned (α=1)

Each cell: `S` normalized sufficiency / `N` normalized necessity; `F/D` strict flip/drop.

| source \ target | D1 | D3 | D4 | D5 |
|---|---:|---:|---:|---:|
| D1 | S=0.954<br>N=0.621<br>F/D=80/97% | S=0.743<br>N=0.710<br>F/D=76/100% | S=0.786<br>N=0.807<br>F/D=58/100% | S=0.845<br>N=1.059<br>F/D=100/100% |
| D3 | S=0.834<br>N=0.590<br>F/D=65/88% | S=0.991<br>N=0.774<br>F/D=100/85% | S=0.770<br>N=0.784<br>F/D=38/100% | S=0.799<br>N=0.907<br>F/D=100/100% |
| D4 | S=0.998<br>N=0.754<br>F/D=99/100% | S=0.772<br>N=0.818<br>F/D=100/100% | S=0.974<br>N=0.761<br>F/D=100/100% | S=0.968<br>N=0.976<br>F/D=98/100% |
| D5 | S=0.969<br>N=0.546<br>F/D=76/99% | S=0.856<br>N=0.596<br>F/D=100/100% | S=0.870<br>N=0.574<br>F/D=64/100% | S=0.942<br>N=0.662<br>F/D=100/100% |

## Qwen3-4B — Raw source vector (α=1.5)

Each cell: `S` normalized sufficiency / `N` normalized necessity; `F/D` strict flip/drop.

| source \ target | D1 | D3 | D4 | D5 |
|---|---:|---:|---:|---:|
| D1 | S=1.066<br>N=1.292<br>F/D=100/100% | S=0.805<br>N=1.226<br>F/D=96/100% | S=0.799<br>N=0.934<br>F/D=77/100% | S=0.794<br>N=0.847<br>F/D=82/100% |
| D3 | S=0.956<br>N=1.471<br>F/D=100/100% | S=1.140<br>N=1.454<br>F/D=100/100% | S=0.820<br>N=1.002<br>F/D=72/100% | S=0.785<br>N=0.836<br>F/D=99/100% |
| D4 | S=1.062<br>N=1.752<br>F/D=100/100% | S=0.610<br>N=2.598<br>F/D=100/100% | S=1.027<br>N=1.342<br>F/D=100/100% | S=1.010<br>N=1.152<br>F/D=100/100% |
| D5 | S=1.170<br>N=1.735<br>F/D=100/100% | S=0.987<br>N=1.895<br>F/D=100/100% | S=1.012<br>N=1.236<br>F/D=100/100% | S=1.033<br>N=1.097<br>F/D=100/100% |

## Qwen3-8B — Target-norm aligned (α=1)

Each cell: `S` normalized sufficiency / `N` normalized necessity; `F/D` strict flip/drop.

| source \ target | D1 | D3 | D4 | D5 |
|---|---:|---:|---:|---:|
| D1 | S=0.965<br>N=0.902<br>F/D=99/92% | S=0.864<br>N=0.702<br>F/D=100/98% | S=0.753<br>N=0.631<br>F/D=100/100% | S=0.868<br>N=1.147<br>F/D=100/100% |
| D3 | S=0.880<br>N=1.092<br>F/D=99/84% | S=1.028<br>N=0.847<br>F/D=100/98% | S=0.640<br>N=0.888<br>F/D=82/100% | S=0.713<br>N=1.759<br>F/D=85/100% |
| D4 | S=0.993<br>N=1.031<br>F/D=100/100% | S=0.619<br>N=0.779<br>F/D=100/100% | S=0.905<br>N=0.853<br>F/D=100/100% | S=0.952<br>N=1.224<br>F/D=100/100% |
| D5 | S=1.000<br>N=0.842<br>F/D=100/85% | S=0.843<br>N=0.645<br>F/D=100/87% | S=0.844<br>N=0.588<br>F/D=100/99% | S=0.931<br>N=0.756<br>F/D=100/100% |

## Qwen3-8B — Raw source vector (α=1.5)

Each cell: `S` normalized sufficiency / `N` normalized necessity; `F/D` strict flip/drop.

| source \ target | D1 | D3 | D4 | D5 |
|---|---:|---:|---:|---:|
| D1 | S=1.198<br>N=1.349<br>F/D=100/100% | S=0.973<br>N=0.853<br>F/D=100/100% | S=0.785<br>N=0.746<br>F/D=100/100% | S=0.842<br>N=1.035<br>F/D=100/100% |
| D3 | S=1.131<br>N=2.437<br>F/D=100/100% | S=1.131<br>N=1.624<br>F/D=100/100% | S=0.712<br>N=1.664<br>F/D=100/100% | S=0.764<br>N=2.273<br>F/D=98/100% |
| D4 | S=1.154<br>N=2.305<br>F/D=100/100% | S=0.633<br>N=1.431<br>F/D=100/100% | S=1.022<br>N=1.350<br>F/D=100/100% | S=0.983<br>N=1.550<br>F/D=100/100% |
| D5 | S=1.175<br>N=2.183<br>F/D=100/100% | S=0.795<br>N=1.405<br>F/D=100/100% | S=1.050<br>N=1.293<br>F/D=100/100% | S=0.994<br>N=1.163<br>F/D=100/100% |

