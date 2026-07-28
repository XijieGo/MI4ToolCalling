# Fixed LxEy head roles across v4 domains

The three heads were selected on the code-domain v2_1500 result and held fixed here. Every row uses the frozen held-out v4 test pairs; no D3/D4/D5 head selection or behavioral filtering occurs in this analysis.

`Δ attention` and `Δ raw write` are clean minus corrupt. Rescue patches clean head-z into corrupt prompts; ablation patches corrupt head-z into clean prompts.

| domain | head(s) | Δ attention | Δ raw tool write | rescue Δ logit | rescue strict recovery | ablation Δ logit | ablation strict loss |
|---|---|---:|---:|---:|---:|---:|---:|
| D3 | L19H31 | +0.0520 | +0.0354 | +1.0562 | 0.0% | -0.6700 | 0.0% |
| D3 | L20H29 | -0.0039 | +0.0217 | +0.5325 | 0.0% | -0.2163 | 0.0% |
| D3 | L20H14 | -0.0129 | +0.0659 | +1.0800 | 0.0% | -0.3925 | 0.0% |
| D3 | L19H31+L20H29+L20H14 | +0.0352 | +0.1230 | +2.3150 | 0.0% | -1.3963 | 0.0% |
| D4 | L19H31 | +0.0567 | +0.0782 | +0.9975 | 0.0% | -0.5987 | 0.0% |
| D4 | L20H29 | -0.0113 | +0.0056 | +0.7613 | 0.0% | -0.2150 | 0.0% |
| D4 | L20H14 | -0.0354 | +0.0322 | +0.5863 | 0.0% | -0.1538 | 0.0% |
| D4 | L19H31+L20H29+L20H14 | +0.0100 | +0.1160 | +2.0963 | 0.0% | -0.9413 | 0.0% |
| D5 | L19H31 | -0.0262 | +0.0806 | +2.1762 | 0.0% | -0.5312 | 0.0% |
| D5 | L20H29 | +0.0076 | +0.0064 | +0.2894 | 0.0% | -0.0550 | 0.0% |
| D5 | L20H14 | -0.0247 | +0.0404 | +0.9362 | 0.0% | -0.1000 | 0.0% |
| D5 | L19H31+L20H29+L20H14 | -0.0432 | +0.1273 | +2.8075 | 0.0% | -0.5675 | 0.0% |

Interpret rescue and ablation together: a head has a cross-domain bridge role only where clean-z raises tool evidence on corrupt prompts and corrupt-z removes it from clean prompts. The three-head row tests their joint effect; it should not be read as a complete circuit proof.
