# v5 adjacent-layer mean-difference sweep

Each raw clean-minus-corrupt vector is fit on 200 Qwen3-8B v5 training pairs at the decoder-block input (`resid_pre`) of that layer. Every intervention uses the same frozen vector on 300 disjoint held-out pairs.

|Layer|$\|\mu_\Delta\|$|Corrupt +$\mu_\Delta$: top-1|strict flips|Corrupt +random: top-1|Clean -$\mu_\Delta$: top-1|strict drops|Clean -random: top-1|
|-:|-:|-:|-:|-:|-:|-:|-:|
|L22|36.336|82.0%|246/300|0.0%|45.7%|163/300|100.0%|
|L23|48.453|100.0%|300/300|0.3%|12.0%|264/300|99.7%|
|L24|66.480|100.0%|300/300|18.7%|0.0%|300/300|97.7%|
|L25|96.720|100.0%|300/300|0.0%|0.0%|300/300|100.0%|

`top-1` is strict: the tool-call logit must be uniquely rank 1. Random directions are independently seeded unit vectors rescaled to the corresponding $\|\mu_\Delta\|$.
