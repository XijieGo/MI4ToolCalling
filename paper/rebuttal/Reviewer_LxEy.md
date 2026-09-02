Thank you for the most technically specific review we received.
We appreciate the comments on the significance and methodology.
We address these concerns through further clarification and additional analyses.
Some key results are summarized first.

- **Beyond single-turn, verb-swapped coding prompts (§1).** The frozen μΔ controls first-token call-versus-text decision in full τ²-bench Telecom and Retail trajectories across all seven models, and it also transfers to verb-free requests.
- **Scaffold-alone baseline (§2).** An empty turn, greeting, or unrelated question never produces `<tool_call>` at top-1. The evidence supports a scaffold-conditioned, wording-sensitive call prior.
- **Verb-position signal transfer (§3).** L19H31, L20H29 and L20H14 carry the signal to the prediction position. Patching their outputs moves both the μΔ coordinate and tool-call logit, with consistent signs in search, database and API scaffolds.
- **Rank and probability distribution (§4).** Across 2,100 suppressed prompts, only 4.9% place the tool-call marker at rank 2 and 0.1% assign it probability ≥0.30. The flip is rarely a near-tie.
- **Post-hoc layer-24 selection (§5).** Refitting μΔ at L22–L25 and comparing it with equal-norm random directions identifies an L23–L25 causal window, with L24 its sharpest point.


### 1. Beyond single-turn, verb-swapped coding prompts. And the generalizability of the mechanistic explanation across model families and even across different scales within Qwen3.

Two classic mechanistic studies, **IOI** [1] and **ROME** [2], also established their mechanisms in a single, tightly controlled setting before testing broader applicability. We followed the same methodology by starting from coding, where tool use is both the most controllable to study and the most widely deployed in current agentic systems. We agree, however, that a mechanism proposed for tool-use decisions should be evaluated beyond this initial template.

To test generalization, we intervened on **real τ²-bench trajectories** [3] rather than synthetic coding prompts. These trajectories come from **Telecom** and **Retail** tasks, contain **5,385–15,690 tokens**, **16–43 available tool schemas**, multiple previous tool calls, and **no verb manipulation of ours**. We intervene only at the **next native decision point**, using the same μΔ estimated from the original coding template **without any re-estimation**: removal on **200 turns** that natively call a tool, induction on **200 turns** that natively answer in text, with no trajectory contributing to any μΔ estimate.

| Direction, gain | Qwen3-4B | Qwen3-8B | Qwen3-14B | Qwen3.5-4B | Qwen3.5-9B | Mistral | Granite |
|---|---:|---:|---:|---:|---:|---:|---:|
| μΔ, 1× | 101/154 | 73/128 | 187/133 | 181/49 | 171/107 | 73/87 | 182/11 |
| Random, 1× | 16/2 | 13/0 | 1/0 | 30/0 | 0/0 | 2/11 | 7/0 |
| μΔ, 1.5× | 199/199 | 106/200 | 200/200 | 190/135 | 181/187 | 161/157 | 200/25 |
| Random, 1.5× | 28/13 | 30/0 | 9/0 | 46/0 | 0/0 | 4/27 | 58/0 |

*Each cell is removal / induction, out of 200 turns each, counted as first-token decisions switched by the intervention. Random is an equal-norm seeded direction.*

Across all **seven models**, μΔ consistently transfers to these previously unseen **multi-turn, multi-tool** trajectories and substantially outperforms the random control. At **1.5×**, removal is nearly complete on **five of the seven** models. While transfer strength varies across architectures (e.g., Granite is easier to suppress than induce, whereas Qwen3-8B shows the opposite), the same direction learned from a **single-turn coding template** remains a strong causal control direction in realistic, long-context, multi-tool settings. These τ²-bench experiments, together with the **600 verb-free requests** reported in our response to Reviewer CJrQ (§3), will be added as a new **Section 3.3**.

### 2. Whether the scaffold alone (no user request) produces <tool_call> as top-1?

We ran the requested scaffold-only baseline on the held-out set. The scaffold alone does not produce <tool_call> as the top-1 prediction (L0–L2: 0%, ≤5×10⁻⁴). Instead, the scaffold establishes a wording-sensitive call prior that the request resolves: adding a neutral verb raises <tool_call> to 86.7%, an analysis verb suppresses it to 0%, while the bare task body lies in between (32.3%).
Accordingly, we replace the term “default” with “scaffold-conditioned, wording-sensitive call prior.” This better reflects the evidence and avoids implying that the scaffold alone deterministically triggers tool use.
This change is terminological rather than causal: all intervention effects are measured relative to their corresponding baseline condition, so neither the intervention results nor the conclusions change.
Detailed results are shown below.

| Request | `<tool_call>` top-1 | Mean prob. |
|---|---:|---:|
| L0–L2: empty, “Hello,” unrelated question | 0.0% | ≤5.0e-04 |
| L3: task body only | 32.3% | 0.3310 |
| L4: neutral verb + body | 86.7% | 0.8547 |
| L5: analysis verb + body | 0.0% | 0.0080 |
| L6: execution verb + body | 100.0% | 0.9993 |
| L7: no scaffold + execution + body | 0.0% | 6.56e-17 |

### 3. How does the verb-position signal transfer?

We searched every attention head in L5–L20 on 300 held-out pairs, measuring its prediction-position write along μΔ and then patching its clean output into the corrupt run. Ranked by the layer-24 μΔ coordinate of their clean output restores, the strongest heads are L19H31, L20H29, and L20H14. Patching each clean output raises the layer-24 μΔ coordinate and the tool-call logit together, recovering 13.0–31.3% of strict predictions, while direct attention to the changed verb barely differs. Joint patches retain the same sign in search, database and API scaffolds, changing the logit by +2.10 to +2.81 in the clean-to-corrupt direction and −0.57 to −1.40 in reverse. These heads provide a tested transfer route, while strict cross-domain recovery remains 0%; the main text presents this as a partial circuit.

| head | Δ attention to verb | Δ μ write | clean output → corrupt: Δ logit | recovery |
|-|-:|-:|-:|-:|
| L19H31 | −0.0072 | +0.7165 | +0.5375 | 31.33% |
| L20H29 | −0.0274 | +0.2348 | +0.3904 | 22.00% |
| L20H14 | −0.0153 | +0.1834 | +0.1529 | 13.00% |

### 4. Rank and probability distribution under suppression

Top-1 behavioral flips follow the convention in causal work on directions such as refusal (Arditi et al. 2024), while your near-tie concern calls for the full output distribution. The submission did not report final ranks or probability masses, so we measured final outputs on 300 held-out suppressed prompts per model; the cited 47.9–61.5% values are layer-wise logit-lens readings, not the output distribution. Across all 2,100 prompts, **102 (4.9%)** place the native marker at exactly rank 2 and only **2 (0.1%)** assign it probability ≥0.30. Granite and Qwen3.5-9B often retain the marker in the top 3, but their median probabilities are only 0.018 and 0.0486. Suppression removes nearly all probability mass even when format pressure keeps the marker nearby in rank, so these are decision flips rather than routine near-tie rerankings. The revision reports rank and probability alongside top-1 in Appendix F.

| Metric | Qwen3-4B | Qwen3-8B | Qwen3-14B | Qwen3.5-4B | Qwen3.5-9B | Mistral | Granite |
|-|-|-|-|-|-|-|-|
| Top-3 | 0.7% | 25.3% | 2.7% | 1.3% | 45.0% | 0.7% | 65.7% |
| Top-10 | 13.3% | 62.0% | 39.0% | 46.3% | 100.0% | 8.7% | 100.0% |
| Median prob. | 1.32e-09 | 4.22e-06 | 3.07e-05 | 0.00183 | 0.0486 | 0.000836 | 0.018 |

### 5. Is layer-24 selection post-hoc and circular?

Layer sweeps followed by held-out causal validation are standard in causal tracing (Meng et al. 2022) and direction discovery (Arditi et al. 2024). Our sweep and validation use separate data and metrics: μΔ is fit on training pairs only, and every intervention we report is scored on held-out pairs that contributed nothing to fitting it. The submission also includes a random feature-ablation control  (+0.28 versus −27.45 on the μΔ projection). Your question motivated a stronger directional test: we refit μΔ at each `resid_pre` from L22–L25 and compared it with an equal-norm random direction on 300 disjoint pairs. μΔ reaches full sufficiency from L23 and full necessity from L24. The L24 random addition is visibly nonzero at 56/300, yet μΔ moves 300/300 there and dominates its control at every layer. The result localizes commitment to an **L23–L25 window**, with L24 its sharpest point; the revision adopts this wording.

| strict `<tool_call>` top-1 rate | L22 | L23 | L24 | L25 |
|-|-:|-:|-:|-:|
| corrupt after +μΔ | 82.0% | 100.0% | 100.0% | 100.0% |
| clean after −μΔ | 45.7% | 12.0% | 0.0% | 0.0% |
| corrupt after +random | 0.0% | 0.3% | 18.7% | 0.0% |
| clean after −random | 100.0% | 99.7% | 97.7% | 100.0% |


### 6. Remaining questions

**Why commitment lags early separability.** A probe tests whether verb identity is readable; attribution tests whether a component causally drives the output (Hewitt & Liang 2019; Belinkov 2022). Our results match the enrichment-and-extraction pattern of Geva et al. 2023: identity is readable early, the L19–L20 heads route it to the final position, and the L23–L25 window makes it causally sufficient.

**Failure cases and boundary verbs.** On the frozen 300-pair manifest, `+μΔ` recovers **297/300** prompts; all three failures are Python prompts with the analysis verb `discuss`, so the residue clusters by verb. In a 20-verb sweep, `inspect` lies at the boundary with 54.3% top-1 and μΔ coordinate 0.076, while all ten execution verbs reach 100% with coordinates 0.794–1.110 (**ρ=0.923, p=7.09e-09**). The mechanism degrades gradually across ambiguous verbs, and the L24 coordinate tracks that gradient.


[1] Wang, Kevin Ro, et al. "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small." ICLR 2022.

[2] Meng, Kevin, et al. "Locating and editing factual associations in GPT." NeurIPS 2022.

[3] Barres, V, et al. τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment. arXiv:2506.07982, 2025.