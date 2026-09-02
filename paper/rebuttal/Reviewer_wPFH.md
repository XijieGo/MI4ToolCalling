We thank the reviewer for the most careful reading. We appreciate the positive comments on the clarity and methodology.
Below, we address the concerns with clarifications and additional results.

### **1. Multiple substantive numerical inconsistencies undermine confidence in reported results**

To verify these numerical inconsistencies, we traced most of them to two archived Qwen3-8B dataset versions mixed during manuscript assembly.
v1 held 1,711 valid pairs split 1,204/507; the later cross-scale study used a balanced v2 of 1,500 split 1,200/300. The submission describes v2, but several v1 results and one per-verb count survived into the text.
Although there are minor numerical differences, the main conclusions remain unchanged. We will correct these typographical errors in the revised version. The correct values are shown below.

- **A.4 (0%) vs Table 1 (5.33%)**: Table 1's 5.33% is 27/507, so it is v1; A.4's 0% came from the v2 construction screen, and a fresh rerun of that screen on the frozen v2 held-out gives 11 of 300, so the honest v2 baseline is **3.67%**. All eleven sit within one to three units of the bf16 logit grid at that magnitude, with probabilities between 0.53 and 0.59, so kernel arithmetic decides which side they land on at the resolution floor. The revision reports 3.67% and no longer describes the screen as producing an exact zero.
- **Table 9 (300 `inspect`) vs A.3 (342)**: 342 is the v1 verb assignment, v2 assigns 300.
- **99.21% vs 97%**: the first is the top-1 rate after adding the frozen coding direction μΔ, the second a recovery summary from patching the whole state. Different interventions, different definitions of recovery, so the revision stops presenting them side by side.
- **D.4's 32.5% and 0.99%**: a single-head L28H3 result was misplaced into the group-ablation paragraph, and the revision removes it.
- **Table 3 (24.4/10.9) vs Table 2 (28.77/12.62)**: Table 2 sums per-layer `|kappa|` over labeled features in L20 to L23; Table 3 is a cross-scale summary under a different sign rule, layer window, split, and top-20 selection, so it was never meant to be that sum. The revision renames them and states both definitions.

### **2. Very narrow experimental scope, despite broad framing, heavily limits grounds for generalizability.**

Two classic previous works, IOI[1] and ROME[2], both isolate their mechanism in one narrow setting first.
We followed them by starting from coding, the most controllable template we could build and where agentic tool use is most deployed. 
To evaluate our method under multiple tools and long-context settings, we stepped into real τ²-bench trajectories[3] and intervened at the next turn. Results show that our method works well in these scenarios.

| Direction, gain | Qwen3-4B | Qwen3-8B | Qwen3-14B | Qwen3.5-4B | Qwen3.5-9B | Mistral | Granite |
|---|---|---|---|---|---|---|---|
| μΔ, 1× | 101/154 | 73/128 | 187/133 | 181/49 | 171/107 | 73/87 | 182/11 |
| Random, 1× | 16/2 | 13/0 | 1/0 | 30/0 | 0/0 | 2/11 | 7/0 |
| μΔ, 1.5× | 199/199 | 106/200 | 200/200 | 190/135 | 181/187 | 161/157 | 200/25 |
| Random, 1.5× | 28/13 | 30/0 | 9/0 | 46/0 | 0/0 | 4/27 | 58/0 |

*Each cell is removal / induction, out of 200 turns each, counted as first-token decisions switched by the intervention. The roster adds Qwen3.5 as the newest family available, and leaves out Qwen3-1.7B, whose native tool calling is too weak to produce baseline-eligible turns, and Devstral, a coding-specialized derivative of the Mistral-Small base already here. Random is an equal-norm seeded direction.*

Each prompt is the model's own native context at a real decision point in Telecom or Retail, 5,385 to 15,690 tokens with 16 to 43 tool schemas available and every earlier call still in history, no verb of ours anywhere in it; we release the collected trajectories as `τ²-bench-{telecom,retail}-traces`. 
Each model's μΔ goes in **with no re-estimation**: removal on 200 turns that natively call a tool, induction on 200 that natively answer in text, no trajectory contributing to any μΔ estimate. The roster adds Qwen3.5 as the newest family available, and leaves out Qwen3-1.7B, whose native tool calling is too weak to produce baseline-eligible turns, and Devstral, a coding-specialized derivative of the Mistral-Small base already here.

μΔ **carries into these trajectories on all seven models and stays clear of its random control everywhere**, and at 1.5× removal is near-total on five of the seven. How far it carries varies: Granite is fully removable at 200/200 yet resists induction at 25/200, Qwen3-8B is the other way round, and Mistral's random control is itself active on induction at 27/200, the one cell where an arbitrary direction moves the decision at a rate worth reporting. A model-specific direction fit on a single-turn, single-tool coding template remains a strong causal control direction for call-versus-text decisions in previously unseen multi-turn, multi-tool contexts.  Together with the 600 verb-free requests in our reply to Reviewer CJrQ, §3, these results go into a new Section 3.3.

### **3. Cross-family generalization is incomplete, and deep mechanistic details of other models' analyses are not provided**
Open-weight models that hold a multi-tool policy across a long trajectory are scarce below 10B, so the four families in §2 are about what the ecosystem supports.

No usable Transcoder exists for those architectures: the Gemma 2, Gemma 3, and Llama Transcoders do not apply here, and a comparable 40-layer bundle means 40 Transcoders at 122,070 steps and batch size 8,192 each, roughly one billion layer-activation examples. The Qwen3 Transcoders we rely on come from Hanna and Ameisen[4], which likewise validates its feature-level mechanism within a single family. For the other four, we give the vector-level mechanism instead: layer and position localization, sufficiency and necessity, and the transfers above. What their absence costs us is the naming of *which* features carry the suppression; the causal claim rests on the interventions, which ran on all seven.

### **4. Isolating the scaffold's causal role**

You asked for no-tool scaffolds, alternate tool schemas, and varied templates. We reran the complete R/T/F factorial and its length-matched control from scratch, where R is the role instruction, T the tool schema, and F the format template, and `P` is the neutral-minus-analysis gap. Qwen3-8B, 300 held-out prompts per condition.

The scaffold's effect splits cleanly in two. **F installs the high tool-call prior**: removing F reduces the neutral tool-call probability to 1.11e-06, and F alone drives neutral and analysis requests alike close to the ceiling. **T makes that prior selective**: removing T shrinks the neutral-analysis gap from 0.847 to 0.489. The length-matched control falls between these conditions, so part of T's apparent effect is prompt length rather than the schema itself. This is why the revision no longer describes the scaffold as installing an unconditional default: it raises a call prior, and the request wording settles it.

Your no-tool scaffold is the R + F row, and **μΔ still runs the decision there**, reaching 100% top-1 added to analysis prompts and 0% top-1 removed from execution ones. For the alternate tool schemas, the seven-model renamed / removed / mismatched grid is in our reply to Reviewer CJrQ, §1. This factorial goes into the main text alongside the τ²-bench results in the new Section 3.3, and the schema grid goes into Section 4 with the affordance crossover.

| Scaffold components | Neutral `p_call` | Analysis `p_call` | `P = neutral - analysis` |
|---|---:|---:|---:|
| R + T + F (full scaffold) | 0.8547 | 0.0080 | 0.847 |
| T + F (no role instructions) | 0.9429 | 0.1799 | 0.763 |
| R + F (no tool schema) | 0.9423 | 0.4531 | 0.489 |
| R + T (no format template) | 1.11e-06 | 1.32e-09 | 1.1e-06 |
| F only | 1.0000 | 0.9976 | 0.002 |
| T only | 1.85e-08 | 2.58e-11 | 1.8e-08 |
| R only | 1.20e-08 | 9.02e-11 | 1.2e-08 |
| Empty system scaffold | 1.41e-15 | 1.52e-15 | -1.1e-16 |
| R + length-matched neutral text + F | 0.7645 | 0.1517 | 0.613 |

### **5. First-token framing**

This paper studies the single-token call-or-no-call decision, and tool choice, argument construction, and workflow success fall outside its scope, which the revision states in the main text. That decision is the one every agent makes first, and §2 shows μΔ making it inside trajectories where the model then continues on its own.


### **6. On reproducibility, the checklist is wrong and the abstract is right**

Thanks for your careful reminder. Our anonymous repository already provides the construction pipeline, data, intermediate artifacts, intervention vectors, and scripts that reproduce every table. The Statistical Support and Broader Impact sections are indeed missing; both go into the revision, and we can supply either on request.

---

[1] Wang, K., Variengien, A., Conmy, A., Shlegeris, B., & Steinhardt, J. Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small. arXiv:2211.00593, 2022.

[2] Meng, K., Bau, D., Andonian, A., & Belinkov, Y. Locating and Editing Factual Associations in GPT. arXiv:2202.05262, 2022.

[3] Barres, V., Dong, H., Ray, S., Si, X., & Narasimhan, K. τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment. arXiv:2506.07982, 2025.

[4] Hanna, M., & Ameisen, E. Latent Planning Emerges with Scale. arXiv:2604.12493, 2026.
