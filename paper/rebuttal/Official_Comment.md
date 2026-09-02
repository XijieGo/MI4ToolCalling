# Official Comment: Cross-family feature-level evidence for the suppression mechanism

We thank Reviewer wPFH and Reviewer LxEy for raising this core concern promptly: the feature-level mechanism had been confirmed only within the Qwen3 family. We have now **closed that gap with two new analyses.** First, Transcoder sparse-feature analyses for Qwen3.5, Mistral, and Granite. Second, native MLP-unit causal interventions across all seven models. Both confirm that **the feature-level suppression mechanism generalizes beyond the Qwen family.**

### **1. Transcoder sparse-feature analysis on Qwen3.5, Mistral, and Granite**

Since our initial rebuttal, we trained targeted Transcoders for Granite, Mistral and Qwen3.5 (training details in §3). We fit the direction and select features on the training split, then evaluate frozen selections on the held-out split following exactly the same implementation as used throughout the paper.

|Model|*l*|*r*(*l*,*p*) (%)|Suff.|Necc.|MLP/Attn|K_corrupt / K_clean|Max Attn (pp)|
|:-|-:|-:|-:|-:|-:|-:|-:|
|Qwen3.5-4B|31|96.4|0.83|0.95|1.64|**14.23 / 10.12 = 1.406**|47.3|
|Qwen3.5-9B|29|97.9|0.95|0.97|1.90|**20.36 / 11.31 = 1.800**|66.1|
|Mistral-3.2-24B|26|98.2|0.87|0.95|1.49|**7.28 /  4.41 = 1.650**|8.8|
|Granite-3.3-8B|35|98.8|0.98|0.89|2.45|**37.22 / 16.53 = 2.251**|25.2|

> **The feature-level suppression mechanism generalizes beyond the Qwen family.**

---

### **2. Native MLP-unit interventions confirm suppression dominance in 6/7 models**

Besides, to make the feature-level mechanism more solid, we also decomposed MLP writes exactly as $\mathrm{MLP}(x)=\sum_n z_n(x) \cdot w_n$, which requires no auxiliary model and applies directly to every architecture. If suppression drives the decision, swapping suppressor activations from corrupt into clean prompts, i.e. S-units where $z^c_n \leftarrow z^{\*}_n$ , should disrupt tool calling, while swapping driver activations from clean into corrupt prompts, i.e. E-units where$z^{*}_n \leftarrow z^c_n$, should fail to restore it. We test this across all seven models, selecting equal-budget S and E units on training data by$|\kappa_n|$ ranking, and evaluating all shifts on 300 held-out pairs.

|Model|S→clean ($\Delta m$)|E→clean ($\Delta m$)|S→corrupt ($\Delta m$)|E→corrupt ($\Delta m$)|S : E|
|:-|-:|-:|-:|-:|-:|
|Qwen3-4B|−0.268|-0.103|+1.948|+0.514|3.60×|
|Qwen3-8B|−1.759|-0.239|+5.943|+0.886|6.84×|
|Qwen3-14B|−1.482|-1.073|+6.443|+2.138|2.47×|
|Qwen3.5-4B|+0.092|-0.253|+0.238|+0.184|0.76×|
|Qwen3.5-9B|−0.216|-0.008|+0.343|+0.146|3.62×|
|Mistral-3.2-24B|−1.472|-0.965|+3.276|+0.855|2.60×|
|Granite-3.3-8B|−1.475|-0.292|+1.995|+0.464|4.59×|

$\Delta m$: shift in action margin; $m=\ell_{\text{tool-call}}-\max_{t\neq\text{tool-call}}\ell_t$; S:E$=(|\Delta m_{S\rightarrow\text{clean}}|+|\Delta m_{S\rightarrow\text{corrupt}}|)\big/(|\Delta m_{E\rightarrow\text{clean}}|+|\Delta m_{E\rightarrow\text{corrupt}}|)$; values above 1 indicate suppression dominance.

> **Across six of the seven models, S:E consistently exceeds 1: suppressors actively control the decision, drivers do not.**

---

### **3. Training details and held-out reconstruction**

Each Transcoder was trained on 512-token blocks sampled from the EleutherAI/SmolLM2-135M-10B dataset, using a 16× dictionary expansion, bf16 precision with SDPA, a learning rate of$(2\times10^{-4})$, and scheduled sparsity regularization. For each layer, training used four DDP ranks, each with an activation batch size of 2,048, yielding a global batch size of 8192. Each completed per-layer run therefore comprised 122,071 optimization steps, corresponding to approximately 1 billion activation positions. To reduce distribution shift in the tool-calling setting, we mixed tool-calling prompts into the original training corpus at a rate of 0.15%.

|Backbone|Layer|FVU ↓|$L_0$ ↓|$\Delta$ LM Loss ↓|
|:-|-:|-:|-:|-:|
|Qwen3.5-4B|28|0.388|108.2|0.027378|
|Qwen3.5-4B|29|0.314|111.2|0.026016|
|Qwen3.5-4B|30|0.504|119.9|0.029742|
|Qwen3.5-4B|31|0.174|151.6|0.028957|
|Qwen3.5-9B|26|0.340|187.7|0.032801|
|Qwen3.5-9B|27|0.419|187.5|0.035431|
|Qwen3.5-9B|28|0.244|161.1|0.023886|
|Qwen3.5-9B|29|0.343|133.8|0.031238|
|Granite|32|0.382|179.5|0.022535|
|Granite|33|0.415|155.8|0.027714|
|Granite|34|0.401|154.1|0.032503|
|Granite|35|0.571|90.6|0.036389|
|Mistral|25|0.431|156.4|0.033518|
|Mistral|26|0.298|143.1|0.029423|
|Mistral|27|0.521|152.9|0.029158|
|Mistral|28|0.278|147.4|0.026722|

*$FVU$ is the reconstruction mean-squared error normalized by the variance of the target MLP outputs; lower values indicate higher reconstruction fidelity.$L_0$ is the mean number of active Transcoder features per token and measures representation sparsity.$\Delta$ LM Loss denotes the increase in held-out language-modeling loss after replacing the corresponding MLP with its Transcoder approximation.*

> **Overall, the completed Transcoders achieve stable held-out reconstruction under sparse activation budgets, providing sufficient fidelity for their use as an analysis tool in our experiments.**

# response to the Reviewer wPFH

We appreciate this comment, which pushed us to produce evidence that strengthens the paper.

Our Official Comment above presents the new evidence directly: Transcoder analyses for Mistral, Granite, and Qwen3.5, plus native MLP-unit interventions across all seven models, all confirm that **the feature-level suppression mechanism is NOT Qwen-specific.** The core finding, that analysis verbs suppress a scaffold-conditioned call prior through localized MLP features rather than execution verbs actively promoting it, now holds across three architectural families with both sparse-feature and dense-unit causal evidence.

### **Why This Work Meets the Main-Track Bar**

With that gap now closed, we briefly summarize why this work meets the main-track bar:

- **An important and underexplored question.** No prior mechanistic study had opened up agentic tool-call decisions. Understanding *why* LLMs invoke tools (not just *whether* they do) is foundational to AI safety and agent alignment, yet the field had no causal account of it.

- **An elegant and reusable experimental method.** Verb substitution provides a clean causal handle inside prompts spanning thousands of tokens, a methodological contribution the field can apply directly to multi-step planning, tool selection among alternatives, and argument construction.

- **Deep, multi-level mechanistic analysis.** We do not stop at behavioral correlates. We deliver module-level localization, feature-level causal decomposition via Transcoder, and targeted ablations that outperform random controls by 98x. This is the kind of mechanistic depth the main track expects.

- **Broad generalization across models and settings.** The mechanism transfers zero-shot to multi-turn τ²-bench trajectories across seven models, 600 verb-free requests, and now three architectural families (Qwen, Mistral, Granite) at the feature level. The evidence now spans three architectural families at every level of analysis.

We are grateful for the careful engagement throughout this discussion. The reviewer's scrutiny pushed us toward stronger evidence and a more rigorous paper.

# response to the Reviewer LxEy

Thank you for the concise and actionable feedback. You raised three points and we address each.

**1. The Qwen-only mechanistic account.** Closed in the Official Comment above, with Transcoder feature evidence for Mistral and Granite and native MLP-unit interventions on all seven models.

**2. Scoping the title.** You are right that the title should reflect what is demonstrated, and we should have engaged with this earlier. Two changes we propose:

- Replace "A Scaffold Default" with **"A Scaffold-Conditioned Call Prior"**, consistent with the terminology change we adopted after the scaffold-only baseline showed the scaffold alone never produces `<tool_call>` at top-1.
- ⟨Choose one:⟩
  - *Option A (question retained):* "How Do Agentic LLMs Decide to Call Tools? A Scaffold-Conditioned Call Prior Controlled by Suppression"
  - *Option B (scope in the title):* "Deciding to Call a Tool: A Scaffold-Conditioned Call Prior Controlled by Suppression in Single-Turn Agentic Prompts"

We are happy to adopt whichever you prefer, or your own wording.

**3. τ²-bench as rates with denominators at matched 1× gain.** Reported below in that format, consistent with the synthetic table. Each cell is out of 200 turns; removal on turns that natively call a tool, induction on turns that natively answer in text; random is an equal-norm seeded direction.

| Model | Removal, μΔ (1×) | Removal, random (1×) | Induction, μΔ (1×) | Induction, random (1×) |
| :- | -: | -: | -: | -: |
| Qwen3-4B | 101/200 = 50.5% | 16/200 = 8.0% | 154/200 = 77.0% | 2/200 = 1.0% |
| Qwen3-8B | 73/200 = 36.5% | 13/200 = 6.5% | 128/200 = 64.0% | 0/200 = 0.0% |
| Qwen3-14B | 187/200 = 93.5% | 1/200 = 0.5% | 133/200 = 66.5% | 0/200 = 0.0% |
| Qwen3.5-4B | 181/200 = 90.5% | 30/200 = 15.0% | 49/200 = 24.5% | 0/200 = 0.0% |
| Qwen3.5-9B | 171/200 = 85.5% | 0/200 = 0.0% | 107/200 = 53.5% | 0/200 = 0.0% |
| Mistral-3.2-24B | 73/200 = 36.5% | 2/200 = 1.0% | 87/200 = 43.5% | 11/200 = 5.5% |
| Granite-3.3-8B | 182/200 = 91.0% | 7/200 = 3.5% | 11/200 = 5.5% | 0/200 = 0.0% |

At matched 1× gain the effect is clear of its random control in every cell, and the variation across models is visible rather than smoothed by the 1.5× row: Granite is readily removable (91.0%) but resists induction (5.5%), Qwen3-8B is the reverse, and Mistral's random control is itself active on induction (5.5%), the one cell where an arbitrary direction moves the decision at a rate worth reporting. The revision reports the 1× rates as the primary table and the 1.5× results as a dose-response follow-up.

Thank you for all three suggestions.
