# OpenReview: Submission13798

## Meta Review of Submission13798 by Area Chair 8i1A

*Meta Review by Area Chair 8i1A · 23 Jul 2026, 06:07 (modified: 24 Jul 2026, 01:51)*

### Metareview
This paper studies how LLM agents decide to call tools using mechanistic interpretability analysis. The reviewers find that the problem is interesting and the results is promising.

However, there are several major concerns shared across multiple reviewers that need to be well-addressed in the rebuttal:

overclaim on the title and narrow problem: the problem that this paper is studying is a very specific setting on the verb cued, while the title and the motivation is on a very broad tool use
very limited experiment: the evaluation is only on the coding tools, not clear if the results/finding on other tools
unclear how the method work for other type of tool calling, e.g. when agents call tools without any action verbs
experiments on other model family
some other major concerns that should be addressed in the rebuttal:

ablation on scaffold result
concerns on major inconsistency of the numerical results
Based on initial review, the decision is towards Rejection.


---

## Response to the Meta-Review: Clarifications and Additional Results

*Official Comment by Authors (Lijie Hu, Tingxu Han, Jiahao Zhang, Wei Song, +4 more) · 29 Jul 2026, 03:13 (modified: 30 Jul 2026, 00:00)*

### Comment
We thank the Area Chair for the summary of shared concerns, and answer the five points in order.

### 1. Overclaim on the title and narrow problem; evaluation only on coding tools
We took each model's μΔ, fit on the single-turn coding template, and applied it unchanged in real τ²-bench trajectories: 5,385–15,690 tokens, 16–43 tool schemas, multi-turn. Removal runs on 200 turns that natively call a tool; induction on 200 that natively answer in text.

| Direction, gain | Qwen3-4B | Qwen3-8B | Qwen3-14B | Qwen3.5-4B | Qwen3.5-9B | Mistral | Granite |
| --- | --- | --- | --- | --- | --- | --- | --- |
| μΔ, 1× | 101/154 | 73/128 | 187/133 | 181/49 | 171/107 | 73/87 | 182/11 |
| Random, 1× | 16/2 | 13/0 | 1/0 | 30/0 | 0/0 | 2/11 | 7/0 |
| μΔ, 1.5× | 199/199 | 106/200 | 200/200 | 190/135 | 181/187 | 161/157 | 200/25 |
| Random, 1.5× | 28/13 | 30/0 | 9/0 | 46/0 | 0/0 | 4/27 | 58/0 |
removal / induction, out of 200 each. Random is equal-norm. Qwen3-1.7B (weak native calling) and Devstral (Mistral variant) swapped for Qwen3.5.

μΔ transfers to unseen multi-turn, multi-tool trajectories, clear of its random control everywhere, so the title now matches the evidence.

### 2. Tool calling without any action verbs
We built 600 verb-free requests across code (APPS), search (FEVER), database (Spider), and API (ART-E), five non-imperative carriers per domain, screened by a verb blocklist. Keeping the ones each model already answers with a tool call, we subtracted μΔ at the coding-localized layer and position.

| Model | N | −μΔ | −random |
| --- | --- | --- | --- |
| Qwen3-4B | 119 | 100.0% | 23.5% |
| Qwen3-8B | 161 | 99.4% | 12.4% |
| Qwen3-14B | 161 | 66.5% | 0.6% |
| Qwen3.5-4B | 189 | 100.0% | 39.7% |
| Qwen3.5-9B | 184 | 100.0% | 0.5% |
| Mistral-3.2-24B | 147 | 90.5% | 63.9% |
| Granite-3.3-8B | 200 | 85.5% | 5.0% |
Strict drops of the native tool-call marker, α=1.5. N: screened verb-free requests with baseline tool calls.

μΔ governs verb-free calls too. The mechanism is not tied to an explicit action verb.

### 3. Ablation on scaffold result
We reran the complete R/T/F factorial from scratch, R the role instructions, T the tool schema, F the format template. Qwen3-8B, 300 held-out prompts per condition.

| Scaffold | Neutral p_call | Analysis p_call | Gap |
| --- | --- | --- | --- |
| R + T + F (full) | 0.8547 | 0.0080 | 0.847 |
| T + F (no role) | 0.9429 | 0.1799 | 0.763 |
| R + F (no tool schema) | 0.9423 | 0.4531 | 0.489 |
| R + T (no format) | 1.11e-06 | 1.32e-09 | ~0 |
| F only | 1.0000 | 0.9976 | 0.002 |
| R + length-matched text + F | 0.7645 | 0.1517 | 0.613 |
Full scaffold with no user request (empty, "Hello," or unrelated): <tool_call> is never top-1, p ≤ 5e-04.

The format template installs the tool-call default; the tool schema makes it selective, and the request wording resolves it. The revision attributes the default to the format template rather than the scaffold as a whole.

### 4. Concerns on major inconsistency of the numerical results
Two archived Qwen3-8B dataset versions were mixed during assembly: v1 (1,711 pairs, 1,204/507) and the balanced v2 (1,500, 1,200/300) that the paper describes, and several v1 numbers survived into the text. Across three repeated runs, a small number of boundary prompts (under 5%) shift between calling and not calling: their logits sit at the bf16 resolution floor, where kernel arithmetic decides the side. The rest are stable. The differences are minor and the conclusions unchanged; we correct them in the revision.

| Item | In submission | Corrected |
| --- | --- | --- |
| Corrupt baseline (A.4 vs Table 1) | 0% vs 5.33% | 3.67% (v2, 11/300); 5.33% was v1, 27/507 |
| inspect count (Table 9 vs A.3) | 300 vs 342 | 300; 342 was the v1 assignment |
| 99.21% vs 97% | presented side by side | different interventions |
| D.4 32.5% vs 0.99% | contradictory | single-head L28H3 misplaced into group ablation; removed |
| K ratio (Table 3 vs Table 2) | 24.4/10.9 vs 28.77/12.62 | different sign rule, window, selection; both renamed and defined |
| Code availability | abstract vs checklist | abstract is right; checklist corrected |
All causal results were computed on the frozen v2 held-out split and are unchanged. We re-checked every number in the paper against its source artifact, not only the flagged six.

### 5. Experiments on other model family
Every intervention above ran on seven models across four families. No usable transcoder exists outside Qwen3, and training a comparable 40-layer bundle costs roughly one billion layer-activation examples. Open-weight agentic models under 20B are also scarce, so the four families we test are close to what the ecosystem supports, and we added Qwen3.5 during the discussion period as the newest one available.

Data, code, and intermediate artifacts for every table above are in the rebuttal/ folder of our anonymous repository, https://anonymous.4open.science/r/MI4ToolCalling.


---

## Official Review of Submission13798 by Reviewer wPFH

*Official Review by Reviewer wPFH · 26 Jun 2026, 14:30 (modified: 29 Jul 2026, 02:31)*

### Summary
This paper studies how large language models(LLMs) with agentic capabilities internally decide whether to call external tools or answer directly in text. The paper investigates a single-turn code-completion task with a write_file tool. The authors create controlled prompt pairs differing by only one word, testing execution verbs like "write" versus analysis verbs like "discuss", while holding the tool scaffold and task fixed. Using mechanistic interpretability techniques, including activation patching and vector interventions, they identify a localized residual-stream direction at layer 24 that causally mediates this call/no-call decision on the first generated token, which precedes tool use. The authors argue that tool calling acts as a scaffold-induced default behavior, with analysis verbs suppressing rather than execution verbs actively promoting it. Cross-model experiments across Qwen3 variants and other LLMs show similar localization patterns, providing causal evidence for a compact internal representation of agentic decisions in long, scaffolded prompts.

Contribution Type: General: Most submissions will fall into this type.
### Strengths And Weaknesses
#### Strengths

The paper investigates an important underexplored question of whether agentic LLMs internally decide whether to call a tool or provide a direct answer, employing an elegant verb-substitution design that provides a clean causal handle within long, confounded prompts. It provides a detailed description of the experiment and an extensive analysis of the results.

Mechanistic analysis goes beyond correlation: activation patching localizes the decision, and vector addition/removal interventions provide causal evidence that a single residual-stream direction 
 at one layer/position is both sufficient and necessary for first-token tool calling.

Transcoder decomposition evidence suggests interpretable feature-level evidence showing analysis verbs suppress scaffold-biased features rather than execution verbs actively promoting tool use, with multi-stage tracing identifying compact representation rather than distributed processing.

Cross-model checks across Qwen3 variants and additional families strengthen robustness claims, while transparent limitations acknowledgment and detailed appendices document dataset construction, failed circuit searches, and downstream analyses comprehensively.

#### Weaknesses

Multiple substantive numerical inconsistencies undermine confidence in reported results. Appendix A.4 claims Qwen3-1.7B/4B/8B produces 0% tool calls on all 1,500 filtered corrupt prompts, yet Table 1 reports a 5.33% corrupt baseline for the same mode. Table 9 reports exactly 300 "inspect" corrupt prompts in the final dataset, while Appendix A.3 states 342 retained "inspect" pairs pass the filter. Appendix D.4 contains contradictory ablation claims: removing bridge heads and late readout makes the clean top-1 fall to 32.5%, but later states that the same intervention drops only 0.99%. Additional discrepancies include Table 3 reporting K_corrupt/K_clean as 24.4/10.9, while Table 2 values sum to 28.77/12.62, and mean vector addition achieving 99.21% recovery, whereas full state patching at L24 yields only 97%.

Very narrow experimental scope, despite broad framing, heavily limits grounds for generalizability. A single-turn code-completion task with a single write_file tool and first-token prediction does not establish whether the mechanism generalizes to multi-tool agents, search/API use, argument generation, or naturally occurring prompts. The paper successfully reuses existing methods but does not introduce a new framework to potentially compensate for the narrow scope.

Cross-family generalization is incomplete, and deep mechanistic details of other models' analyses are not provided. While vector localization effects appear across model families, the detailed suppressor-feature story depends heavily on Qwen-specific transcoders unavailable for other architectures.

The "scaffold default" interpretation lacks direct scaffold ablations such as no-tool scaffolds, alternate templates, and different tool instructions, which would be needed to confirm that the scaffold itself installs a default rather than analysis verbs merely suppressing specific features.

First-token framing is useful but incomplete: it addresses whether tool calls begin (<tool_call> vs text) but not whether correct tools are selected, arguments written correctly, or agentic workflows succeed.

Reproducibility documentation contains inconsistencies: the abstract states code is available, while the checklist marks it unavailable. The Statistical Support and Broader Impact sections referenced in the checklist are not visible in the manuscript text.

Quality: 3: good
Clarity: 3: good
Significance: 3: good
Originality: 3: good
### Questions
Please clarify the contradictory values outlined in the weaknesses and the reproducibility contradiction.

Please provide experiments directly isolating the scaffold's causal role, such as no-tool scaffolds, alternate tool schemas, or varied templates, measuring whether suppression effects persist and have mechanistic confirmation?

Please elaborate on the mechanistic details for other models' experiments.

### Limitations
yes

Rating: 3: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.
Confidence: 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
Ethical Concerns: NO or VERY MINOR ethics concerns only
Paper Formatting Concerns:
no

Code Of Conduct Acknowledgement: Yes
Responsible Reviewing Acknowledgement: Yes

---

## Official Comment by Reviewer wPFH

*Official Comment by Reviewer wPFH · 02 Aug 2026, 02:55*

### Comment
Dear authors,

Thank you for addressing most of the outlined problems and improving the paper.

Even though the rebuttal fixed some issues and added behavioral validation, the core limitation remains: the feature-level mechanism is Qwen3-only. The cross-family generalization claim announced in the title remains unconfirmed at deeper levels outside the Qwen family. The paper provided as an example that single-model evaluation is enough (Hanna and Ameisen) can only be treated as a venue precedent, not evidence that the design is universally adequate. The fundamental gap between claim scope (agentic LLMs) and evidence depth (one architecture's mechanism) isn't fixable by reframing alone.

This is obviously a strong workshop paper, but for the main track submission, my recommendation Borderline Reject stands.


---

## A friendly reminder

*Official Comment by Authors (Lijie Hu, Tingxu Han, Jiahao Zhang, Wei Song, +4 more) · 01 Aug 2026, 23:16*

### Comment
Thank you again for your careful and constructive review!

As the discussion period is nearing its end, we wanted to kindly ask whether our clarifications and additional results have adequately addressed your main concerns. If any questions or reservations remain, we would greatly appreciate the opportunity to provide further clarification.

If you feel that the concerns have been sufficiently addressed, we would be very grateful if you could take these updates into account when reconsidering your evaluation.


---

## Correction: Response to Your Review

*Official Comment by Authors (Lijie Hu, Tingxu Han, Jiahao Zhang, Wei Song, +4 more) · 28 Jul 2026, 23:57*

### Comment
We apologize for the confusion. Earlier today, we mistakenly posted a response intended for another reviewer under your review. Please disregard that earlier comment. Below, we provide the response addressing your review.

We thank the reviewer for the most careful reading. We appreciate the positive comments on the clarity and methodology. Below, we address the concerns with clarifications and additional results.

### 1. Multiple substantive numerical inconsistencies undermine confidence in reported results
To verify these numerical inconsistencies, we traced most of them to two archived Qwen3-8B dataset versions mixed during manuscript assembly. v1 held 1,711 valid pairs split 1,204/507; the later cross-scale study used a balanced v2 of 1,500 split 1,200/300. The submission describes v2, but several v1 results and one per-verb count survived into the text. Although there are minor numerical differences, the main conclusions remain unchanged. We will correct these typographical errors in the revised version. The correct values are shown below.

A.4 (0%) vs Table 1 (5.33%): Table 1's 5.33% is 27/507, so it is v1; A.4's 0% came from the v2 construction screen, and a fresh rerun of that screen on the frozen v2 held-out gives 11 of 300, so the honest v2 baseline is 3.67%. All eleven sit within one to three units of the bf16 logit grid at that magnitude, with probabilities between 0.53 and 0.59, so kernel arithmetic decides which side they land on at the resolution floor. The revision reports 3.67% and no longer describes the screen as producing an exact zero.
Table 9 (300 inspect) vs A.3 (342): 342 is the v1 verb assignment, v2 assigns 300.
99.21% vs 97%: the first is the top-1 rate after adding the frozen coding direction μΔ, the second a recovery summary from patching the whole state. Different interventions, different definitions of recovery, so the revision stops presenting them side by side.
D.4's 32.5% and 0.99%: a single-head L28H3 result was misplaced into the group-ablation paragraph, and the revision removes it.
Table 3 (24.4/10.9) vs Table 2 (28.77/12.62): Table 2 sums per-layer |kappa| over labeled features in L20 to L23; Table 3 is a cross-scale summary under a different sign rule, layer window, split, and top-20 selection, so it was never meant to be that sum. The revision renames them and states both definitions.
### 2. Very narrow experimental scope, despite broad framing, heavily limits grounds for generalizability.
Two classic previous works, IOI[1] and ROME[2], both isolate their mechanism in one narrow setting first. We followed them by starting from coding, the most controllable template we could build and where agentic tool use is most deployed. To evaluate our method under multiple tools and long-context settings, we stepped into real τ²-bench trajectories[3] and intervened at the next turn. Results show that our method works well in these scenarios.

| Direction, gain | Qwen3-4B | Qwen3-8B | Qwen3-14B | Qwen3.5-4B | Qwen3.5-9B | Mistral | Granite |
| --- | --- | --- | --- | --- | --- | --- | --- |
| μΔ, 1× | 101/154 | 73/128 | 187/133 | 181/49 | 171/107 | 73/87 | 182/11 |
| Random, 1× | 16/2 | 13/0 | 1/0 | 30/0 | 0/0 | 2/11 | 7/0 |
| μΔ, 1.5× | 199/199 | 106/200 | 200/200 | 190/135 | 181/187 | 161/157 | 200/25 |
| Random, 1.5× | 28/13 | 30/0 | 9/0 | 46/0 | 0/0 | 4/27 | 58/0 |
Each cell is removal / induction, out of 200 turns each, counted as first-token decisions switched by the intervention. The roster adds Qwen3.5 as the newest family available, and leaves out Qwen3-1.7B, whose native tool calling is too weak to produce baseline-eligible turns, and Devstral, a coding-specialized derivative of the Mistral-Small base already here. Random is an equal-norm seeded direction.

Each prompt is the model's own native context at a real decision point in Telecom or Retail, 5,385 to 15,690 tokens with 16 to 43 tool schemas available and every earlier call still in history, no verb of ours anywhere in it; we release the collected trajectories as τ²-bench-{telecom,retail}-traces. Each model's μΔ goes in with no re-estimation: removal on 200 turns that natively call a tool, induction on 200 that natively answer in text, no trajectory contributing to any μΔ estimate. The roster adds Qwen3.5 as the newest family available, and leaves out Qwen3-1.7B, whose native tool calling is too weak to produce baseline-eligible turns, and Devstral, a coding-specialized derivative of the Mistral-Small base already here.

μΔ carries into these trajectories on all seven models and stays clear of its random control everywhere, and at 1.5× removal is near-total on five of the seven. How far it carries varies: Granite is fully removable at 200/200 yet resists induction at 25/200, Qwen3-8B is the other way round, and Mistral's random control is itself active on induction at 27/200, the one cell where an arbitrary direction moves the decision at a rate worth reporting. A model-specific direction fit on a single-turn, single-tool coding template remains a strong causal control direction for call-versus-text decisions in previously unseen multi-turn, multi-tool contexts. Together with the 600 verb-free requests in our reply to Reviewer CJrQ, §3, these results go into a new Section 3.3.

### 3. Cross-family generalization is incomplete, and deep mechanistic details of other models' analyses are not provided
Open-weight models that hold a multi-tool policy across a long trajectory are scarce below 10B, so the four families in §2 are about what the ecosystem supports.

No usable Transcoder exists for those architectures: the Gemma 2, Gemma 3, and Llama Transcoders do not apply here, and a comparable 40-layer bundle means 40 Transcoders at 122,070 steps and batch size 8,192 each, roughly one billion layer-activation examples. The Qwen3 Transcoders we rely on come from Hanna and Ameisen[4], which likewise validates its feature-level mechanism within a single family. For the other four, we give the vector-level mechanism instead: layer and position localization, sufficiency and necessity, and the transfers above. What their absence costs us is the naming of which features carry the suppression; the causal claim rests on the interventions, which ran on all seven.

### 4. Isolating the scaffold's causal role
You asked for no-tool scaffolds, alternate tool schemas, and varied templates. We reran the complete R/T/F factorial and its length-matched control from scratch, where R is the role instruction, T the tool schema, and F the format template, and P is the neutral-minus-analysis gap. Qwen3-8B, 300 held-out prompts per condition.

The scaffold's effect splits cleanly in two. F installs the high tool-call prior: removing F reduces the neutral tool-call probability to 1.11e-06, and F alone drives neutral and analysis requests alike close to the ceiling. T makes that prior selective: removing T shrinks the neutral-analysis gap from 0.847 to 0.489. The length-matched control falls between these conditions, so part of T's apparent effect is prompt length rather than the schema itself. This is why the revision no longer describes the scaffold as installing an unconditional default: it raises a call prior, and the request wording settles it.

Your no-tool scaffold is the R + F row, and μΔ still runs the decision there, reaching 100% top-1 added to analysis prompts and 0% top-1 removed from execution ones. For the alternate tool schemas, the seven-model renamed / removed / mismatched grid is in our reply to Reviewer CJrQ, §1. This factorial goes into the main text alongside the τ²-bench results in the new Section 3.3, and the schema grid goes into Section 4 with the affordance crossover.

| Scaffold components | Neutral p_call | Analysis p_call | P = neutral - analysis |
| --- | --- | --- | --- |
| R + T + F (full scaffold) | 0.8547 | 0.0080 | 0.847 |
| T + F (no role instructions) | 0.9429 | 0.1799 | 0.763 |
| R + F (no tool schema) | 0.9423 | 0.4531 | 0.489 |
| R + T (no format template) | 1.11e-06 | 1.32e-09 | 1.1e-06 |
| F only | 1.0000 | 0.9976 | 0.002 |
| T only | 1.85e-08 | 2.58e-11 | 1.8e-08 |
| R only | 1.20e-08 | 9.02e-11 | 1.2e-08 |
| Empty system scaffold | 1.41e-15 | 1.52e-15 | -1.1e-16 |
| R + length-matched neutral text + F | 0.7645 | 0.1517 | 0.613 |
### 5. First-token framing
This paper studies the single-token call-or-no-call decision, and tool choice, argument construction, and workflow success fall outside its scope, which the revision states in the main text. That decision is the one every agent makes first, and §2 shows μΔ making it inside trajectories where the model then continues on its own.

### 6. On reproducibility, the checklist is wrong and the abstract is right
Thanks for your careful reminder. Our anonymous repository already provides the construction pipeline, data, intermediate artifacts, intervention vectors, and scripts that reproduce every table. The Statistical Support and Broader Impact sections are indeed missing; both go into the revision, and we can supply either on request.

[1] Wang, K., Variengien, A., Conmy, A., Shlegeris, B., & Steinhardt, J. Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small. arXiv:2211.00593, 2022.

[2] Meng, K., Bau, D., Andonian, A., & Belinkov, Y. Locating and Editing Factual Associations in GPT. arXiv:2202.05262, 2022.

[3] Barres, V., Dong, H., Ray, S., Si, X., & Narasimhan, K. τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment. arXiv:2506.07982, 2025.

[4] Hanna, M., & Ameisen, E. Latent Planning Emerges with Scale. arXiv:2604.12493, 2026.


---

## Rebuttal by Authors

*Rebuttal by Authors (Lijie Hu, Tingxu Han, Jiahao Zhang, Wei Song, +4 more) · 28 Jul 2026, 19:56 (modified: 28 Jul 2026, 21:24)*

### Rebuttal
Thank you for the most technically specific review we received. We appreciate the comments on the significance and methodology. We address these concerns through further clarification and additional analyses. Some key results are summarized first.

Beyond single-turn, verb-swapped coding prompts (§1). The frozen μΔ controls first-token call-versus-text decision in full τ²-bench Telecom and Retail trajectories across all seven models, and it also transfers to verb-free requests.
Scaffold-alone baseline (§2). An empty turn, greeting, or unrelated question never produces <tool_call> at top-1. The evidence supports a scaffold-conditioned, wording-sensitive call prior.
Rank and probability distribution (§3). Across 2,100 suppressed prompts, only 4.9% place the tool-call marker at rank 2 and 0.1% assign it probability ≥0.30. The flip is rarely a near-tie.
Post-hoc layer-24 selection (§4). Refitting μΔ at L22–L25 and comparing it with equal-norm random directions identifies an L23–L25 causal window, with L24 its sharpest point.
Verb-position signal transfer (§5). L19H31, L20H29 and L20H14 carry the signal to the prediction position. Patching their outputs moves both the μΔ coordinate and tool-call logit, with consistent signs in search, database and API scaffolds.
### 1,2,5. Beyond single-turn, verb-swapped coding prompts. And the generalizability of the mechanistic explanation across model families and even across different scales within Qwen3.
Two classic mechanistic studies, IOI [1] and ROME [2], also established their mechanisms in a single, tightly controlled setting before testing broader applicability. We followed the same methodology by starting from coding, where tool use is both the most controllable to study and the most widely deployed in current agentic systems. We agree, however, that a mechanism proposed for tool-use decisions should be evaluated beyond this initial template.

To test generalization, we intervened on real τ²-bench trajectories [3] rather than synthetic coding prompts. These trajectories come from Telecom and Retail tasks, contain 5,385–15,690 tokens, 16–43 available tool schemas, multiple previous tool calls, and no verb manipulation of ours. We intervene only at the next native decision point, using the same μΔ estimated from the original coding template without any re-estimation.

| Direction, gain | Qwen3-4B | Qwen3-8B | Qwen3-14B | Qwen3.5-4B | Qwen3.5-9B | Mistral | Granite |
| --- | --- | --- | --- | --- | --- | --- | --- |
| μΔ, 1× | 101/154 | 73/128 | 187/133 | 181/49 | 171/107 | 73/87 | 182/11 |
| Random, 1× | 16/2 | 13/0 | 1/0 | 30/0 | 0/0 | 2/11 | 7/0 |
| μΔ, 1.5× | 199/199 | 106/200 | 200/200 | 190/135 | 181/187 | 161/157 | 200/25 |
| Random, 1.5× | 28/13 | 30/0 | 9/0 | 46/0 | 0/0 | 4/27 | 58/0 |
Each cell reports removal / induction, counted as first-token decision flips caused by the intervention. Random denotes an equal-norm seeded direction.

Across all seven models, μΔ consistently transfers to these previously unseen multi-turn, multi-tool trajectories and substantially outperforms the random control. At 1.5×, removal is nearly complete on five of the seven models. While transfer strength varies across architectures (e.g., Granite is easier to suppress than induce, whereas Qwen3-8B shows the opposite), the same direction learned from a single-turn coding template remains a strong causal control direction in realistic, long-context, multi-tool settings. These τ²-bench experiments, together with the 600 verb-free requests reported in our response to Reviewer CJrQ (§3), will be added as a new Section 3.3.

### 3. Whether the scaffold alone (no user request) produces <tool_call> as top-1?
We ran the requested scaffold-only baseline on the held-out set. The scaffold alone does not produce <tool_call> as the top-1 prediction (L0–L2: 0%, ≤5×10⁻⁴). Instead, the scaffold establishes a wording-sensitive call prior that the request resolves: adding a neutral verb raises <tool_call> to 86.7%, an analysis verb suppresses it to 0%, while the bare task body lies in between (32.3%). Accordingly, we replace the term “default” with “scaffold-conditioned, wording-sensitive call prior.” This better reflects the evidence and avoids implying that the scaffold alone deterministically triggers tool use. This change is terminological rather than causal: all intervention effects are measured relative to their corresponding baseline condition, so neither the intervention results nor the conclusions change. Detailed results are shown below.

| Request | <tool_call> top-1 | Mean prob. |
| --- | --- | --- |
| L0–L2: empty, “Hello,” unrelated question | 0.0% | ≤5.0e-04 |
| L3: task body only | 32.3% | 0.3310 |
| L4: neutral verb + body | 86.7% | 0.8547 |
| L5: analysis verb + body | 0.0% | 0.0080 |
| L6: execution verb + body | 100.0% | 0.9993 |
| L7: no scaffold + execution + body | 0.0% | 6.56e-17 |
### 4. How does the verb-position signal transfer?
We searched every attention head in L5–L20 on 300 held-out pairs, measuring its prediction-position write along μΔ and then patching its clean output into the corrupt run. Ranked by the layer-24 μΔ coordinate of their clean output restores, the strongest heads are L19H31, L20H29, and L20H14. Patching each clean output raises the layer-24 μΔ coordinate and the tool-call logit together, recovering 13.0–31.3% of strict predictions, while direct attention to the changed verb barely differs. Joint patches retain the same sign in search, database and API scaffolds, changing the logit by +2.10 to +2.81 in the clean-to-corrupt direction and −0.57 to −1.40 in reverse. These heads provide a tested transfer route, while strict cross-domain recovery remains 0%; the main text presents this as a partial circuit.

| head | Δ attention to verb | Δ μ write | clean output → corrupt: Δ logit | recovery |
| --- | --- | --- | --- | --- |
| L19H31 | −0.0072 | +0.7165 | +0.5375 | 31.33% |
| L20H29 | −0.0274 | +0.2348 | +0.3904 | 22.00% |
| L20H14 | −0.0153 | +0.1834 | +0.1529 | 13.00% |
### 6. Rank and probability distribution under suppression
Top-1 behavioral flips follow the convention in causal work on directions such as refusal (Arditi et al. 2024), while your near-tie concern calls for the full output distribution. The submission did not report final ranks or probability masses, so we measured final outputs on 300 held-out suppressed prompts per model; the cited 47.9–61.5% values are layer-wise logit-lens readings, not the output distribution. Across all 2,100 prompts, 102 (4.9%) place the native marker at exactly rank 2 and only 2 (0.1%) assign it probability ≥0.30. Granite and Qwen3.5-9B often retain the marker in the top 3, but their median probabilities are only 0.018 and 0.0486. Suppression removes nearly all probability mass even when format pressure keeps the marker nearby in rank, so these are decision flips rather than routine near-tie rerankings. The revision reports rank and probability alongside top-1 in Appendix F.

| Metric | Qwen3-4B | Qwen3-8B | Qwen3-14B | Qwen3.5-4B | Qwen3.5-9B | Mistral | Granite |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Top-3 | 0.7% | 25.3% | 2.7% | 1.3% | 45.0% | 0.7% | 65.7% |
| Top-10 | 13.3% | 62.0% | 39.0% | 46.3% | 100.0% | 8.7% | 100.0% |
| Median prob. | 1.32e-09 | 4.22e-06 | 3.07e-05 | 0.00183 | 0.0486 | 0.000836 | 0.018 |
### 7. Is layer-24 selection post-hoc and circular?
Layer sweeps followed by held-out causal validation are standard in causal tracing (Meng et al. 2022) and direction discovery (Arditi et al. 2024). Our sweep and validation use separate data and metrics: μΔ is fit on training pairs only, and every intervention we report is scored on held-out pairs that contributed nothing to fitting it. The submission also includes a random feature-ablation control (+0.28 versus −27.45 on the μΔ projection). Your question motivated a stronger directional test: we refit μΔ at each resid_pre from L22–L25 and compared it with an equal-norm random direction on 300 disjoint pairs. μΔ reaches full sufficiency from L23 and full necessity from L24. The L24 random addition is visibly nonzero at 56/300, yet μΔ moves 300/300 there and dominates its control at every layer. The result localizes commitment to an L23–L25 window, with L24 its sharpest point; the revision adopts this wording.

| strict <tool_call> top-1 rate | L22 | L23 | L24 | L25 |
| --- | --- | --- | --- | --- |
| corrupt after +μΔ | 82.0% | 100.0% | 100.0% | 100.0% |
| clean after −μΔ | 45.7% | 12.0% | 0.0% | 0.0% |
| corrupt after +random | 0.0% | 0.3% | 18.7% | 0.0% |
| clean after −random | 100.0% | 99.7% | 97.7% | 100.0% |
### 8. Remaining questions
Why commitment lags early separability. A probe tests whether verb identity is readable; attribution tests whether a component causally drives the output (Hewitt & Liang 2019; Belinkov 2022). Our results match the enrichment-and-extraction pattern of Geva et al. 2023: identity is readable early, the L19–L20 heads route it to the final position, and the L23–L25 window makes it causally sufficient.

Failure cases and boundary verbs. On the frozen 300-pair manifest, +μΔ recovers 297/300 prompts; all three failures are Python prompts with the analysis verb discuss, so the residue clusters by verb. In a 20-verb sweep, inspect lies at the boundary with 54.3% top-1 and μΔ coordinate 0.076, while all ten execution verbs reach 100% with coordinates 0.794–1.110 (ρ=0.923, p=7.09e-09). The mechanism degrades gradually across ambiguous verbs, and the L24 coordinate tracks that gradient.

[1] Wang, Kevin Ro, et al. "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small." ICLR 2022.

[2] Meng, Kevin, et al. "Locating and editing factual associations in GPT." NeurIPS 2022.

[3] Barres, V, et al. τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment. arXiv:2506.07982, 2025.


---

## Official Review of Submission13798 by Reviewer LxEy

*Official Review by Reviewer LxEy · 22 Jun 2026, 11:23 (modified: 29 Jul 2026, 02:31)*

### Summary
This paper studies how agentic LLMs internally decide whether to call a tool or respond in text. The authors construct minimal contrastive pairs by swapping a single request verb (e.g., "write" → "discuss") in scaffolded code-completion prompts, reliably flipping the model's first-token prediction between <tool_call> and ordinary text. Using activation patching on Qwen3-8B, they localize the decision to one layer-position coordinate and extract a shared vector µ∆ that is causally sufficient and necessary for the decision. Transcoder analysis reveals an asymmetric mechanism: the prompt scaffold installs a tool-call default, and analysis verbs suppress it by activating MLP features in layers 20–23 that write against the tool-call direction — execution verbs simply fail to trigger this suppressor. The mechanism replicates across Qwen3 scales and three additional model families.

Contribution Type: General: Most submissions will fall into this type.
### Strengths And Weaknesses
#### Strengths

The verb-substitution trick for constructing minimal contrastive pairs within long, scaffolded agentic prompts is clever and reusable. Prior mechanistic work examined short prompts (10–30 tokens); this paper makes mechanistic analysis tractable in prompts spanning hundreds of tokens where no single controllable variable was previously obvious.

The activation-patching localization (97% recovery at one layer-position), sufficiency/necessity scores (0.96/0.94), and targeted feature ablation (98× over random control) are well-executed. The interventions are bidirectional and tested on held-out data.

The asymmetry — execution verbs don't generate a positive tool-call signal but simply fail to activate a suppressor — is surprising and connects meaningfully to Arditi et al.'s refusal direction, suggesting a general "default + suppression" organizational principle in instruction-tuned models.

Appendix B honestly documents why three circuit-search methods fail (the signal is dense, not sparse). Appendix E separates early linear separability from late causal commitment and tests four alternative hypotheses. Appendix F provides seven-model triplet diagnostics. The transparency here exceeds the typical standard.

All experiments run on a single A6000 GPU, with layer indices, intervention parameters, dataset construction pipeline, verb-screening criteria, and train/test splits fully specified. Code and data will also be released making it easier to reproduce.

#### Weaknesses

All tasks are from coding benchmarks. Whether the mechanism holds for non-coding tools (search, calendar, database) where execution-vs-analysis is less cleanly binary is untested and unacknowledged.

Mistral, Devstral, and Granite replicate intervention effectiveness, but the mechanistic explanation (suppression features, MLP asymmetry, analysis-non-execution family) is only available for Qwen3 due to Transcoder constraints. The narrative reads as general but rests on one model family. Even within Qwen3, the 1.7B K_corrupt/K_clean ratio (261.2/163.1) differs radically from 8B (24.4/10.9), suggesting structural variation unexplored.

The authors never test whether the scaffold alone (no user request) produces <tool_call> as top-1. Without this, "default being suppressed" vs. "conjunction failing to complete" is an interpretive choice, not an established fact. This is a single forward pass that would resolve a core framing question.

The paper characterizes formation (L20–23 MLPs writing suppression) and readout (downstream attention + MLP34) but leaves the early-layer verb recognition and mid-layer transfer from verb-position to prediction-position entirely uncharacterized. The actual "decision" — classifying the verb — happens early and is never examined. The paper explains where the answer is stored, not how it's computed.

The title asks "How Do Agentic LLMs Decide to Call Tools?" but the answer covers only: single-turn, verb-swap-mediated, code-completion, first-token-only, one scaffold template. Real tool-call decisions arise from implicit intent ("I'm stuck on this function"), multi-turn context, ambiguous requests, and non-coding domains — none tested.

The paper's own evidence (lines 228–233) reveals <tool_call> remains top-3 on 47.9–61.5% of corrupt prompts at later layers. The entire "flip" story may be a marginal reranking at a near-tie rather than a fundamental decision. No rank distributions, probability masses, or dose-response curves (fractional µ∆ scaling) are reported. This is never acknowledged as a limitation.

Layer-24 selection is post-hoc and partially circular. The layer is chosen because activation patching peaks there; µ∆ is then defined as the mean difference at that layer; then its effectiveness is demonstrated at that layer. Adding the mean clean−corrupt vector to a corrupt state approximately reconstructs the clean state by construction. The paper never reports a random-direction control of equal norm, nor tests whether µ∆ computed at L22 or L23 performs comparably — which would distinguish "layer 24 is uniquely special" from "any late-formation layer works."

Quality: 3: good
Clarity: 2: not good
Significance: 2: not good
Originality: 3: good
### Questions
What is the full rank and probability distribution of <tool_call> on corrupt prompts? Lines 228–233 note that <tool_call> stays in the top-3 at layers 34–35 on 47.9–61.5% of corrupt pairs. But the paper never reports the actual output rank distribution or probability mass on corrupt prompts. If <tool_call> is typically rank 2 with probability 0.30+ even under suppression, the intervention story is "nudging a near-tie" rather than "flipping a decision." A histogram of <tool_call> rank on the 300 held-out corrupt prompts would resolve this.

How does the verb-position signal transfer to the prediction position? Appendix E elegantly separates early linear separability from late causal commitment, confirming the verb-position signal exists early but doesn't causally commit until later. However, the mechanism of transfer remains uncharacterized: which attention heads in layers ~10–20 attend from position p back to position v and carry the verb-identity information? A targeted attention-pattern analysis (e.g., identifying heads whose verb-position attention mass differs across clean/corrupt) would close this gap.

Does the scaffold alone (without any user request) produce <tool_call> as top-1? The paper calls the scaffold's effect a "default" but never tests the literal baseline: scaffold + empty/minimal user request. Appendix E.3 tests the global-direction hypothesis and finds it weaker than local L24 intervention, but that's a different question. The "default" terminology implies <tool_call> would be predicted even without execution intent. Running the scaffold with no user request (or a neutral request like "Hello") would either confirm the default framing or reframe the mechanism as a learned conjunction.

What characterizes the ~3% of held-out prompts where adding µ∆ fails to recover top-1? Table 1 shows 99.21% recovery after adding µ∆. Do the failure cases cluster by task source, programming language, prompt length, or verb? If they reveal systematic boundary conditions (e.g., tasks where the "decision" involves more than verb identity), they would bound the mechanism's scope and suggest what additional factors contribute. If they're random, that strengthens the universality claim. Either answer is informative.

Why does the model delay causal commitment to L25 when probe separability is available at L1? The ~24-layer gap between "information is linearly readable" and "information causally drives the output" is unexplained.

### Limitations
Several critical limitations remain unaddressed or underplayed:

The top-1 metric's fragility is never acknowledged as a limitation. The authors' own evidence (lines 228–233) shows <tool_call> stays in the top-3 on 47.9–61.5% of corrupt prompts. If the suppression signal is merely demoting <tool_call> from rank 1 to rank 2–3, the "decision" is a marginal reranking — not the fundamental computational choice the paper's framing implies. This should be explicitly discussed as a limitation of the metric, not just mentioned in passing as evidence for "format pressure."

The early-layer transfer mechanism is uncharacterized, yet the title claims to explain "how" models decide. Appendix E separates early separability from late commitment but explicitly leaves the transfer mechanism uncharacterized. The paper should acknowledge more directly (in the main text, not just implicitly via omission) that the causal chain from verb recognition to prediction-position commitment is incomplete.

The delay between early separability (L1) and late causal commitment (L25) is unexplained. This ~24-layer gap is a significant unexplained phenomenon that bears directly on whether the "default + suppression" account is complete or whether additional computation is happening in the middle layers.

The "default" terminology is presented as established fact without the corresponding baseline experiment. Running the scaffold without a user request would either confirm or refute this choice of framing. The absence of this trivial experiment is a limitation of the evidence base that should be stated.

The verb set is curated for maximal contrast, and the paper doesn't discuss how rapidly the mechanism degrades as verbs become more ambiguous. The verb-screening tables (Tables 5–6) show many verbs with intermediate rates (e.g., "inspect" at 55.9% <tool_call> on Qwen3-8B corrupt prompts). The paper retains only clean pairs per task but never discusses what happens mechanistically at the boundary — where both interpretations are partially valid.

The domain is exclusively code completion. All 1,500 pairs are drawn from coding benchmarks. Tool-calling in the real world spans many domains — where the relationship between user intent and tool necessity is far more ambiguous than "write code vs. discuss code." The paper never tests whether the same mechanism governs tool-call decisions for non-coding tools (e.g., a search tool invoked by "look up X" vs. "what do you know about X"). The verb-substitution trick may only work cleanly in the coding domain where execution vs. analysis maps neatly onto tool-call vs. text. This domain restriction is never acknowledged as a limitation.

Cross-family "generalization" only tests that µ∆ works, not why. For Mistral, Devstral, and Granite, the paper reports intervention effectiveness but cannot test the core interpretive claims (suppression-dominated formation, analysis-non-execution features, MLP asymmetry) because transcoders don't exist for those architectures. The mechanistic explanation generalizes from one model family; the other three are black-box replications of the vector's effect.

The mechanism is only tested on prompts where a clean verb swap exists. Real agentic requests often lack an explicit action verb entirely: "I'm stuck on this function," "what's wrong with solve.py," "can you help me with the sort logic." The paper's construction requires a single-token verb that cleanly flips behavior — but many real tool-call decisions are driven by implicit intent, multi-word phrasing, or contextual cues that resist this decomposition. Whether the same L24 locus and µ∆ direction mediate verb-free decisions is untested and unacknowledged.

Rating: 3: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.
Confidence: 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
Ethical Concerns: NO or VERY MINOR ethics concerns only
Paper Formatting Concerns:
NA

Code Of Conduct Acknowledgement: Yes
Responsible Reviewing Acknowledgement: Yes

---

## A friendly reminder

*Official Comment by Authors (Lijie Hu, Tingxu Han, Jiahao Zhang, Wei Song, +4 more) · 01 Aug 2026, 23:18*

### Comment
Thank you again for your detailed and thoughtful review!

As the discussion period is approaching its end, we wanted to follow up and ask whether the new analyses and experiments have sufficiently addressed your main concerns. If any issues remain unclear or require further evidence, we would be grateful for the opportunity to respond.

If you find that the major concerns have been resolved, we would sincerely appreciate your consideration of these updates in your final evaluation.


---

## Official Comment by Reviewer LxEy

*Official Comment by Reviewer LxEy · 02 Aug 2026, 23:48*

### Comment
I thank the authors for the additional experiments. The main gap that the mechanistic account is Qwen-only still remains. The title needs to be scoped down to clearly reflect that. Also, please report the τ²-bench results as rates with denominators, at matched (1×) gain — consistent with your synthetic table. Hence, my score stands.


---

## Official Review of Submission13798 by Reviewer CJrQ

*Official Review by Reviewer CJrQ · 11 Jun 2026, 10:26 (modified: 29 Jul 2026, 02:31)*

### Summary
This paper investigates tool calling behavior of LLMs by using execution-oriented verbs to causally manipulate tool calls, specifically "<tool_call>". The paper uses a matched verb-pair setup and also identifies a vector that is necessary and sufficient for producing tool call behavior. The paper is tightly scoped but goes very deep into this one behavior, ie generating "<tool_call>".

Contribution Type: General: Most submissions will fall into this type.
### Strengths And Weaknesses
Both a strength and a weakness of the paper is its tight scope. It is quite intuitive that execution verbs produce tool calls more than analysis verbs. But the authors show that this intuitive behavioral contrast is implemented by a localized residual-stream vector that is both necessary and sufficient for the first-token tool-call decision, and they trace how this vector is formed and read out.

One concern is that the verb contrast may be confounded with the scaffold's tool description. It is not clear whether the model is learning an abstract distinction between directive vs descriptive verbs or simply matching execution verbs like "write" and "build" to the coding tool affordance. It may help to test this with renamed, removed, or mismatched tools. For example, if the available tool is renamed from a coding tool to a generic tool, or if the tool description no longer affords "write" and "build", do the same verbs still produce the same vector and tool-call behavior?

Another related concern is that in practice, agents often call tools without any action verbs. For example, "what does example_file.txt say?" will prompt a tool call without explicitly saying "write" or "build". The paper should clarify whether the proposed mechanism applies to cases where tool use is useful but not explicitly cued by an execution verb. It could potentially be related to reasoning traces, in which case it would be interesting to see if reasoning traces reliably mediate tool calling by generating action verbs.

Overall, I see this as a strong mechanistic paper on a narrow but important decision point. My main concern is that the broader agentic framing may overreach unless the authors show that the same mechanism extends beyond verb-cued coding-tool calls.

Quality: 4: excellent
Clarity: 4: excellent
Significance: 2: not good
Originality: 3: good
### Questions
Is the verb contrast confounded with the scaffold's tool description? Test with renamed, removed, or mismatched tools. For example, rename the coding tool to a generic tool, or remove the "write"/"build" affordance, and check whether the same verbs still produce the vector and the tool-call behavior. Your score increases if the vector survives these manipulations.
Does the mechanism apply when tool use is useful but not cued by an execution verb, e.g., "what does example_file.txt say?"
Do reasoning traces reliably mediate tool calling by generating action verbs?
Does the mechanism extend beyond verb-cued coding-tool calls? The broader agentic framing may overreach otherwise.
### Limitations
Mostly yes. Recommend tempering the broad agentic framing until generalization beyond verb-cued coding-tool calls is shown.

Rating: 5: Accept: Technically solid paper, with high potential value on at least one sub-area of AI or moderate-to-high impact on more than one area of AI, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Ethical Concerns: NO or VERY MINOR ethics concerns only
Paper Formatting Concerns:
None.

Code Of Conduct Acknowledgement: Yes
Responsible Reviewing Acknowledgement: Yes

---

## Rebuttal by Authors

*Rebuttal by Authors (Lijie Hu, Tingxu Han, Jiahao Zhang, Wei Song, +4 more) · 28 Jul 2026, 19:56 (modified: 28 Jul 2026, 21:24)*

### Rebuttal
Thank you for this insightful review, which helped us identify several important directions for strengthening the paper. Below, we address your concerns in detail with additional analyses, new results, and supporting data.

### 1. μΔ survives renaming, removal, and tool mismatch
We ran all three manipulations you proposed. For each model we took μΔ as estimated under its original tool schema and applied it, unchanged, under the edited ones.

Renamed keeps the affordance and schema, changing only the target function name.
Removed replaces the target with opaque f1, an empty description, and neutral parameter names.
Mismatched swaps in get_weather, leaving no tool that fits the request.
μΔ survives all of them. It turns non-calls into calls and calls into non-calls at or near ceiling, before the edits and after every one of them, and removal stays effective on all seven models. The schema decides how often a call is on the table; μΔ decides the call once it is. Two things in the tables below deserve a pointer:

The Qwen3.5 models start below ceiling on their own untouched schemas: 99/300 induction for the 4B and 210/300 removal for the 9B. The sufficiency they lack is not something the edits took from them, and renaming even moves the first from 99 to 104.
The thin denominators under get_weather measure the schema, not a thin sample of μΔ. A weather tool leaves an execution request nothing worth calling, so the Qwen3-8B baseline opens 3 calls instead of 300, and μΔ switched off all 3.
Induction, strict flips after adding μΔ. The denominator is that schema's own baseline-corrupt non-tool population, out of 300 held-out pairs per model.

| Manipulation | Qwen3-4B | Qwen3-8B | Qwen3-14B | Qwen3.5-4B | Qwen3.5-9B | Mistral | Granite |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original schema | 300/300 | 300/300 | 300/300 | 99/300 | 267/300 | 300/300 | 300/300 |
| Renamed | 300/300 | 300/300 | 300/300 | 104/300 | 238/296 | 300/300 | 283/283 |
| Removed | 287/287 | 245/245 | 300/300 | 50/298 | 203/299 | 283/283 | 300/300 |
| Mismatched | 298/300 | 300/300 | 299/300 | 20/300 | 55/300 | 298/298 | 249/300 |
Removal, strict drops after subtracting μΔ. The denominator is how many of the 300 still open a call under that schema, so it also measures the schema's own effect.

| Manipulation | Qwen3-4B | Qwen3-8B | Qwen3-14B | Qwen3.5-4B | Qwen3.5-9B | Mistral | Granite |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original schema | 300/300 | 300/300 | 300/300 | 285/300 | 210/300 | 294/300 | 300/300 |
| Renamed | 300/300 | 300/300 | 296/296 | 271/278 | 197/300 | 299/300 | 281/300 |
| Removed | 230/230 | 299/299 | 137/137 | 97/105 | 289/289 | 299/299 | 199/199 |
| Mismatched | N/A | 3/3 | 14/14 | 143/147 | 285/285 | 30/30 | 9/9 |
The same split shows up in a manipulation we ran alongside yours, affordance reversal. Replace write_file with submit_review and the polarity swaps. Re-oriented to whichever verb now behaves as the call verb, μΔ still gives strict flip 1.00 and drop 1.00 on held-out pairs. It did not weaken; the two verbs traded sides:

| User verb | write_file schema | submit_review schema |
| --- | --- | --- |
| Write | 100% | 34% |
| Review | 4% | 100% |
First-token <tool_call> top-1 rate, Qwen3-8B.

So neither of your candidates was wrong, and the reason the submission could not choose between them is that it never put them anywhere they disagree.

### 2. The mechanism extends beyond verb-cued coding-tool calls
To test the mechanism where agents actually run, we stepped into real τ²-bench[1] trajectories we did not author and intervened on the next turn. Each prompt is the model's own native context at a real decision point in Telecom or Retail, 5,385 to 15,690 tokens with 16 to 43 tool schemas available and every earlier call still in history, no verb of ours anywhere in it. Each model's μΔ goes in with no re-estimation: removal on 200 turns that natively call a tool, induction on 200 that natively answer in text.

The causal effect carries into these trajectories on all seven models and stays clear of its random control everywhere. How far it carries varies: Granite is fully removable at 200/200 yet resists induction at 25/200, Qwen3-8B is the other way round, and Mistral's random control is itself active on induction at 27/200, the one cell where an arbitrary direction moves the decision at a rate worth reporting.

| Direction, gain | Qwen3-4B | Qwen3-8B | Qwen3-14B | Qwen3.5-4B | Qwen3.5-9B | Mistral | Granite |
| --- | --- | --- | --- | --- | --- | --- | --- |
| μΔ, 1× | 101/154 | 73/128 | 187/133 | 181/49 | 171/107 | 73/87 | 182/11 |
| Random, 1× | 16/2 | 13/0 | 1/0 | 30/0 | 0/0 | 2/11 | 7/0 |
| μΔ, 1.5× | 199/199 | 106/200 | 200/200 | 190/135 | 181/187 | 161/157 | 200/25 |
| Random, 1.5× | 28/13 | 30/0 | 9/0 | 46/0 | 0/0 | 4/27 | 58/0 |
Each cell is removal / induction, counted as first-token decisions switched by the intervention. The roster adds Qwen3.5 as the newest family available, and leaves out Qwen3-1.7B, whose native tool calling is too weak to produce baseline-eligible turns, and Devstral, a coding-specialized derivative of the Mistral-Small base already here. Random is an equal-norm seeded direction.

### 3. μΔ also governs tool calls not cued by an action verb
Your example is a fair one, and it sits outside our matched-pair design by construction, since a pair needs a verb to swap. That also means a verb-free request can only ever be held-out test data for us, the stronger place for it to sit. To find out whether μΔ was carrying these calls too, we built 600 verb-free requests, kept the ones each model already answers with a tool call, and subtracted μΔ at the layer and position localized on coding data, against a same-norm random control.

Content: APPS[2] (code), FEVER[3] (search), Spider[4] (database), and Vectrix ART-E (API), with any source ID used in fitting excluded.
Form: five hand-written non-imperative carriers per domain, 30 items each, such as a direct question ("Is it true that ...") or a first-person backlog ("I haven't gotten to ... yet"), all screened by a regex blocklist covering execution and analysis verb families.
Scoring: all 600 re-rendered in each model's native template, at most ten per domain and carrier in fixed construction order, which is why N differs by model.
The weakest case is Mistral: at α=1.5 μΔ removes 90.5% of these calls against 63.9% for the random control, so the margin holds there but it is the thinnest of the seven.

One limit is worth stating plainly. The test is conditional on a baseline call, so what it supports is a shared downstream state that both kinds of request pass through. It does not claim that implicit intent and an explicit verb build that state the same way.

| Model | N | (-μΔ), α=1 | (-μΔ), α=1.5 | (-r), α=1.5 |
| --- | --- | --- | --- | --- |
| Qwen3-4B | 119 | 100.0% | 100.0% | 23.5% |
| Qwen3-8B | 161 | 85.7% | 99.4% | 12.4% |
| Qwen3-14B | 161 | 52.2% | 66.5% | 0.6% |
| Qwen3.5-4B | 189 | 100.0% | 100.0% | 39.7% |
| Qwen3.5-9B | 184 | 96.2% | 100.0% | 0.5% |
| Mistral-Small-3.2-24B | 147 | 80.3% | 90.5% | 63.9% |
| Granite-3.3-8B | 200 | 34.5% | 85.5% | 5.0% |
Strict drops of each model's native first tool-call marker among these baseline-positive requests. r is a near-orthogonal, norm-matched random direction.

Do reasoning traces reliably mediate tool calling by generating action verbs?

One rollout already shows why this needs its own experiment rather than an extension of ours. With thinking enabled on a verb-free read_file prompt, the trace reaches the decision to call the tool before any output token is emitted, and applying (-1.5μΔ) and (-3μΔ) after </think> moves <tool_call> from rank 1 to ranks 2 and 32 while the rollout still emits the same call a sentence later. A finished trace carries the plan past a single late edit, so the trace itself is what a real test would have to intervene on.

Where this goes in the paper
The main text takes three additions: a new Section 3.3 for the τ²-bench intervention and the verb-free requests, and the scaffold ablation reported to Reviewer wPFH. The schema variants, the affordance crossover, and the reasoning-trace rollout go to the appendix, and the default claim reads "a scaffold-conditioned, wording-sensitive call prior" rather than the default response to every concrete task. The title and the motivation stand as submitted.

[1] Barres, V., Dong, H., Ray, S., Si, X., & Narasimhan, K. τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment. arXiv:2506.07982, 2025.

[2] Hendrycks, D., Basart, S., Kadavath, S., Mazeika, M., Arora, A., Guo, E., Burns, C., Puranik, S., He, H., Song, D., & Steinhardt, J. Measuring Coding Challenge Competence With APPS. NeurIPS 2021 Datasets and Benchmarks Track, 2021.

[3] Thorne, J., Vlachos, A., Christodoulopoulos, C., & Mittal, A. FEVER: a Large-scale Dataset for Fact Extraction and VERification. NAACL-HLT, 2018.

[4] Yu, T., Zhang, R., Yang, K., Yasunaga, M., Wang, D., Li, Z., Ma, J., Li, I., Yao, Q., Roman, S., Zhang, Z., & Radev, D. Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task. EMNLP, 2018.


---

## Official Comment by Reviewer CJrQ

*Official Comment by Reviewer CJrQ · 31 Jul 2026, 09:42*

### Comment
I thank the authors for these additional experiments. They resolve my concerns, and I maintain my rating (5).


---

## Official Comment by Authors

*Replying to Official Comment by Reviewer CJrQ*

*Official Comment by Authors (Lijie Hu, Tingxu Han, Jiahao Zhang, Wei Song, +4 more) · 01 Aug 2026, 00:59*

### Comment
Thank you for your kind support. We are grateful that our rebuttal was able to resolve your concerns!

Your affordance question led to the write_file → submit_review crossover, which separated the two explanations you raised. Both turned out to be doing work: the schema determines whether a call is on the table, and μΔ determines whether it is made. Your second question, tool calls with no action verb, became the 600-item verb-free set, which is now our clearest evidence that μΔ is not an artifact of the verb swap.

On the framing: rather than narrow it, we treated your final question as a test the framing had to pass. The τ²-bench trajectories and the verb-free requests are the result, and both go into a new Section 3.3, with the schema variants and the affordance crossover in the appendix. The Significance rating was given before these results existed, so we would be glad if you felt it worth revisiting, entirely as you see fit.


---

## Official Comment by Reviewer CJrQ

*Replying to Official Comment by Authors*

*Official Comment by Reviewer CJrQ · 03 Aug 2026, 22:09*

### Comment
Thank you for the follow-up. The verb-free and τ²-bench results strengthen the broader relevance of the finding, so I am raising Significance to 3.
