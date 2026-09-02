Summary:
This paper studies how agentic LLMs internally decide whether to call a tool or respond in text. The authors construct minimal contrastive pairs by swapping a single request verb (e.g., "write" → "discuss") in scaffolded code-completion prompts, reliably flipping the model's first-token prediction between <tool_call> and ordinary text. Using activation patching on Qwen3-8B, they localize the decision to one layer-position coordinate and extract a shared vector µ∆ that is causally sufficient and necessary for the decision. Transcoder analysis reveals an asymmetric mechanism: the prompt scaffold installs a tool-call default, and analysis verbs suppress it by activating MLP features in layers 20–23 that write against the tool-call direction — execution verbs simply fail to trigger this suppressor. The mechanism replicates across Qwen3 scales and three additional model families.

Contribution Type: General: Most submissions will fall into this type.
Strengths And Weaknesses:
Strengths

The verb-substitution trick for constructing minimal contrastive pairs within long, scaffolded agentic prompts is clever and reusable. Prior mechanistic work examined short prompts (10–30 tokens); this paper makes mechanistic analysis tractable in prompts spanning hundreds of tokens where no single controllable variable was previously obvious.

The activation-patching localization (97% recovery at one layer-position), sufficiency/necessity scores (0.96/0.94), and targeted feature ablation (98× over random control) are well-executed. The interventions are bidirectional and tested on held-out data.

The asymmetry — execution verbs don't generate a positive tool-call signal but simply fail to activate a suppressor — is surprising and connects meaningfully to Arditi et al.'s refusal direction, suggesting a general "default + suppression" organizational principle in instruction-tuned models.

Appendix B honestly documents why three circuit-search methods fail (the signal is dense, not sparse). Appendix E separates early linear separability from late causal commitment and tests four alternative hypotheses. Appendix F provides seven-model triplet diagnostics. The transparency here exceeds the typical standard.

All experiments run on a single A6000 GPU, with layer indices, intervention parameters, dataset construction pipeline, verb-screening criteria, and train/test splits fully specified. Code and data will also be released making it easier to reproduce.

Weaknesses

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
Questions:
What is the full rank and probability distribution of <tool_call> on corrupt prompts? Lines 228–233 note that <tool_call> stays in the top-3 at layers 34–35 on 47.9–61.5% of corrupt pairs. But the paper never reports the actual output rank distribution or probability mass on corrupt prompts. If <tool_call> is typically rank 2 with probability 0.30+ even under suppression, the intervention story is "nudging a near-tie" rather than "flipping a decision." A histogram of <tool_call> rank on the 300 held-out corrupt prompts would resolve this.

How does the verb-position signal transfer to the prediction position? Appendix E elegantly separates early linear separability from late causal commitment, confirming the verb-position signal exists early but doesn't causally commit until later. However, the mechanism of transfer remains uncharacterized: which attention heads in layers ~10–20 attend from position p back to position v and carry the verb-identity information? A targeted attention-pattern analysis (e.g., identifying heads whose verb-position attention mass differs across clean/corrupt) would close this gap.

Does the scaffold alone (without any user request) produce <tool_call> as top-1? The paper calls the scaffold's effect a "default" but never tests the literal baseline: scaffold + empty/minimal user request. Appendix E.3 tests the global-direction hypothesis and finds it weaker than local L24 intervention, but that's a different question. The "default" terminology implies <tool_call> would be predicted even without execution intent. Running the scaffold with no user request (or a neutral request like "Hello") would either confirm the default framing or reframe the mechanism as a learned conjunction.

What characterizes the ~3% of held-out prompts where adding µ∆ fails to recover top-1? Table 1 shows 99.21% recovery after adding µ∆. Do the failure cases cluster by task source, programming language, prompt length, or verb? If they reveal systematic boundary conditions (e.g., tasks where the "decision" involves more than verb identity), they would bound the mechanism's scope and suggest what additional factors contribute. If they're random, that strengthens the universality claim. Either answer is informative.

Why does the model delay causal commitment to L25 when probe separability is available at L1? The ~24-layer gap between "information is linearly readable" and "information causally drives the output" is unexplained.

Limitations:
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


## Official Comment to Rebuttal

I thank the authors for the additional experiments. The main gap that the mechanistic account is Qwen-only still remains. The title needs to be scoped down to clearly reflect that. Also, please report the τ²-bench results as rates with denominators, at matched (1×) gain — consistent with your synthetic table. Hence, my score stands.