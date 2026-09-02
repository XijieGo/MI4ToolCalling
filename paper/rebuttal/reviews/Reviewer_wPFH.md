## Official Review

Summary:

This paper studies how large language models(LLMs) with agentic capabilities internally decide whether to call external tools or answer directly in text. The paper investigates a single-turn code-completion task with a write_file tool. The authors create controlled prompt pairs differing by only one word, testing execution verbs like "write" versus analysis verbs like "discuss", while holding the tool scaffold and task fixed. Using mechanistic interpretability techniques, including activation patching and vector interventions, they identify a localized residual-stream direction at layer 24 that causally mediates this call/no-call decision on the first generated token, which precedes tool use. The authors argue that tool calling acts as a scaffold-induced default behavior, with analysis verbs suppressing rather than execution verbs actively promoting it. Cross-model experiments across Qwen3 variants and other LLMs show similar localization patterns, providing causal evidence for a compact internal representation of agentic decisions in long, scaffolded prompts.

Contribution Type: General: Most submissions will fall into this type.
Strengths And Weaknesses:
Strengths

The paper investigates an important underexplored question of whether agentic LLMs internally decide whether to call a tool or provide a direct answer, employing an elegant verb-substitution design that provides a clean causal handle within long, confounded prompts. It provides a detailed description of the experiment and an extensive analysis of the results.

Mechanistic analysis goes beyond correlation: activation patching localizes the decision, and vector addition/removal interventions provide causal evidence that a single residual-stream direction 
 at one layer/position is both sufficient and necessary for first-token tool calling.

Transcoder decomposition evidence suggests interpretable feature-level evidence showing analysis verbs suppress scaffold-biased features rather than execution verbs actively promoting tool use, with multi-stage tracing identifying compact representation rather than distributed processing.

Cross-model checks across Qwen3 variants and additional families strengthen robustness claims, while transparent limitations acknowledgment and detailed appendices document dataset construction, failed circuit searches, and downstream analyses comprehensively.

Weaknesses

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
Questions:
Please clarify the contradictory values outlined in the weaknesses and the reproducibility contradiction.

Please provide experiments directly isolating the scaffold's causal role, such as no-tool scaffolds, alternate tool schemas, or varied templates, measuring whether suppression effects persist and have mechanistic confirmation?

Please elaborate on the mechanistic details for other models' experiments.

Limitations:
yes

Rating: 3: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.
Confidence: 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
Ethical Concerns: NO or VERY MINOR ethics concerns only
Paper Formatting Concerns:
no

Code Of Conduct Acknowledgement: Yes
Responsible Reviewing Acknowledgement: Yes


## Official Comment to Rebuttal

Dear authors,

Thank you for addressing most of the outlined problems and improving the paper.

Even though the rebuttal fixed some issues and added behavioral validation, the core limitation remains: the feature-level mechanism is Qwen3-only. The cross-family generalization claim announced in the title remains unconfirmed at deeper levels outside the Qwen family. The paper provided as an example that single-model evaluation is enough (Hanna and Ameisen) can only be treated as a venue precedent, not evidence that the design is universally adequate. The fundamental gap between claim scope (agentic LLMs) and evidence depth (one architecture's mechanism) isn't fixable by reframing alone.

This is obviously a strong workshop paper, but for the main track submission, my recommendation Borderline Reject stands.