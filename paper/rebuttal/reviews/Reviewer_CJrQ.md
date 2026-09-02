**Summary:**

This paper investigates tool calling behavior of LLMs by using execution-oriented verbs to causally manipulate tool calls, specifically "<tool_call>". The paper uses a matched verb-pair setup and also identifies a vector that is necessary and sufficient for producing tool call behavior. The paper is tightly scoped but goes very deep into this one behavior, ie generating "<tool_call>".

Contribution Type: General: Most submissions will fall into this type.

**Strengths And Weaknesses**:

Both a strength and a weakness of the paper is its tight scope. It is quite intuitive that execution verbs produce tool calls more than analysis verbs. But the authors show that this intuitive behavioral contrast is implemented by a localized residual-stream vector that is both necessary and sufficient for the first-token tool-call decision, and they trace how this vector is formed and read out.

One concern is that the verb contrast may be confounded with the scaffold's tool description. It is not clear whether the model is learning an abstract distinction between directive vs descriptive verbs or simply matching execution verbs like "write" and "build" to the coding tool affordance. It may help to test this with renamed, removed, or mismatched tools. For example, if the available tool is renamed from a coding tool to a generic tool, or if the tool description no longer affords "write" and "build", do the same verbs still produce the same vector and tool-call behavior?

Another related concern is that in practice, agents often call tools without any action verbs. For example, "what does example_file.txt say?" will prompt a tool call without explicitly saying "write" or "build". The paper should clarify whether the proposed mechanism applies to cases where tool use is useful but not explicitly cued by an execution verb. It could potentially be related to reasoning traces, in which case it would be interesting to see if reasoning traces reliably mediate tool calling by generating action verbs.

Overall, I see this as a strong mechanistic paper on a narrow but important decision point. My main concern is that the broader agentic framing may overreach unless the authors show that the same mechanism extends beyond verb-cued coding-tool calls.

Quality: 4: excellent
Clarity: 4: excellent
Significance: 2: not good
Originality: 3: good

**Questions**:

Is the verb contrast confounded with the scaffold's tool description? Test with renamed, removed, or mismatched tools. For example, rename the coding tool to a generic tool, or remove the "write"/"build" affordance, and check whether the same verbs still produce the vector and the tool-call behavior. Your score increases if the vector survives these manipulations.

Does the mechanism apply when tool use is useful but not cued by an execution verb, e.g., "what does example_file.txt say?"

Do reasoning traces reliably mediate tool calling by generating action verbs?

Does the mechanism extend beyond verb-cued coding-tool calls? The broader agentic framing may overreach otherwise.

**Limitations**:

Mostly yes. Recommend tempering the broad agentic framing until generalization beyond verb-cued coding-tool calls is shown.

Rating: 5: Accept: Technically solid paper, with high potential value on at least one sub-area of AI or moderate-to-high impact on more than one area of AI, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations.

Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

Ethical Concerns: NO or VERY MINOR ethics concerns only

Paper Formatting Concerns: None.

Code Of Conduct Acknowledgement: Yes

Responsible Reviewing Acknowledgement: Yes
