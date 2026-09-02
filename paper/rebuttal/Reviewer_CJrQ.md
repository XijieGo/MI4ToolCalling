Thank you for this insightful review, which helped us identify several important directions for strengthening the paper. Below, we address your concerns in detail with additional analyses, new results, and supporting data.

### **1. μΔ survives renaming, removal, and tool mismatch**

We ran all three manipulations you proposed. For each model we took μΔ as estimated under its original tool schema and applied it, unchanged, under the edited ones.

- **Renamed** keeps the affordance and schema, changing only the target function name.
- **Removed** replaces the target with opaque `f1`, an empty description, and neutral parameter names.
- **Mismatched** swaps in `get_weather`, leaving no tool that fits the request.

**The vector survives.** Renaming the tool and stripping its description leave μΔ doing what it did before: five of the seven models stay at ceiling in both directions, and the two Qwen3.5 models track their own untouched-schema baselines rather than falling below them. Removal holds on all seven under every edit, at or near each schema's own baseline.

The `Removed` row is the one that answers your confound most directly. A tool named `f1` with an empty description and neutral parameter names gives "write" and "build" nothing to key to, so affordance matching has nothing left to explain the verb contrast with. The contrast survives there anyway, and μΔ still moves it both ways on all seven models.

`Mismatched` is a different kind of test and reads differently. A weather tool does not hide the affordance, it takes away the reason to call, so it moves the baseline along with the schema: the Qwen3-8B execution-verb baseline opens 3 calls instead of 300, and μΔ switched off all 3. Five models stop calling on their own, and μΔ still opens a call on nearly every prompt. **A direction that encoded "this tool fits the request" would have nothing to assert under a weather schema. This one asserts the call anyway, which is what carrying the decision rather than the match looks like.** Qwen3.5 is the exception in the other direction: it holds its own verb contrast under the foreign schema, and there μΔ, fit on `write_file`, transfers only in part.

One reading note for the tables below. The Qwen3.5 models start below ceiling on their own untouched schemas, 99/300 induction for the 4B and 210/300 removal for the 9B. The sufficiency they lack is not something the edits took from them, and renaming even moves the first from 99 to 104.

---

**Induction, strict flips after adding μΔ.** The denominator is that schema's own baseline-corrupt non-tool population, out of 300 held-out pairs per model.

| Manipulation | Qwen3-4B | Qwen3-8B | Qwen3-14B | Qwen3.5-4B | Qwen3.5-9B | Mistral | Granite |
|---|---|---|---|---|---|---|---|
| **Original schema** | **300/300** | **300/300** | **300/300** | **99/300** | **267/300** | **300/300** | **300/300** |
| Renamed | 300/300 | 300/300 | 300/300 | 104/300 | 238/296 | 300/300 | 283/283 |
| Removed | 287/287 | 245/245 | 300/300 | 50/298 | 203/299 | 283/283 | 300/300 |
| Mismatched | 298/300 | 300/300 | 299/300 | 20/300 | 55/300 | 298/298 | 249/300 |

---

**Removal, strict drops after subtracting μΔ.** The denominator is how many of the 300 still open a call under that schema, so it also measures the schema's own effect. 

| Manipulation | Qwen3-4B | Qwen3-8B | Qwen3-14B | Qwen3.5-4B | Qwen3.5-9B | Mistral | Granite |
|---|---|---|---|---|---|---|---|
| **Original schema** | **300/300** | **300/300** | **300/300** | **285/300** | **210/300** | **294/300** | **300/300** |
| Renamed | 300/300 | 300/300 | 296/296 | 271/278 | 197/300 | 299/300 | 281/300 |
| Removed | 230/230 | 299/299 | 137/137 | 97/105 | 289/289 | 299/299 | 199/199 |
| Mismatched | N/A | 3/3 | 14/14 | 143/147 | 285/285 | 30/30 | 9/9 |

---

The same split shows up in a manipulation we ran alongside yours, **affordance reversal**. Replace `write_file` with `submit_review` and the polarity swaps. Re-oriented to whichever verb now behaves as the call verb, μΔ still gives strict flip 1.00 and drop 1.00 on held-out pairs. It did not weaken; the two verbs traded sides:

| User verb | `write_file` schema | `submit_review` schema |
|---|---|---|
| Write | 100% | 34% |
| Review | 4% | 100% |

*First-token `<tool_call>` top-1 rate, Qwen3-8B.*

So neither of your candidates was wrong, and the reason the submission could not choose between them is that it never put them anywhere they disagree.

---

### **2. The mechanism extends beyond verb-cued coding-tool calls**

To test the mechanism where agents actually run, we stepped into **real τ²-bench[1] trajectories we did not author** and intervened on the next turn. Each prompt is the model's own native context at a real decision point in Telecom or Retail, 5,385 to 15,690 tokens with 16 to 43 tool schemas available and every earlier call still in history, no verb of ours anywhere in it. Each model's μΔ goes in with no re-estimation: removal on 200 turns that natively call a tool, induction on 200 that natively answer in text.

The causal effect **carries into these trajectories on all seven models and stays clear of its random control everywhere**. How far it carries varies: Granite is fully removable at 200/200 yet resists induction at 25/200, Qwen3-8B is the other way round, and Mistral's random control is itself active on induction at 27/200, the one cell where an arbitrary direction moves the decision at a rate worth reporting.

| Direction, gain | Qwen3-4B | Qwen3-8B | Qwen3-14B | Qwen3.5-4B | Qwen3.5-9B | Mistral | Granite |
|---|---|---|---|---|---|---|---|
| μΔ, 1× | 101/154 | 73/128 | 187/133 | 181/49 | 171/107 | 73/87 | 182/11 |
| Random, 1× | 16/2 | 13/0 | 1/0 | 30/0 | 0/0 | 2/11 | 7/0 |
| μΔ, 1.5× | 199/199 | 106/200 | 200/200 | 190/135 | 181/187 | 161/157 | 200/25 |
| Random, 1.5× | 28/13 | 30/0 | 9/0 | 46/0 | 0/0 | 4/27 | 58/0 |

*Each cell is removal / induction, counted as first-token decisions switched by the intervention. The roster adds Qwen3.5 as the newest family available, and leaves out Qwen3-1.7B, whose native tool calling is too weak to produce baseline-eligible turns, and Devstral, a coding-specialized derivative of the Mistral-Small base already here. Random is an equal-norm seeded direction.*

---

### **3. μΔ also governs tool calls not cued by an action verb**

Your example is a fair one, and it sits outside our matched-pair design by construction, since a pair needs a verb to swap. That also means **a verb-free request can only ever be held-out test data for us**, the stronger place for it to sit. To find out whether μΔ was carrying these calls too, we built 600 verb-free requests, kept the ones each model already answers with a tool call, and subtracted μΔ at the layer and position localized on coding data, against a same-norm random control.

- **Content**: APPS[2] (code), FEVER[3] (search), Spider[4] (database), and Vectrix ART-E (API), with any source ID used in fitting excluded.
- **Form**: five hand-written non-imperative carriers per domain, 30 items each, such as a direct question ("Is it true that ...") or a first-person backlog ("I haven't gotten to ... yet"), all screened by a regex blocklist covering execution and analysis verb families.
- **Scoring**: all 600 re-rendered in each model's native template, at most ten per domain and carrier in fixed construction order, which is why N differs by model.

**The weakest case is Mistral**: at α=1.5 μΔ removes 90.5% of these calls against 63.9% for the random control, so the margin holds there but it is the thinnest of the seven.

**One limit is worth stating plainly.** The test is conditional on a baseline call, so what it supports is a shared downstream state that both kinds of request pass through. It does not claim that implicit intent and an explicit verb build that state the same way.

| Model | N | (-μΔ), α=1 | (-μΔ), α=1.5 | (-r), α=1.5 |
|---|---:|---:|---:|---:|
| Qwen3-4B | 119 | 100.0% | 100.0% | 23.5% |
| Qwen3-8B | 161 | 85.7% | 99.4% | 12.4% |
| Qwen3-14B | 161 | 52.2% | 66.5% | 0.6% |
| Qwen3.5-4B | 189 | 100.0% | 100.0% | 39.7% |
| Qwen3.5-9B | 184 | 96.2% | 100.0% | 0.5% |
| Mistral-Small-3.2-24B | 147 | 80.3% | 90.5% | 63.9% |
| Granite-3.3-8B | 200 | 34.5% | 85.5% | 5.0% |

*Strict drops of each model's native first tool-call marker among these baseline-positive requests. r is a near-orthogonal, norm-matched random direction.*

> Do reasoning traces reliably mediate tool calling by generating action verbs?

**One rollout already shows why this needs its own experiment** rather than an extension of ours. With thinking enabled on a verb-free `read_file` prompt, the trace reaches the decision to call the tool before any output token is emitted, and applying (-1.5μΔ) and (-3μΔ) after `</think>` moves `<tool_call>` **from rank 1 to ranks 2 and 32** while the rollout still emits the same call a sentence later. A finished trace carries the plan past a single late edit, so the trace itself is what a real test would have to intervene on.

---

### **Where this goes in the paper**

The main text takes three additions: **a new Section 3.3** for the τ²-bench intervention and the verb-free requests, and the scaffold ablation reported to Reviewer wPFH. The schema variants, the affordance crossover, and the reasoning-trace rollout go to the appendix, and the default claim reads "a scaffold-conditioned, wording-sensitive call prior" rather than the default response to every concrete task.

We would like to keep the title, and we think the new experiments earn it. Of the five grounds on which the three reviews contested our scope, four now have direct answers: τ²-bench trajectories for multi-turn and multi-tool, the 600 verb-free requests for verb-cueing and for search, database and API alongside code, and the R/T/F factorial with the schema grid for the single template. The fifth we state rather than argue: this paper is about whether a call begins, not which tool is chosen or how its arguments are written, and the abstract now says so.

---

[1] Barres, V., Dong, H., Ray, S., Si, X., & Narasimhan, K. τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment. arXiv:2506.07982, 2025.

[2] Hendrycks, D., Basart, S., Kadavath, S., Mazeika, M., Arora, A., Guo, E., Burns, C., Puranik, S., He, H., Song, D., & Steinhardt, J. Measuring Coding Challenge Competence With APPS. NeurIPS 2021 Datasets and Benchmarks Track, 2021.

[3] Thorne, J., Vlachos, A., Christodoulopoulos, C., & Mittal, A. FEVER: a Large-scale Dataset for Fact Extraction and VERification. NAACL-HLT, 2018.

[4] Yu, T., Zhang, R., Yang, K., Yasunaga, M., Wang, D., Li, Z., Ma, J., Li, I., Yao, Q., Roman, S., Zhang, Z., & Radev, D. Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task. EMNLP, 2018.