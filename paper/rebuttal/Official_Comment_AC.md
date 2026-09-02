We thank the Area Chair for the summary of shared concerns, and answer the five points in order.

### **1. Overclaim on the title and narrow problem; evaluation only on coding tools**

We took each model's μΔ, fit on the single-turn coding template, and applied it unchanged in real τ²-bench trajectories: 5,385–15,690 tokens, 16–43 tool schemas, multi-turn. Removal runs on 200 turns that natively call a tool; induction on 200 that natively answer in text.

| Direction, gain | Qwen3-4B | Qwen3-8B | Qwen3-14B | Qwen3.5-4B | Qwen3.5-9B | Mistral | Granite |
|---|---:|---:|---:|---:|---:|---:|---:|
| μΔ, 1× | 101/154 | 73/128 | 187/133 | 181/49 | 171/107 | 73/87 | 182/11 |
| Random, 1× | 16/2 | 13/0 | 1/0 | 30/0 | 0/0 | 2/11 | 7/0 |
| μΔ, 1.5× | 199/199 | 106/200 | 200/200 | 190/135 | 181/187 | 161/157 | 200/25 |
| Random, 1.5× | 28/13 | 30/0 | 9/0 | 46/0 | 0/0 | 4/27 | 58/0 |

*removal / induction, out of 200 each. Random is equal-norm. Qwen3-1.7B (weak native calling) and Devstral (Mistral variant) swapped for Qwen3.5.*

> **μΔ transfers to unseen multi-turn, multi-tool trajectories, clear of its random control everywhere, so the title now matches the evidence.**

### **2. Tool calling without any action verbs**

We built 600 verb-free requests across code (APPS), search (FEVER), database (Spider), and API (ART-E), five non-imperative carriers per domain, screened by a verb blocklist. Keeping the ones each model already answers with a tool call, we subtracted μΔ at the coding-localized layer and position.

| Model | N | −μΔ | −random |
|---|---:|---:|---:|
| Qwen3-4B | 119 | 100.0% | 23.5% |
| Qwen3-8B | 161 | 99.4% | 12.4% |
| Qwen3-14B | 161 | 66.5% | 0.6% |
| Qwen3.5-4B | 189 | 100.0% | 39.7% |
| Qwen3.5-9B | 184 | 100.0% | 0.5% |
| Mistral-3.2-24B | 147 | 90.5% | 63.9% |
| Granite-3.3-8B | 200 | 85.5% | 5.0% |

*Strict drops of the native tool-call marker, α=1.5.*

> **μΔ governs verb-free calls too. The mechanism is not tied to an explicit action verb.**

### **3. Ablation on scaffold result**

We reran the complete R/T/F factorial from scratch, R the role instructions, T the tool schema, F the format template. Qwen3-8B, 300 held-out prompts per condition.

| Scaffold | Neutral `p_call` | Analysis `p_call` | Gap |
|---|---:|---:|---:|
| R + T + F (full) | 0.8547 | 0.0080 | 0.847 |
| T + F (no role) | 0.9429 | 0.1799 | 0.763 |
| R + F (no tool schema) | 0.9423 | 0.4531 | 0.489 |
| R + T (no format) | 1.11e-06 | 1.32e-09 | ~0 |
| F only | 1.0000 | 0.9976 | 0.002 |
| R + length-matched text + F | 0.7645 | 0.1517 | 0.613 |

*Full scaffold with no user request (empty, "Hello," or unrelated): `<tool_call>` is never top-1, p ≤ 5e-04.*

> **The format template installs the tool-call default; the tool schema makes it selective, and the request wording resolves it. The revision attributes the default to the format template rather than the scaffold as a whole.**

### **4. Concerns on major inconsistency of the numerical results**

The reviewer is right on every count. Two archived Qwen3-8B dataset versions were mixed during assembly: v1 (1,711 pairs, 1,204/507) and the balanced v2 (1,500, 1,200/300) that the paper describes, and several v1 numbers survived into the text. Across three repeated runs, a small number of boundary prompts (under 5%) shift between calling and not calling: their logits sit at the bf16 resolution floor, where kernel arithmetic decides the side. The rest are stable. The differences are minor and the conclusions unchanged; we correct them in the revision.

| Item | In submission | Corrected |
|---|---|---|
| Corrupt baseline (A.4 vs Table 1) | 0% vs 5.33% | **3.67%** (v2, 11/300); 5.33% was v1, 27/507 |
| `inspect` count (Table 9 vs A.3) | 300 vs 342 | 300; 342 was the v1 assignment |
| 99.21% vs 97% | compared directly | different interventions, no longer compared |
| D.4 32.5% vs 0.99% | contradictory | single-head L28H3 misplaced into group ablation; removed |
| K ratio (Table 3 vs Table 2) | 24.4/10.9 vs 28.77/12.62 | different sign rule, window, selection; both renamed and defined |
| Code availability | abstract vs checklist | abstract is right; checklist corrected |

> **All causal results were computed on the frozen v2 held-out split and are unchanged. We re-checked every number in the paper against its source artifact, not only the flagged six.**

### **5. Experiments on other model family**

Every intervention above ran on seven models across four families. No usable transcoder exists outside Qwen3, and training a comparable 40-layer bundle costs roughly one billion layer-activation examples. Open-weight agentic models under 20B are also scarce, so the four families we test are close to what the ecosystem supports, and we added Qwen3.5 during the discussion period as the newest one available.

---

Data, code, and intermediate artifacts for every table above are in the `rebuttal/` folder of our anonymous repository, https://anonymous.4open.science/r/MI4ToolCalling.
