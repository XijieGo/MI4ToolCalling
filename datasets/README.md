---
license: apache-2.0
task_categories:
- text-generation
language:
- en
tags:
- mechanistic-interpretability
- tool-calling
- agentic-llm
- transcoder
- feature-steering
size_categories:
- 10K<n<100K
---

# How Do Agentic LLMs Decide to Call Tools? A Tool-Call Vector Shaped by Suppression

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-NeurIPS%202026-blue)](https://huggingface.co/papers)
[![HuggingFace Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-yellow)](https://huggingface.co/datasets/XijieGong/MI4ToolCalling)
[![HuggingFace Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-green)](https://huggingface.co/XijieGong/MI4ToolCalling)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-red.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

Official repository and dataset for the paper:  
**"How Do Agentic LLMs Decide to Call Tools? A Tool-Call Vector Shaped by Suppression"**

---

## 📌 Overview

Tool calling is central to modern agentic LLMs, yet the internal mechanism dictating whether an LLM decides to invoke an external tool or respond directly (the **call-or-no-call decision**) has remained poorly understood. Agentic prompts are notoriously long and complex, heavily scaffolded with role instructions, tool schemas, and format templates, making mechanistic analysis difficult.

In this work, we present a mechanistic interpretability study across diverse model families (Qwen, Mistral, Granite):
- **Minimal Contrastive Pairs**: We construct minimal contrastive prompt pairs where a single request verb (e.g., an execution verb like *write* vs. an analysis verb like *discuss*) reliably flips the tool-call decision (outputting `<tool_call>` as the first generated token).
- **The Tool-Call Vector ($\mu_\Delta$)**: We trace the decision to a single internal steering vector $\mu_\Delta$, demonstrating that it is both **causally necessary and sufficient** to govern the tool-calling decision.
- **Scaffold Default Shaped by Suppression**: Using Transcoder analysis (cross-layer Sparse Autoencoders), we uncover how this vector forms: **the agentic scaffold establishes tool-calling as the baseline default**, while analysis requests actively **suppress** this default through specific internal features that signal no tool is needed.

---

## 📂 Dataset Structure

This repository contains the canonical contrastive datasets across multiple model architectures and domains:

```text
datasets/
├── granite_3p3_8b/          # IBM Granite 3.3 8B inputs
│   ├── pair/                # Screened native contrastive pairs (train / heldout)
│   ├── multi_domain/        # Multi-domain controls (code, retrieval, communication, ops)
│   ├── tau2_bench/          # Tau2 trajectory candidate turns
│   └── verb_free/           # Requests with implicit intent
├── mistral_3p2_24b/         # Mistral 3.2 24B inputs
├── qwen3_4b/                # Qwen3 4B inputs
├── qwen3_8b/                # Qwen3 8B inputs (screened native + 300/200 clean reruns)
│   ├── controlled/          # 1,200 train and 300 test controlled code-domain pairs
│   ├── multi_domain/        # Multi-domain controls
│   ├── verb_free/           # 600 cleaned requests + annotated source
│   └── tau2_bench/          # Tau2 benchmark candidate turns
├── qwen3_14b/               # Qwen3 14B inputs
├── qwen35_4b/               # Qwen3.5 4B inputs
└── qwen35_9b/               # Qwen3.5 9B inputs
```

### Dataset Subsets
- **`pair/`**: Minimal contrastive prompt pairs curated to isolate the call-or-no-call decision.
- **`controlled/`**: Code-domain prompt pairs used for precision localization and causal intervention experiments.
- **`multi_domain/`**: Cross-domain tasks spanning code generation, database retrieval, operational workflows, and communication.
- **`verb_free/`**: Natural prompts and requests with implicit intent to evaluate whether the mechanism generalizes beyond explicit imperative verbs.
- **`tau2_bench/`**: Multi-turn dialogue scenarios derived from complex agent trajectories.

---

## 🚀 Quick Start

### Using Hugging Face Datasets
You can inspect and load the dataset directly via Python:

```python
from datasets import load_dataset

# Load the dataset from Hugging Face Hub
ds = load_dataset("XijieGong/MI4ToolCalling")
print(ds)
```

---

## 🔬 Key Scientific Findings

1. **First-Token Call-or-No-Call Decision**: In frontier agentic models (e.g., Qwen3, Mistral), the decision to call a tool is committed at the very first generated token (`<tool_call>`).
2. **Causal Necessity and Sufficiency**: Adding or subtracting the steering vector $\mu_\Delta$ flips the first token between direct response and `<tool_call>` with near-100% fidelity without degrading downstream generation quality.
3. **Suppression, Not Excitation**: Transcoder feature decompositions reveal that the tool-use prompt scaffold pre-activates the tool-calling circuit. Requests that do not require tools actively recruit suppressor features to shut off the tool-calling pathway.

---

## 📖 Citation

If you find this repository, dataset, or paper useful in your research, please cite:

```bibtex
@article{mi4toolcalling2026,
  title={How Do Agentic LLMs Decide to Call Tools? A Tool-Call Vector Shaped by Suppression},
  author={Anonymous Authors},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```

---

## 📜 License

This project and its associated datasets are licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
