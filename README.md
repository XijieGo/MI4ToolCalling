---
license: apache-2.0
tags:
- mechanistic-interpretability
- tool-calling
- agentic-llm
- transcoder
- feature-steering
---

# [NeurIPS 2026] How Do Agentic LLMs Decide to Call Tools?

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-NeurIPS%202026-3B6EA8)](#citation)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-F9AB00)](https://huggingface.co/datasets/XijieGong/MI4ToolCalling)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFCC4D)](https://huggingface.co/XijieGong/MI4ToolCalling)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-red.svg)](https://www.apache.org/licenses/LICENSE-2.0)

</div>

> Official repository for the NeurIPS 2026 paper **"How Do Agentic LLMs Decide to Call Tools? A Tool-Call Vector Shaped by Suppression."**

## Overview

When an agentic language model receives a tool-enabled prompt, it must decide whether to call a tool before it begins its response. We study this first-token **call-or-no-call** decision through controlled contrastive prompts across Qwen, Mistral, and Granite model families.

We identify a **tool-call vector** that causally controls this decision boundary, and use cross-layer sparse autoencoders (Transcoders) to characterize how it forms. Our results support a suppression account: the agentic scaffold establishes tool calling as a default, while requests that do not require a tool recruit internal features that suppress that default.

## Release

This repository contains:

- Canonical contrastive datasets for tool-calling and no-tool requests.
- Experiment code for causal intervention, readout, Transcoder formation, and scaffold-ablation analyses.
- Per-model result artifacts and cross-model transfer evaluations.
- A compact `mi4tc` library for data handling, interventions, and measurements.

## Repository Structure

```text
MI4Toolcalling/
├── datasets/       # Contrastive inputs, organized by model family
├── experiments/    # Reproduction pipelines and analysis runners
├── results/        # Released measurements and evaluation artifacts
├── scripts/        # Data preparation and utility scripts
├── src/mi4tc/      # Core experimental library
└── tests/          # Input and package validation
```

Model-specific material is aligned across `datasets/<model>/`, `experiments/<model>/`, and `results/<model>/`.

## Installation

```bash
pip install -e .
python tests/test_inputs.py
```

External base-model and Transcoder checkpoint locations are configured through the local environment; released datasets and model artifacts are linked above.

## Citation

```bibtex
@inproceedings{mi4toolcalling2026,
  title     = {How Do Agentic LLMs Decide to Call Tools? A Tool-Call Vector Shaped by Suppression},
  author    = {Anonymous Authors},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

This project and its released datasets are licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
