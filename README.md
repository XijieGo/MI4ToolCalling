Official code for the NeurIPS 2026 paper **"How Do Agentic LLMs Decide to Call Tools? A Tool-Call Vector Shaped by Suppression."**

[Paper](https://neurips.cc/virtual/2026/poster/148769) · [Dataset](https://huggingface.co/datasets/XijieGong/MI4ToolCalling) · [Transcoder Checkpoints](https://huggingface.co/XijieGong/MI4ToolCalling)

## Overview

This repository contains experiments on the first-token tool-call decision across seven models from the Qwen, Mistral, and Granite families. It provides vector estimation, causal intervention, scaffold ablation, Transcoder feature analysis, and cross-domain and multi-turn evaluation code.

## Repository structure

```text
MI4ToolCalling/
├── experiments/    # Model-specific and cross-model experiment runners
├── scripts/        # Data preparation, execution and analysis scripts
├── src/mi4tc/      # Shared loaders, hooks, metrics and checkpoint adapters
├── tests/          # Input integrity and causal-measurement checks
├── .env.example    # Model and checkpoint path configuration
└── LICENSE         # Apache License 2.0
```

Datasets and checkpoint weights are distributed through Hugging Face. Downloaded inputs go under `datasets/`; experiment outputs go under `results/` or `runs/`. These local directories are excluded from Git.

## Installation

Use Python 3.10 or later:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[model,mistral,analysis,hub,test]"
```

GPU experiments use PyTorch and Transformers. The optional `legacy` extra adds TransformerLens for the MLP patch comparison.

## Download inputs

The dataset repository is a frozen file bundle. Download it into the local dataset directory so that manifests and rendered prompts retain their relative paths:

```bash
hf download XijieGong/MI4ToolCalling --type dataset --local-dir datasets
python tests/test_inputs.py
```

The seven primary model keys and corresponding base models are:

| Model key | Hugging Face base model |
|---|---|
| `qwen3_4b` | `Qwen/Qwen3-4B` |
| `qwen3_8b` | `Qwen/Qwen3-8B` |
| `qwen3_14b` | `Qwen/Qwen3-14B` |
| `qwen35_4b` | `Qwen/Qwen3.5-4B` |
| `qwen35_9b` | `Qwen/Qwen3.5-9B` |
| `mistral_3p2_24b` | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` |
| `granite_3p3_8b` | `ibm-granite/granite-3.3-8b-instruct` |

For example, download Qwen3-8B and its Transcoders:

```bash
hf download Qwen/Qwen3-8B --local-dir external/models/Qwen3-8B
hf download mwhanna/qwen3-8b-transcoders \
  --include "*.safetensors" --local-dir external/transcoders/Qwen3-8B
```

Download the Qwen3.5, Mistral and Granite Transcoders:

```bash
hf download XijieGong/MI4ToolCalling --local-dir external/transcoders/release
```

By default, runners use `external/models/<model-name>` and `external/transcoders/`. To use existing local weights, copy `.env.example` to `.env`, edit the paths, and export them before running:

```bash
cp .env.example .env
# Edit .env with your local model and checkpoint paths.
set -a
source .env
set +a
```

## Run experiments

From the repository root:

```bash
# Fit the training-split vector and evaluate held-out sufficiency / necessity.
bash scripts/run_tool_call_vector.sh qwen3_8b

# Evaluate cross-domain, verb-free and multi-turn transfer.
bash scripts/run_transfer.sh qwen3_8b

# Evaluate scaffold, formation and downstream readout mechanisms.
bash scripts/run_qwen3_8b_full_mechanism.sh
```

Omit the model argument to run the vector or transfer script across all seven models. The vector runners use the fixed zero-based block-input layers documented in the [vector instructions](experiments/cross_model/tool_call_vector/README.md). Per-run outputs record the model, input split, layer and hook convention.

The detailed Qwen3-8B feature-family and mediation analyses use `experiments/qwen3_8b/formation_transcoder/reanalyze.py` and `complete_controls.py`. The cross-model mechanism runner is `experiments/cross_model/recompute_mechanisms.py`. Each provides command-line options through `--help`.

See the [cross-model instructions](experiments/cross_model/README.md), [transfer instructions](experiments/cross_model/transfer/README.md) and [Qwen3-8B localization control](experiments/qwen3_8b/pair/README.md) for additional commands.

## Validation

```bash
python tests/test_inputs.py
python -m unittest discover -s tests -v
```

The input validator checks the frozen manifests, prompt hashes and splits. The unit tests check attention observation, residual-write accounting, block-input intervention conventions and parallel tau2 metric aggregation.

## Citation

```bibtex
@misc{gong2026agenticllmsdecidetools,
      title={How Do Agentic LLMs Decide to Call Tools? A Tool-Call Vector Shaped by Suppression}, 
      author={Xijie Gong and Tingxu Han and Jiahao Zhang and Wei Song and Ziqi Ding and Hanqi Yan and Youcheng Sun and Lijie Hu},
      year={2026},
      eprint={2610.09624},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2610.09624}, 
}
```

## License

Original project code is licensed under [Apache License 2.0](LICENSE). Source benchmark content and third-party checkpoints retain their original licenses. The bundled tau2 rendering templates retain the [tau2-bench MIT license](experiments/cross_model/transfer/templates/LICENSE); dataset and checkpoint provenance is documented in the corresponding Hugging Face repositories.
