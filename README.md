# Steering Benchmark

Benchmarking framework for activation steering in LLMs.

It separates **model adapters**, **dataset loaders**, and **steering methods** so you can run controlled comparisons without rewriting the training/evaluation loop.

## Features

- Unified runner for steering methods that produce hidden-state interventions.
- Registry-based components (`models`, `datasets`, `methods`) defined in YAML.
- Built-in support for sweeps (`factors`, `layers`, `prompt_variants`, `ood`) and side-effect evaluation.
- Utilities for result aggregation (`report.py`) and expected-value checks (`compare.py`).

## Repository Structure

```text
steering_benchmark/
├── configs/
│   ├── experiments/        # Runnable experiment YAMLs
│   ├── registry/           # models.yaml, datasets.yaml, methods.yaml
│   └── expected/           # Reference values for compare.py
├── scripts/
│   ├── activate_env.sh
│   └── run_benchmark.sh
├── steering_benchmark/
│   ├── core/               # runner, interventions, hooks, metrics
│   ├── datasets/           # dataset loaders + BaseDataset
│   ├── methods/            # steering methods + base interface
│   ├── models/             # model adapter implementations
│   ├── cli.py              # CLI entrypoint
│   ├── report.py
│   └── compare.py
├── tests/
├── results/
├── requirements.txt
└── pyproject.toml
```

## Installation

### Prerequisites

- Python `>=3.10`
- Hugging Face access token for gated models (for example Llama 2/3), exported as `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`

### Editable install

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -e .
```

Optional:

```bash
pip install -e ".[sae]"      # sae-lens + safetensors
```

### Environment and Cache

The helper scripts set these automatically:

```bash
export HF_HOME=data/hf_home
export TRANSFORMERS_CACHE=data/hf_home/transformers
export HF_DATASETS_CACHE=data/hf_home/datasets
```

If you use `scripts/activate_env.sh`, note it currently expects a shared env path:

```text
/data/arshan/hallucination/steering_benchmark/.venv
```

Adjust that file or activate your environment manually if your setup differs.

## Quick Start

### Run any experiment

```bash
scripts/run_benchmark.sh configs/experiments/diffmean_tinygpt2_nqswap_context.yaml
```

Or directly with Python:

```bash
python -m steering_benchmark \
  --experiment configs/experiments/diffmean_tinygpt2_nqswap_context.yaml \
  --registry-dir configs/registry \
  --output-dir results
```

## How Experiments Work

Each experiment YAML declares:

- `model`: key in `configs/registry/models.yaml`
- `dataset`: key in `configs/registry/datasets.yaml`
- `method`: key in `configs/registry/methods.yaml`
- optional `*_overrides` for per-run customization
- `run` block for train/eval/test behavior

Minimal example:

```yaml
name: smoke_tinygpt2_nqswap
model: tiny_gpt2
dataset: nq_swap_hf
method: diffmean
method_overrides:
  layer: 1
run:
  eval:
    group: context
    metrics: ["exact_match"]
    max_examples: 8
  factors: [0.5, 1.0]
  splits:
    train: train
    eval: eval
    test: test
```

Runner behavior:

1. Fit steering method on train split.
2. Evaluate baseline + each factor on eval split.
3. Select best factor with `run.factor_selection.metric` (default: first metric).
4. Report baseline and steered metrics on test split.
5. Optionally compute sweeps and side-effect datasets.

## Result Format

Each run writes `results/<experiment_name>.json` with top-level keys:

- `config`
- `factor_selection`
- `test`
- `outputs`
- `sweeps` (alias: `curves`)
- `side_effects`
- `cost`

`test` contains:

- `baseline`: metric summary without intervention
- `steered`: metric summary with selected factor

## Data Formats

### JSONL template datasets

The `jsonl_template` loader expects rows like:

```json
{"question":"...","context":"...","answer_context":"...","answer_param":"..."}
```

Field mapping, prompt templates, and targets are configured in `configs/registry/datasets.yaml`.

### NQ-Swap notes

- Loader: `nq_swap`
- Source: `pminervini/NQ-Swap` via Hugging Face Datasets
- Supports context-vs-param targets, k-shot demonstrations, split policies, and answer extraction policies

Useful eval option for strict QA matching:

```yaml
run:
  eval:
    answer_extraction: first_line
```

## Utilities

### Generate summary tables

```bash
python -m steering_benchmark.report \
  --results-dir results \
  --metric exact_match \
  --out results/report.json
```

### Compare against expected values

```bash
python -m steering_benchmark.compare \
  --results results/spare_llama2_nqswap.json \
  --expected configs/expected/spare_emm_parametric.yaml \
  --dataset nqswap \
  --model llama2_7b \
  --metric exact_match \
  --tolerance 1.0
```
