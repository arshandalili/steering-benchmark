# Steering Benchmark

A plug-and-play benchmarking framework for steering methods that modify hidden states in LLMs. The pipeline separates **models**, **datasets**, and **steering methods**, so you can swap any of them without rewriting the evaluation loop.

## Key ideas
- **Unified intervention interface**: all steering methods output an intervention that modifies hidden states at a chosen layer/token position.
- **Registries**: add new datasets or methods by dropping in a module and registering it.
- **Config-driven**: models, datasets, and methods live in YAML registries; experiments reference them and add overrides.

## Layout
```
steering_benchmark/
  configs/
    registry/            # models, datasets, methods
    experiments/         # experiment configs
    expected/            # expected results for comparisons
  steering_benchmark/
    core/                # runner, hooks, metrics
    datasets/            # dataset loaders
    methods/             # steering methods
    models/              # model adapters
  scripts/
    activate_env.sh      # use the shared conda/venv
    run_benchmark.sh     # run an experiment
```

## Environment
This benchmark expects the conda/venv at:
```
/data/arshan/hallucination/steering_benchmark/.venv
```
Activate it with:
```
source steering_benchmark/scripts/activate_env.sh
```
If your default HuggingFace cache is quota-limited, set:
```
export HF_HOME=/data/arshan/hallucination/steering_benchmark/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
```

## Running
```
steering_benchmark/scripts/run_benchmark.sh steering_benchmark/configs/experiments/diffmean_tinygpt2_nqswap_context.yaml
steering_benchmark/scripts/run_benchmark.sh steering_benchmark/configs/experiments/diffmean_tinygpt2_nqswap_param.yaml
steering_benchmark/scripts/run_benchmark.sh steering_benchmark/configs/experiments/actadd_tinygpt2_nqswap_context.yaml
steering_benchmark/scripts/run_benchmark.sh steering_benchmark/configs/experiments/actadd_tinygpt2_nqswap_param.yaml
```

## Dataset format
The default loader (`jsonl_template`) expects JSONL with fields like:
```
{"question": "...", "context": "...", "answer_context": "...", "answer_param": "..."}
```
Prompts are created from templates in `configs/registry/datasets.yaml`. Override the field mapping or templates per dataset if needed.

### NQ-Swap (HuggingFace)
The `nq_swap_hf`, `nq_swap_openbook`, and `nq_swap_closebook` dataset loaders pull from `pminervini/NQ-Swap` via 🤗 Datasets. They use:
- `sub_context` + `sub_answer` for **context** targets
- `org_answer` for **parametric** targets

Evaluation uses the same prompt (with swapped `sub_context`) and reports metrics for both targets:
`exact_match` (context) and `exact_match_param` (parametric).
For SAE-aligned reproduction (same prompt format and no train/eval/test split), use:
- `nq_swap_openbook_sae` or `nq_swap_closebook_sae` (shared splits, no shuffling)
- `run.eval.answer_extraction: first_line` to match the paper's `split("\n")[0]` post-processing.
For open-ended generations, `run.eval.answer_extraction: auto` strips common prefixes and
evaluates the first line/sentence before exact-match scoring.

## AxBench-style factor selection
AxBench selects the best steering **factor** on an evaluation split, then reports results on a held-out test split. Our runner follows this pattern by sweeping `run.factors`, selecting the best factor via `run.factor_selection.metric`, and evaluating the selected factor on `eval.test_split`.

## Steering methods
### DiffMean
Computes the difference between mean hidden states from two groups and adds the resulting direction during generation.

### ActAdd (Activation Addition)
Computes a steering direction from paired positive/negative prompts and adds the direction at a chosen layer/token position.

### Prompt baseline
Adds a natural-language instruction before the prompt (e.g., “use the provided context” or “ignore context and answer from parametric knowledge”).

### SpARE (SAE-based)
Uses a sparse autoencoder (SAE) to compute the feature-space difference between two groups, selects top features, and decodes the resulting direction back into hidden space.

The SAE weights file should contain:
- `encoder_weight` (features x hidden)
- `decoder_weight` (hidden x features)
- optional `encoder_bias`, `decoder_bias`

You can point to the weights via `configs/registry/methods.yaml` or override in the experiment config.

## Comparing to expected results
To compare your run against the expected SpARE values from the paper:
```
python -m steering_benchmark.compare \
  --results results/spare_llama2_nqswap.json \
  --expected steering_benchmark/configs/expected/spare_emm_parametric.yaml \
  --dataset nqswap \
  --model llama2_7b \
  --metric exact_match
```
Adjust tolerance as needed with `--tolerance`.

To compare against the ActAdd baseline table values (reported for Llama3‑8B, Llama2‑7B, Gemma‑2‑9B):
```
python -m steering_benchmark.compare \
  --results results/actadd_tinygpt2_nqswap_param.json \
  --expected steering_benchmark/configs/expected/actadd_emm_emc.yaml \
  --dataset nqswap \
  --model llama2_7b \
  --metric EMM
```
