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

## Running
```
steering_benchmark/scripts/run_benchmark.sh steering_benchmark/configs/experiments/diffmean_llama2_nqswap.yaml
steering_benchmark/scripts/run_benchmark.sh steering_benchmark/configs/experiments/spare_llama2_nqswap.yaml
```

## Dataset format
The default loader (`jsonl_template`) expects JSONL with fields like:
```
{"question": "...", "context": "...", "answer_context": "...", "answer_param": "..."}
```
Prompts are created from templates in `configs/registry/datasets.yaml`. Override the field mapping or templates per dataset if needed.

## Steering methods
### DiffMean
Computes the difference between mean hidden states from two groups and adds the resulting direction during generation.

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
