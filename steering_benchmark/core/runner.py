"""Benchmark runner."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from steering_benchmark.core.metrics import METRICS
from steering_benchmark.registry import DATASET_REGISTRY, METHOD_REGISTRY, MODEL_REGISTRY
from steering_benchmark.utils.seed import set_seed


def _load_model(model_cfg):
    cfg = dict(model_cfg)
    model_type = cfg.pop("type", "hf")
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    return MODEL_REGISTRY[model_type](**cfg)


def _load_dataset(dataset_cfg):
    loader = dataset_cfg.get("loader")
    if loader not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset loader: {loader}")
    cfg = dict(dataset_cfg)
    cfg.pop("loader", None)
    return DATASET_REGISTRY[loader](**cfg)


def _load_method(method_cfg):
    name = method_cfg.get("name")
    if name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method: {name}")
    cfg = dict(method_cfg)
    cfg.pop("name", None)
    return METHOD_REGISTRY[name](cfg)


def _evaluate(model, dataset, group: str, max_examples: int, metrics: List[str], gen_cfg: Dict, intervention=None) -> Dict:
    scores = {m: [] for m in metrics}
    outputs = []
    examples = dataset.iter_group(group, limit=max_examples)
    for ex in examples:
        pred = model.generate(ex.prompt, intervention=intervention, gen_cfg=gen_cfg)
        outputs.append({"prompt": ex.prompt, "target": ex.target, "prediction": pred, "group": ex.group})
        for metric in metrics:
            scores[metric].append(METRICS[metric](pred, ex.target))

    summary = {m: (sum(vals) / max(len(vals), 1)) for m, vals in scores.items()}
    return {"summary": summary, "outputs": outputs}


def run_benchmark(config: Dict) -> Dict:
    run_cfg = config.get("run", {})
    seed = run_cfg.get("seed")
    if seed is not None:
        set_seed(int(seed))

    model = _load_model(config["model"])
    dataset = _load_dataset(config["dataset"])
    method = _load_method(config["method"])

    intervention = method.fit(model, dataset, run_cfg)

    eval_cfg = run_cfg.get("eval", {})
    group = eval_cfg.get("group", "context")
    max_examples = int(eval_cfg.get("max_examples", 128))
    metrics = eval_cfg.get("metrics", ["exact_match"])
    gen_cfg = eval_cfg.get("generation", {"max_new_tokens": 16, "do_sample": False})

    baseline = _evaluate(model, dataset, group, max_examples, metrics, gen_cfg, intervention=None)
    steered = _evaluate(model, dataset, group, max_examples, metrics, gen_cfg, intervention=intervention)

    return {
        "config": config,
        "baseline": baseline["summary"],
        "steered": steered["summary"],
        "outputs": {
            "baseline": baseline["outputs"],
            "steered": steered["outputs"],
        },
    }


def save_results(results: Dict, output_dir: str, name: str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result_file = output_path / f"{name}.json"
    with result_file.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return result_file
