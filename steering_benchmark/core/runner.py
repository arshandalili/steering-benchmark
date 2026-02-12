"""Benchmark runner."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from steering_benchmark.core.metrics import METRICS
from steering_benchmark.core.intervention import scale_intervention
from steering_benchmark.registry import DATASET_REGISTRY, METHOD_REGISTRY, MODEL_REGISTRY
from steering_benchmark.utils.seed import set_seed


def _postprocess_prediction(pred: str, policy: Optional[str]) -> str:
    if pred is None:
        return ""
    if not policy or policy == "none":
        return pred
    text = pred.strip()
    lowered = text.lower()

    if policy in {"answer_only", "auto"}:
        for marker in ("answer:", "assistant:", "response:"):
            idx = lowered.rfind(marker)
            if idx != -1:
                text = text[idx + len(marker) :].strip()
                lowered = text.lower()
                break
        text = re.sub(r"^(the answer is|answer is|it is|it's)\s+", "", text, flags=re.IGNORECASE).strip()

    if policy in {"first_line", "line", "auto"}:
        lines = text.splitlines()
        if lines:
            text = lines[0].strip()
        else:
            text = ""

    if policy in {"first_sentence", "auto"} and text:
        parts = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
        text = parts[0].strip()

    return text


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


def _evaluate(
    model,
    dataset,
    group: str,
    max_examples: Optional[int],
    metrics: List[str],
    gen_cfg: Dict,
    intervention=None,
    split: Optional[str] = None,
    include_outputs: bool = True,
    prompt_transform=None,
    answer_extraction: Optional[str] = None,
    progress: bool = False,
    progress_desc: Optional[str] = None,
) -> Dict:
    scores: Dict[str, List[float]] = {m: [] for m in metrics}
    outputs = []
    examples = dataset.iter_group(group, limit=max_examples, split=split)
    if progress:
        try:
            from tqdm import tqdm

            total = max_examples
            if total is None:
                if hasattr(dataset, "_get_indices"):
                    try:
                        total = len(dataset._get_indices(split))  # type: ignore[attr-defined]
                    except Exception:
                        total = None
                if total is None and hasattr(dataset, "__len__"):
                    try:
                        total = len(dataset)  # type: ignore[arg-type]
                    except Exception:
                        total = None
            examples = tqdm(examples, total=total, desc=progress_desc)
        except Exception:
            pass
    for ex in examples:
        prompt = ex.prompt
        if prompt_transform is not None:
            prompt = prompt_transform(ex)
        pred_raw = model.generate(prompt, intervention=intervention, gen_cfg=gen_cfg)
        pred = _postprocess_prediction(pred_raw, answer_extraction)
        if include_outputs:
            outputs.append(
                {
                    "prompt": ex.prompt,
                    "target": ex.target,
                    "prediction": pred_raw,
                    "prediction_extracted": pred,
                    "group": ex.group,
                    "targets": ex.targets,
                }
            )
        for metric in metrics:
            scores[metric].append(METRICS[metric](pred, ex.target))
            if ex.targets:
                for key, target in ex.targets.items():
                    metric_key = f"{metric}_{key}"
                    if metric_key not in scores:
                        scores[metric_key] = []
                    scores[metric_key].append(METRICS[metric](pred, target))

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
    max_examples = eval_cfg.get("max_examples", 128)
    if max_examples is not None:
        max_examples = int(max_examples)
    metrics = eval_cfg.get("metrics", ["exact_match"])
    gen_cfg = eval_cfg.get("generation", {"max_new_tokens": 16, "do_sample": False})
    answer_extraction = eval_cfg.get("answer_extraction")
    include_outputs = bool(run_cfg.get("save_outputs", True))
    progress = bool(eval_cfg.get("progress", run_cfg.get("progress", True)))

    splits_cfg = run_cfg.get("splits", {})
    eval_split = eval_cfg.get("split", splits_cfg.get("eval"))
    test_split = eval_cfg.get("test_split", splits_cfg.get("test"))

    factors = run_cfg.get("factors", None) or config["method"].get("factors") or [1.0]
    factors = [float(f) for f in factors]

    factor_metric = run_cfg.get("factor_selection", {}).get("metric", metrics[0])

    prompt_transform = getattr(method, "transform_prompt", None)

    eval_baseline = _evaluate(
        model,
        dataset,
        group,
        max_examples,
        metrics,
        gen_cfg,
        intervention=None,
        split=eval_split,
        include_outputs=False,
        prompt_transform=None,
        answer_extraction=answer_extraction,
        progress=progress,
        progress_desc=f"eval {group} baseline",
    )

    eval_factor_results = {}
    for factor in factors:
        scaled = scale_intervention(intervention, factor)
        result = _evaluate(
            model,
            dataset,
            group,
            max_examples,
            metrics,
            gen_cfg,
            intervention=scaled,
            split=eval_split,
            include_outputs=False,
            prompt_transform=prompt_transform,
            answer_extraction=answer_extraction,
            progress=progress,
            progress_desc=f"eval {group} factor {factor}",
        )
        eval_factor_results[str(factor)] = result["summary"]

    best_factor = max(
        factors,
        key=lambda f: eval_factor_results[str(f)].get(factor_metric, float("-inf")),
    )

    test_baseline = _evaluate(
        model,
        dataset,
        group,
        max_examples,
        metrics,
        gen_cfg,
        intervention=None,
        split=test_split,
        include_outputs=include_outputs,
        prompt_transform=None,
        answer_extraction=answer_extraction,
        progress=progress,
        progress_desc=f"test {group} baseline",
    )
    test_steered = _evaluate(
        model,
        dataset,
        group,
        max_examples,
        metrics,
        gen_cfg,
        intervention=scale_intervention(intervention, best_factor),
        split=test_split,
        include_outputs=include_outputs,
        prompt_transform=prompt_transform,
        answer_extraction=answer_extraction,
        progress=progress,
        progress_desc=f"test {group} steered",
    )

    return {
        "config": config,
        "factor_selection": {
            "metric": factor_metric,
            "best_factor": best_factor,
            "eval_baseline": eval_baseline["summary"],
            "eval_factors": eval_factor_results,
        },
        "test": {
            "baseline": test_baseline["summary"],
            "steered": test_steered["summary"],
        },
        "outputs": {
            "baseline": test_baseline["outputs"] if include_outputs else [],
            "steered": test_steered["outputs"] if include_outputs else [],
        },
    }


def save_results(results: Dict, output_dir: str, name: str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result_file = output_path / f"{name}.json"
    with result_file.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return result_file
