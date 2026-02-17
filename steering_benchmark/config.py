"""Config loading and merge helpers."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigError(RuntimeError):
    pass


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_experiment(
    experiment_path: str | Path,
    registry_dir: str | Path,
) -> Dict[str, Any]:
    exp = load_yaml(experiment_path)
    if not exp:
        raise ConfigError(f"Empty experiment config: {experiment_path}")

    registry_dir = Path(registry_dir)
    models = load_yaml(registry_dir / "models.yaml")
    datasets = load_yaml(registry_dir / "datasets.yaml")
    methods = load_yaml(registry_dir / "methods.yaml")

    model_key = exp.get("model")
    dataset_key = exp.get("dataset")
    method_key = exp.get("method")

    if model_key not in models:
        raise ConfigError(f"Unknown model: {model_key}")
    if dataset_key not in datasets:
        raise ConfigError(f"Unknown dataset: {dataset_key}")
    if method_key not in methods:
        raise ConfigError(f"Unknown method: {method_key}")

    model_cfg = deep_merge(models[model_key], exp.get("model_overrides", {}))
    dataset_cfg = deep_merge(datasets[dataset_key], exp.get("dataset_overrides", {}))
    method_cfg = deep_merge(methods[method_key], exp.get("method_overrides", {}))

    resolved = {
        "name": exp.get("name", f"{method_key}_{model_key}_{dataset_key}"),
        "model": model_cfg,
        "dataset": dataset_cfg,
        "method": method_cfg,
        "run": exp.get("run", {}),
        "output_dir": exp.get("output_dir", "results"),
        "registry": {
            "models": models,
            "datasets": datasets,
            "methods": methods,
        },
    }
    return resolved
