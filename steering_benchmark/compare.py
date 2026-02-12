"""Compare benchmark results to expected values."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_json(path: str) -> dict:
    import json

    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare results to expected values")
    parser.add_argument("--results", required=True, help="Path to results JSON")
    parser.add_argument("--expected", required=True, help="Path to expected YAML")
    parser.add_argument("--dataset", required=True, help="Dataset key in expected file")
    parser.add_argument("--model", required=True, help="Model key in expected file")
    parser.add_argument("--metric", default="exact_match", help="Metric key in results")
    parser.add_argument("--tolerance", type=float, default=1.0, help="Allowed absolute diff")
    args = parser.parse_args()

    results = load_json(args.results)
    expected = load_yaml(args.expected)

    if "values" in expected:
        expected_value = expected["values"][args.dataset][args.model]
    elif "metrics" in expected:
        metric_key = args.metric
        metric_block = expected["metrics"]
        if metric_key in metric_block:
            expected_value = metric_block[metric_key][args.dataset][args.model]
        else:
            metric_key = metric_key.upper()
            if metric_key not in metric_block:
                raise KeyError(f"Metric '{args.metric}' not found in expected file")
            expected_value = metric_block[metric_key][args.dataset][args.model]
    else:
        raise KeyError("Expected file must contain 'values' or 'metrics'")

    metric_key = args.metric
    metric_aliases = {"EMM": "exact_match_param", "EMC": "exact_match"}
    if metric_key.upper() in metric_aliases:
        metric_key = metric_aliases[metric_key.upper()]

    actual_value = results.get("test", {}).get("steered", {}).get(metric_key)
    if actual_value is None:
        actual_value = results.get("steered", {}).get(metric_key)
    if actual_value is None:
        raise KeyError(f"Metric '{args.metric}' not found in results")

    diff = abs(actual_value - expected_value)
    status = "PASS" if diff <= args.tolerance else "FAIL"
    print(
        f"{status}: {args.dataset}/{args.model} {args.metric} "
        f"actual={actual_value:.2f} expected={expected_value:.2f} diff={diff:.2f} tol={args.tolerance:.2f}"
    )


if __name__ == "__main__":
    main()
