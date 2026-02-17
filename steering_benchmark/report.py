"""Generate summary tables from benchmark outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_results(results_dir: Path) -> List[Dict]:
    results = []
    for path in results_dir.glob("*.json"):
        with path.open("r", encoding="utf-8") as handle:
            results.append(json.load(handle))
    return results


def _auc_from_curve(curve: Dict[str, Dict[str, float]], metric: str) -> float:
    if not curve:
        return 0.0
    values = []
    for _, stats in curve.items():
        if metric in stats:
            values.append(stats[metric])
    if not values:
        return 0.0
    return sum(values) / len(values)


def generate_tables(results: List[Dict], metric: str) -> Dict:
    leaderboard = []
    cost_table = []
    robustness = []
    for entry in results:
        config = entry.get("config", {})
        method = config.get("method", {}).get("name", "unknown")
        dataset = config.get("dataset", {}).get("loader", "unknown")
        model = config.get("model", {}).get("hf_id", config.get("model", {}).get("type", "unknown"))
        score = entry.get("test", {}).get("steered", {}).get(metric)
        leaderboard.append(
            {"method": method, "dataset": dataset, "model": model, "metric": metric, "score": score}
        )
        cost = entry.get("cost", {})
        if cost:
            cost_table.append({"method": method, "dataset": dataset, "model": model, **cost})
        curves = entry.get("sweeps", {}).get("factors", {})
        robustness.append(
            {
                "method": method,
                "dataset": dataset,
                "model": model,
                "metric": metric,
                "auc": _auc_from_curve(curves, metric),
            }
        )
    return {
        "leaderboard": leaderboard,
        "cost_vs_performance": cost_table,
        "robustness_auc": robustness,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark summary tables")
    parser.add_argument("--results-dir", required=True, help="Directory containing result JSON files")
    parser.add_argument("--metric", default="exact_match", help="Metric to report")
    parser.add_argument("--out", default=None, help="Output path for JSON tables")
    args = parser.parse_args()

    results = load_results(Path(args.results_dir))
    tables = generate_tables(results, args.metric)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(tables, handle, indent=2)
        print(f"Saved tables to {out_path}")
    else:
        print(json.dumps(tables, indent=2))


if __name__ == "__main__":
    main()
