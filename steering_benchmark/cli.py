"""Command-line interface for steering benchmark."""
from __future__ import annotations

import argparse
from pathlib import Path

from steering_benchmark import config as config_lib
import steering_benchmark.datasets  # noqa: F401
import steering_benchmark.methods  # noqa: F401
import steering_benchmark.models  # noqa: F401
from steering_benchmark.core.runner import run_benchmark, save_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run steering benchmark")
    parser.add_argument("--experiment", required=True, help="Path to experiment YAML")
    parser.add_argument(
        "--registry-dir",
        default=str(Path(__file__).resolve().parent.parent / "configs" / "registry"),
        help="Directory with registry YAML files",
    )
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = config_lib.resolve_experiment(args.experiment, args.registry_dir)
    if args.output_dir:
        config["output_dir"] = args.output_dir

    results = run_benchmark(config)
    result_path = save_results(results, config["output_dir"], config["name"])
    print(f"Saved results to {result_path}")


if __name__ == "__main__":
    main()
