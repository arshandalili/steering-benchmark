#!/usr/bin/env bash
set -euo pipefail

ENV_PATH="/data/arshan/hallucination/steering_benchmark/.venv"

if [ -f "${ENV_PATH}/bin/activate" ]; then
  # Works for both venv and conda envs created with a prefix.
  # shellcheck disable=SC1090
  source "${ENV_PATH}/bin/activate"
elif command -v conda >/dev/null 2>&1; then
  conda activate "${ENV_PATH}"
else
  echo "Could not find environment at ${ENV_PATH}." >&2
  exit 1
fi
