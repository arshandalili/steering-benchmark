#!/usr/bin/env bash
set -euo pipefail

ENV_PATH="/data/arshan/hallucination/steering_benchmark/.venv"

if [ -f "${ENV_PATH}/bin/activate" ]; then
  # Works for both venv and conda envs created with a prefix.
  # shellcheck disable=SC1090
  source "${ENV_PATH}/bin/activate"
  return 0
fi

if [ -x "${ENV_PATH}/bin/python" ]; then
  export VIRTUAL_ENV="${ENV_PATH}"
  export PATH="${ENV_PATH}/bin:${PATH}"
  return 0
fi

if command -v conda >/dev/null 2>&1; then
  conda activate "${ENV_PATH}"
  return 0
fi

echo "Could not find environment at ${ENV_PATH}." >&2
return 1
