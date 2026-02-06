#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/activate_env.sh"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <experiment_yaml> [extra args]" >&2
  exit 1
fi

EXP_PATH="$1"
shift

python -m steering_benchmark --experiment "${EXP_PATH}" "$@"
