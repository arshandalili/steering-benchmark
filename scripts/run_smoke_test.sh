#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/activate_env.sh"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export HF_HOME="${ROOT_DIR}/data/hf_home"
export TRANSFORMERS_CACHE="${ROOT_DIR}/data/hf_home/transformers"
export HF_DATASETS_CACHE="${ROOT_DIR}/data/hf_home/datasets"
python -m steering_benchmark.cli \
  --experiment "${ROOT_DIR}/configs/experiments/smoke_tinygpt2.yaml" \
  --registry-dir "${ROOT_DIR}/configs/registry"
