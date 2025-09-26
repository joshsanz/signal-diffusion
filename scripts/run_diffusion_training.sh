#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
CONFIG_PATH=${1:-"${REPO_ROOT}/config/diffusion/flowers.toml"}

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found at ${CONFIG_PATH}" >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  TRAINER=(uv run python -m signal_diffusion.training.diffusion)
elif command -v python3 >/dev/null 2>&1; then
  TRAINER=(python3 -m signal_diffusion.training.diffusion)
else
  echo "Unable to locate 'uv' or 'python3' to launch training" >&2
  exit 1
fi

OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/runs/diffusion/flowers"}
mkdir -p "${OUTPUT_DIR}"

echo "=== Launching diffusion training with config: ${CONFIG_PATH} ==="
"${TRAINER[@]}" "${CONFIG_PATH}" --output-dir "${OUTPUT_DIR}"

echo "Training complete. Outputs written to ${OUTPUT_DIR}"
