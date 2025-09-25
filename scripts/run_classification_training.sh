#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
CONFIG_PATH="${REPO_ROOT}/config/classification/test_gender_health_age.toml"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found at ${CONFIG_PATH}" >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  TRAINER=(uv run python -m signal_diffusion.training.classification)
elif command -v python3 >/dev/null 2>&1; then
  TRAINER=(python -m signal_diffusion.training.classification)
else
  echo "Unable to locate 'uv' or 'python' to launch training" >&2
  exit 1
fi

OUTPUT_ROOT=${OUTPUT_ROOT:-"${REPO_ROOT}/runs/test-classifier-smoke"}
mkdir -p "${OUTPUT_ROOT}"

run_training() {
  local dataset=$1
  local config=$2
  local out_dir=$3

  echo "=== Training ${dataset} classifier (config: ${config}) ==="
  mkdir -p "${out_dir}"
  "${TRAINER[@]}" --output-dir "${out_dir}" "${config}"
}

run_training "parkinsons" "${CONFIG_PATH}" "${OUTPUT_ROOT}/parkinsons"

TMP_CONFIG=$(mktemp "${TMPDIR:-/tmp}/seed-config-XXXXXX.toml")
trap 'rm -f "${TMP_CONFIG}"' EXIT

if command -v python3 >/dev/null 2>&1; then
  PYTHONDONTWRITEBYTECODE=1 python3 - "${CONFIG_PATH}" "${TMP_CONFIG}" <<'PY'
from pathlib import Path
import sys
src = Path(sys.argv[1])
dst = Path(sys.argv[2])
text = src.read_text()
if 'name = "parkinsons"' not in text:
    raise SystemExit("template config does not include parkinsons dataset name")
dst.write_text(text.replace('name = "parkinsons"', 'name = "seed"', 1))
PY
elif command -v perl >/dev/null 2>&1; then
  perl -0pe 's/name = "parkinsons"/name = "seed"/' "${CONFIG_PATH}" > "${TMP_CONFIG}"
else
  echo "Need python3 or perl to adapt config for SEED dataset" >&2
  exit 1
fi

run_training "seed" "${TMP_CONFIG}" "${OUTPUT_ROOT}/seed"

echo "All runs complete. Outputs written to ${OUTPUT_ROOT}" 
