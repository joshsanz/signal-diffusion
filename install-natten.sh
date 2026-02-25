#!/usr/bin/env bash
set -euo pipefail

_die() {
  echo "install-natten.sh: $*" >&2
  exit 1
}

# Tools we rely on.
command -v git >/dev/null 2>&1 || _die "git not found"
command -v make >/dev/null 2>&1 || _die "make not found"
command -v uv >/dev/null 2>&1 || _die "uv not found (install uv and run 'uv sync')"

# Ensure `uv pip` installs into the project venv.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  _die "no active virtualenv detected; run 'source .venv/bin/activate' first"
fi

# Ensure we use the correct nvcc.
# Many native builds ignore CUDACXX and just execute `nvcc` from PATH.
if [[ -z "${CUDA_HOME:-}" ]]; then
  if [[ -d "/usr/local/cuda-12.8" ]]; then
    export CUDA_HOME="/usr/local/cuda-12.8"
  elif [[ -d "/usr/local/cuda" ]]; then
    export CUDA_HOME="/usr/local/cuda"
  else
    _die "CUDA_HOME is not set and no CUDA toolkit found under /usr/local/cuda-*"
  fi
fi

export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CUDACXX="${CUDACXX:-${CUDA_HOME}/bin/nvcc}"

echo "CUDA_HOME=${CUDA_HOME}"
echo "nvcc=$(command -v nvcc)"
nvcc --version | sed -n '1,6p'

# Limit worker count to avoid overwhelming WSL/Cursor.
# Override by exporting NATTEN_WORKERS (e.g. NATTEN_WORKERS=2).
NATTEN_WORKERS="${NATTEN_WORKERS:-8}"

# Clone / update source.
NATTEN_DIR="NATTEN"
NATTEN_TAG="v0.21.1"

if [[ ! -d "${NATTEN_DIR}" ]]; then
  git clone --recursive --branch "${NATTEN_TAG}" https://github.com/SHI-Labs/NATTEN "${NATTEN_DIR}"
fi

cd "${NATTEN_DIR}"
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || _die "${NATTEN_DIR} is not a git repo"
git fetch --tags --force
git checkout -f "${NATTEN_TAG}"

# IMPORTANT: Remove any leftover files from other NATTEN versions.
# This prevents mixed installs like having both `natten/token_permute.py` and
# `natten/token_permute/` present, which can break imports.
git clean -xfd

# Keep everything local by rewriting Makefile's pip invocations.
sed -i 's/pip /uv pip /' Makefile
sed -i 's/pip$/uv pip/' Makefile
sed -i 's/pip3/uv pip/' Makefile
sed -i 's/-y //' Makefile

# Ensure CMake doesn't keep a cached nvcc from a previous build.
rm -rf build_dir

# Build + install.
make WORKERS="${NATTEN_WORKERS}"

echo "============================================================="
echo "You may wish to run 'make test' to verify correct compilation"
echo "Has libnatten:"
python3 -c "import natten; print(natten.HAS_LIBNATTEN)"
echo "If everything checks out, you may remove the NATTEN directory"

