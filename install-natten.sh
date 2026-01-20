#!/bin/bash
#set -euo pipefail

# Error if we're not in a venv
if ! command -v python >/dev/null 2>&1; then
  echo "python executable not found: make sure uv's venv is activated " >&2
  exit 1
fi

# Clone source
git clone --recursive https://github.com/SHI-Labs/NATTEN
cd NATTEN
git checkout "v0.21.1" .

# Change to uv's pip implementation to keep everything local
sed -i 's/pip /uv pip /' Makefile
sed -i 's/pip$/uv pip/' Makefile
sed -i 's/pip3/uv pip/' Makefile
sed -i 's/-y //' Makefile

# Build
make WORKERS=$(nproc)

# Final messages
echo "============================================================="
echo "You may wish to run 'make test' to verify correct compilation"
echo "Has libnatten:"
python -c "import natten; print(natten.HAS_LIBNATTEN)"

echo "If everything checks out, you may remove the NATTEN directory"

