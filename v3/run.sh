#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Determinism env (must be set before python starts).
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

python3 -u "${SCRIPT_DIR}/train.py" --config v3_xfmr_rope_focal "$@"
