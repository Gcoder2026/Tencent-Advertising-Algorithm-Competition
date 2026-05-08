#!/bin/bash
# Build the Step-1 (training) submission zip for AngelML.
#
# Step-1 zip contains everything the platform needs to run training:
#   run.sh, prepare.sh, train.py, src/*, configs/*, requirements.txt.
# It does NOT contain: infer.py, tests/, docs, README, __pycache__, .git, .pyc.
#
# Usage:
#   bash build_step1_submission.sh [output.zip]
# Default output: step1_train.zip in the script's directory.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET="${1:-step1_train.zip}"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

# Stage the contents.
mkdir -p "${WORKDIR}/pkg"
cp "${SCRIPT_DIR}/run.sh"          "${WORKDIR}/pkg/"
cp "${SCRIPT_DIR}/prepare.sh"      "${WORKDIR}/pkg/" || true
cp "${SCRIPT_DIR}/train.py"        "${WORKDIR}/pkg/"
cp "${SCRIPT_DIR}/requirements.txt" "${WORKDIR}/pkg/" || true

mkdir -p "${WORKDIR}/pkg/configs"
cp "${SCRIPT_DIR}/configs/__init__.py"        "${WORKDIR}/pkg/configs/"
cp "${SCRIPT_DIR}/configs/baseline.py"        "${WORKDIR}/pkg/configs/"
cp "${SCRIPT_DIR}/configs/first_submission.py" "${WORKDIR}/pkg/configs/"

mkdir -p "${WORKDIR}/pkg/src"
cp "${SCRIPT_DIR}/src/__init__.py"   "${WORKDIR}/pkg/src/"
cp "${SCRIPT_DIR}/src/utils.py"      "${WORKDIR}/pkg/src/"
cp "${SCRIPT_DIR}/src/data.py"       "${WORKDIR}/pkg/src/"
cp "${SCRIPT_DIR}/src/model.py"      "${WORKDIR}/pkg/src/"
cp "${SCRIPT_DIR}/src/optimizers.py" "${WORKDIR}/pkg/src/"
cp "${SCRIPT_DIR}/src/checkpoint.py" "${WORKDIR}/pkg/src/"
cp "${SCRIPT_DIR}/src/audit.py"      "${WORKDIR}/pkg/src/"
cp "${SCRIPT_DIR}/src/trainer.py"    "${WORKDIR}/pkg/src/"

# Strip pycache and bytecode.
find "${WORKDIR}/pkg" -type d -name '__pycache__' -prune -exec rm -rf '{}' +
find "${WORKDIR}/pkg" -name '*.pyc' -delete

# Make scripts executable so the platform can run them.
chmod +x "${WORKDIR}/pkg/run.sh" 2>/dev/null || true
chmod +x "${WORKDIR}/pkg/prepare.sh" 2>/dev/null || true

# Deterministic zip: sorted entries, no extra attrs.
( cd "${WORKDIR}/pkg" && find . -type f | sort | zip -X -@ "${SCRIPT_DIR}/${TARGET}" >/dev/null )

# Report.
echo "Wrote ${SCRIPT_DIR}/${TARGET}"
SIZE=$(du -h "${SCRIPT_DIR}/${TARGET}" | awk '{print $1}')
echo "Size: ${SIZE}"
echo "Contents:"
( cd "${WORKDIR}/pkg" && find . -type f | sort )
echo "---"
sha256sum "${SCRIPT_DIR}/${TARGET}" 2>/dev/null || echo "(sha256sum not available)"
