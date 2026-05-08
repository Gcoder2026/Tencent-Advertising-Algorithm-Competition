#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET="${1:-step3_infer.zip}"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

# Inference upload contains ONLY the files the platform needs at infer time.
mkdir -p "${WORKDIR}/pkg"
cp "${SCRIPT_DIR}/infer.py" "${WORKDIR}/pkg/"
cp "${SCRIPT_DIR}/prepare.sh" "${WORKDIR}/pkg/" || true
mkdir -p "${WORKDIR}/pkg/configs"
cp "${SCRIPT_DIR}/configs/__init__.py" "${WORKDIR}/pkg/configs/"
cp "${SCRIPT_DIR}/configs/baseline.py" "${WORKDIR}/pkg/configs/"
mkdir -p "${WORKDIR}/pkg/src"
cp "${SCRIPT_DIR}/src/__init__.py" "${WORKDIR}/pkg/src/"
cp "${SCRIPT_DIR}/src/utils.py" "${WORKDIR}/pkg/src/"
cp "${SCRIPT_DIR}/src/data.py" "${WORKDIR}/pkg/src/"
cp "${SCRIPT_DIR}/src/model.py" "${WORKDIR}/pkg/src/"
cp "${SCRIPT_DIR}/src/checkpoint.py" "${WORKDIR}/pkg/src/"

# Strip pycache.
find "${WORKDIR}/pkg" -type d -name '__pycache__' -prune -exec rm -rf '{}' +
find "${WORKDIR}/pkg" -name '*.pyc' -delete

# Deterministic zip (sorted entries).
( cd "${WORKDIR}/pkg" && find . -type f | sort | zip -X -@ "${SCRIPT_DIR}/${TARGET}" )

echo "Wrote ${SCRIPT_DIR}/${TARGET}"
sha256sum "${SCRIPT_DIR}/${TARGET}"
ls -lh "${SCRIPT_DIR}/${TARGET}"
