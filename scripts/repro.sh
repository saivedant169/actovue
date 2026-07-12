#!/usr/bin/env bash
# Reproduce the headline overhead number on a fresh GPU host. This is the launch
# gate: the number the README quotes has to come back within a percent of this.
#
#   MODEL=Qwen/Qwen2.5-7B-Instruct PROBE=actovue/qwen2.5-7b-halu-probe-v1 \
#       bash scripts/repro.sh
#
# It sets the pod up (idempotent), then runs the overhead benchmark at batch 32.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PROBE="${PROBE:-actovue/qwen2.5-7b-halu-probe-v1}"
BATCH="${BATCH:-32}"
MAX_TOKENS="${MAX_TOKENS:-256}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${WORKSPACE:-/workspace}"
VENV="${WORKSPACE}/actovue-venv"

bash "${REPO_DIR}/scripts/pod_setup.sh"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

echo ">> running overhead benchmark: ${MODEL} batch ${BATCH}"
python "${REPO_DIR}/bench/run_overhead.py" \
    --model "${MODEL}" \
    --probe "${PROBE}" \
    --batch "${BATCH}" \
    --max-tokens "${MAX_TOKENS}"
