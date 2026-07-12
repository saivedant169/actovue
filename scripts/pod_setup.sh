#!/usr/bin/env bash
# Idempotent bootstrap for a rented GPU host (RunPod, Vast, Lambda, or a laptop
# with an NVIDIA card). Safe to re-run: it skips work that is already done.
#
# Blackwell cards (RTX 50 series, compute capability sm_120) need CUDA 12.8 or
# newer wheels. That is the cu128 vLLM nightly and a matching torch, which is why
# the index below points at cu128. Hopper (H100) works with the same wheels.
#
#   bash scripts/pod_setup.sh
#
# Then set your probe and serve:
#   export ACTOVUE_PROBE=actovue/qwen2.5-7b-halu-probe-v1
#   vllm serve Qwen/Qwen2.5-7B-Instruct \
#       --worker-extension-cls actovue.worker_ext.ProbeWorkerExtension

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
VENV="${WORKSPACE}/actovue-venv"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_CUDA="${TORCH_CUDA:-cu128}"

echo ">> workspace: ${WORKSPACE}"
mkdir -p "${WORKSPACE}"

if ! command -v uv >/dev/null 2>&1; then
    echo ">> installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

if [ ! -d "${VENV}" ]; then
    echo ">> creating venv (python ${PYTHON_VERSION})"
    uv venv --python "${PYTHON_VERSION}" "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

echo ">> installing torch + vllm nightly (${TORCH_CUDA})"
uv pip install --quiet torch --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
uv pip install --quiet --pre vllm --extra-index-url "https://wheels.vllm.ai/nightly"

echo ">> installing actovue (editable)"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
uv pip install --quiet -e "${REPO_DIR}"

python - <<'PY'
import torch
print(f">> torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f">> gpu: {name}, compute capability {cap[0]}.{cap[1]}")
PY

echo ">> done. HF cache lives under \$HF_HOME; set it to a persistent volume to avoid re-downloads."
