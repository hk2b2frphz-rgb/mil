#!/usr/bin/env bash
# Build the isolated Python 3.12 runtime used by the V100 Qwen3-TTS jobs.
# The cluster's V100/570-series-driver environment needs a CUDA 12 wheel.  Do
# not use uv's auto backend: it may choose CPU torch when CUDA is not detected
# reliably during resolution.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -f "$PWD/scripts/setup_proxy.sh" ]]; then
    # shellcheck source=/dev/null
    source "$PWD/scripts/setup_proxy.sh"
fi

VLLM_ENV_DIR="${VLLM_ENV_DIR:-$PWD/.venv-vllm-omni}"
# Keep the pair on the CUDA-12-compatible 0.21 release line. vLLM 0.22+ pulls
# CUDA 13-only dependencies, which cannot run against the cluster driver.
VLLM_VERSION="${VLLM_VERSION:-0.21.0}"
VLLM_OMNI_VERSION="${VLLM_OMNI_VERSION:-0.21.0rc1}"
UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cu129}"
CPU_ARCH="$(uname -m)"
VLLM_WHEEL_URL="${VLLM_WHEEL_URL:-https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu129-cp38-abi3-manylinux_2_34_${CPU_ARCH}.whl}"
VLLM_INSTALL_SPEC="${VLLM_INSTALL_SPEC:-$VLLM_WHEEL_URL}"

command -v uv >/dev/null 2>&1 || {
    echo "ERROR: uv is required" >&2
    exit 1
}

uv venv --python 3.12 --seed "$VLLM_ENV_DIR"
# --reinstall replaces a previous CPU-only torch in an existing venv. Install
# vLLM and Omni together so the later dependency install cannot replace the
# CUDA torch/torchaudio pair with PyPI's CPU wheel.
uv pip install --python "$VLLM_ENV_DIR/bin/python" \
    --reinstall \
    "$VLLM_INSTALL_SPEC" \
    "vllm-omni==$VLLM_OMNI_VERSION" \
    numpy soundfile sphn uroman scipy PyYAML \
    --torch-backend="$UV_TORCH_BACKEND"

"$VLLM_ENV_DIR/bin/python" - <<'PY'
import torch
import torchaudio
import vllm
import vllm_omni

print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("vllm:", vllm.__version__)
print("vllm-omni:", getattr(vllm_omni, "__version__", "unknown"))
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is unavailable: refusing to keep a CPU-only vLLM-Omni environment"
    )
print("gpu:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
PY

echo "vLLM-Omni environment ready: $VLLM_ENV_DIR"
