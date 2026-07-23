#!/usr/bin/env bash
# Build the isolated Python 3.12 runtime used by the V100 Qwen3-TTS jobs.
#
# PyTorch's cu128/cu129 wheels no longer contain Volta (sm70) kernels.  Install
# the cu126 wheel explicitly, verify sm70 support, and build vLLM against that
# PyTorch/CUDA combination instead of using vLLM's prebuilt cu129 wheel.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -f "$PWD/scripts/setup_proxy.sh" ]]; then
    # shellcheck source=/dev/null
    source "$PWD/scripts/setup_proxy.sh"
fi

VLLM_ENV_DIR="${VLLM_ENV_DIR:-$PWD/.venv-vllm-omni}"
VLLM_VERSION="${VLLM_VERSION:-0.21.0}"
VLLM_SRC_DIR="${VLLM_SRC_DIR:-$PWD/.vendor/vllm-v${VLLM_VERSION}}"
VLLM_OMNI_VERSION="${VLLM_OMNI_VERSION:-0.21.0rc1}"
VLLM_CUDA_MODULE="${VLLM_CUDA_MODULE:-cuda12.6_cudnn9.7.1_nccl2.24.3}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.11.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0}"
MAX_JOBS="${MAX_JOBS:-4}"
NVCC_THREADS="${NVCC_THREADS:-2}"
VLLM_PYTHON="$VLLM_ENV_DIR/bin/python"

if ! command -v module >/dev/null 2>&1; then
    for init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
        if [[ -r "$init" ]]; then
            # shellcheck source=/dev/null
            source "$init"
            break
        fi
    done
fi
command -v module >/dev/null 2>&1 || {
    echo "ERROR: environment-modules is unavailable" >&2
    exit 1
}
module load "$VLLM_CUDA_MODULE"

command -v nvcc >/dev/null 2>&1 || {
    echo "ERROR: $VLLM_CUDA_MODULE did not provide nvcc" >&2
    exit 1
}
command -v uv >/dev/null 2>&1 || {
    echo "ERROR: uv is required" >&2
    exit 1
}
command -v git >/dev/null 2>&1 || {
    echo "ERROR: git is required to build vLLM" >&2
    exit 1
}

NVCC_RELEASE="$(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | tail -n 1)"
if [[ "$NVCC_RELEASE" != 12.6* ]]; then
    echo "ERROR: expected CUDA 12.6 from $VLLM_CUDA_MODULE, got ${NVCC_RELEASE:-unknown}" >&2
    exit 1
fi

if [[ -z "${CUDA_HOME:-}" ]]; then
    CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
fi
export CUDA_HOME TORCH_CUDA_ARCH_LIST MAX_JOBS NVCC_THREADS

echo "CUDA module: $VLLM_CUDA_MODULE"
echo "CUDA_HOME: $CUDA_HOME"
echo "nvcc: $(command -v nvcc) (release $NVCC_RELEASE)"
echo "PyTorch index: $PYTORCH_INDEX_URL"
echo "vLLM source: $VLLM_SRC_DIR (v$VLLM_VERSION)"
echo "CUDA architectures: $TORCH_CUDA_ARCH_LIST"

uv venv --python 3.12 --seed "$VLLM_ENV_DIR"

# Reinstalling is intentional: an earlier setup may have left CPU-only or
# cu129 PyTorch in this same environment.
uv pip install --python "$VLLM_PYTHON" \
    --reinstall \
    --index-url "$PYTORCH_INDEX_URL" \
    "torch==$TORCH_VERSION" \
    "torchaudio==$TORCHAUDIO_VERSION" \
    "torchvision==$TORCHVISION_VERSION"

"$VLLM_PYTHON" - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("compiled CUDA arches:", torch.cuda.get_arch_list())
if torch.version.cuda != "12.6":
    raise SystemExit(f"expected PyTorch cu126, got CUDA {torch.version.cuda}")
if "sm_70" not in torch.cuda.get_arch_list():
    raise SystemExit("installed PyTorch does not contain V100 sm70 kernels")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable on this node")
print("gpu:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
PY

mkdir -p "$(dirname "$VLLM_SRC_DIR")"
if [[ ! -e "$VLLM_SRC_DIR" ]]; then
    git clone --branch "v$VLLM_VERSION" --depth 1 \
        https://github.com/vllm-project/vllm.git "$VLLM_SRC_DIR"
elif [[ ! -d "$VLLM_SRC_DIR/.git" ]]; then
    echo "ERROR: $VLLM_SRC_DIR exists but is not a vLLM git checkout" >&2
    exit 1
fi

VLLM_SOURCE_TAG="$(git -C "$VLLM_SRC_DIR" describe --tags --exact-match 2>/dev/null || true)"
if [[ "$VLLM_SOURCE_TAG" != "v$VLLM_VERSION" ]]; then
    echo "ERROR: $VLLM_SRC_DIR is at ${VLLM_SOURCE_TAG:-an untagged commit}, expected v$VLLM_VERSION" >&2
    echo "Use an empty VLLM_SRC_DIR for the requested version." >&2
    exit 1
fi

(
    cd "$VLLM_SRC_DIR"
    # Remove only torch/torchaudio/torchvision pins. The source build must use
    # the already verified cu126 PyTorch rather than resolving another wheel.
    "$VLLM_PYTHON" use_existing_torch.py --prefix
    uv pip install --python "$VLLM_PYTHON" -r requirements/build/cuda.txt
    uv pip uninstall --python "$VLLM_PYTHON" vllm >/dev/null 2>&1 || true
    uv pip install --python "$VLLM_PYTHON" --no-build-isolation -e .
)

uv pip install --python "$VLLM_PYTHON" \
    "vllm-omni==$VLLM_OMNI_VERSION" \
    numpy soundfile sphn uroman scipy PyYAML

"$VLLM_PYTHON" - <<'PY'
import torch
import torchaudio
import vllm
import vllm_omni

print("torch:", torch.__version__)
print("torchaudio:", torchaudio.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("vllm:", vllm.__version__)
print("vllm-omni:", getattr(vllm_omni, "__version__", "unknown"))
print("compiled CUDA arches:", torch.cuda.get_arch_list())
if not torch.cuda.is_available():
    raise SystemExit("CUDA became unavailable after installing vLLM-Omni")
if "sm_70" not in torch.cuda.get_arch_list():
    raise SystemExit("sm70 support was replaced while installing vLLM-Omni")
print("gpu:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("CUDA smoke:", (torch.ones(1, device="cuda") + 1).item())
PY

echo "vLLM-Omni environment ready: $VLLM_ENV_DIR"
