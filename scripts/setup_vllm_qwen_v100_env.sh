#!/usr/bin/env bash
# Build a separate, V100-safe vLLM runtime for Qwen/Qwen3.6-27B.
#
# Keep this environment separate from A100, Gemma and Qwen3-TTS runtimes.
# Pin vLLM to the cu126-compatible release used by this cluster.
# vLLM's prebuilt CUDA 12.8/12.9 wheels also omit Volta (sm70), so build the
# CUDA 12.6 source tree against the cu126 PyTorch wheel instead.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -f "$PWD/scripts/setup_proxy.sh" ]]; then
    # shellcheck source=/dev/null
    source "$PWD/scripts/setup_proxy.sh"
fi

VLLM_QWEN_V100_ENV_DIR="${VLLM_QWEN_V100_ENV_DIR:-$PWD/.venv-vllm-qwen-v100}"
VLLM_VERSION="${VLLM_VERSION:-0.21.0}"
[[ "$VLLM_VERSION" == 0.21.0 ]] || {
    echo "ERROR: this V100 profile requires VLLM_VERSION=0.21.0." >&2; exit 1;
}
VLLM_QWEN_V100_SRC_DIR="${VLLM_QWEN_V100_SRC_DIR:-$PWD/.vendor/vllm-qwen-v100-v${VLLM_VERSION}}"
VLLM_QWEN_V100_BUILD_ROOT="${VLLM_QWEN_V100_BUILD_ROOT:-$PWD/.vllm-build/qwen-v100-v${VLLM_VERSION}}"
VLLM_QWEN_V100_BUILD_LOG="${VLLM_QWEN_V100_BUILD_LOG:-$PWD/vllm_qwen_v100_build.log}"
QWEN_V100_MODEL="${QWEN_V100_MODEL:-Qwen/Qwen3.6-27B}"
VLLM_CUDA_MODULE="${VLLM_CUDA_MODULE:-cuda12.6_cudnn9.7.1_nccl2.24.3}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
CUDA_WHEEL_TAG="${CUDA_WHEEL_TAG:-$(basename "$PYTORCH_INDEX_URL")}"
EXPECTED_TORCH_CUDA="${EXPECTED_TORCH_CUDA:-12.6}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.11.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0}"
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers==4.57.1}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0}"
MAX_JOBS="${MAX_JOBS:-8}"
NVCC_THREADS="${NVCC_THREADS:-2}"
mkdir -p "$VLLM_QWEN_V100_ENV_DIR"
VLLM_QWEN_V100_ENV_DIR="$(cd "$VLLM_QWEN_V100_ENV_DIR" && pwd)"
VLLM_PYTHON="$VLLM_QWEN_V100_ENV_DIR/bin/python"

if ! command -v module >/dev/null 2>&1; then
    for module_init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
        if [[ -r "$module_init" ]]; then
            # shellcheck source=/dev/null
            source "$module_init"
            break
        fi
    done
fi
command -v module >/dev/null 2>&1 || { echo "ERROR: environment-modules is unavailable." >&2; exit 1; }
module load "$VLLM_CUDA_MODULE"
command -v nvcc >/dev/null 2>&1 || { echo "ERROR: $VLLM_CUDA_MODULE did not provide nvcc." >&2; exit 1; }
command -v uv >/dev/null 2>&1 || { echo "ERROR: uv is required." >&2; exit 1; }
command -v git >/dev/null 2>&1 || { echo "ERROR: git is required." >&2; exit 1; }

NVCC_RELEASE="$(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | tail -n 1)"
[[ "$NVCC_RELEASE" == "$EXPECTED_TORCH_CUDA"* ]] || {
    echo "ERROR: expected CUDA $EXPECTED_TORCH_CUDA, got ${NVCC_RELEASE:-unknown}." >&2; exit 1;
}
CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
export CUDA_HOME TORCH_CUDA_ARCH_LIST MAX_JOBS NVCC_THREADS
export UV_EXTRA_INDEX_URL="$PYTORCH_INDEX_URL"
export UV_INDEX_STRATEGY="unsafe-best-match"
export PIP_EXTRA_INDEX_URL="$PYTORCH_INDEX_URL"

verify_torch_stack() {
    STAGE="$1" EXPECT_TORCH="$TORCH_VERSION+$CUDA_WHEEL_TAG" \
    EXPECT_TORCHAUDIO="$TORCHAUDIO_VERSION+$CUDA_WHEEL_TAG" \
    EXPECT_TORCHVISION="$TORCHVISION_VERSION+$CUDA_WHEEL_TAG" \
    EXPECT_CUDA="$EXPECTED_TORCH_CUDA" "$VLLM_PYTHON" - <<'PY'
import os
from importlib.metadata import version
import torch

expected = {"torch": os.environ["EXPECT_TORCH"], "torchaudio": os.environ["EXPECT_TORCHAUDIO"], "torchvision": os.environ["EXPECT_TORCHVISION"]}
problems = [f"{name}={version(name)}, expected {want}" for name, want in expected.items() if version(name) != want]
print(f"[{os.environ['STAGE']}] torch={torch.__version__}, cuda={torch.version.cuda}, arches={torch.cuda.get_arch_list()}")
if torch.version.cuda != os.environ["EXPECT_CUDA"]:
    problems.append(f"torch CUDA={torch.version.cuda}, expected {os.environ['EXPECT_CUDA']}")
if "sm_70" not in torch.cuda.get_arch_list():
    problems.append("PyTorch wheel does not contain sm_70")
if not torch.cuda.is_available():
    problems.append("CUDA is unavailable")
if problems:
    raise SystemExit("\n".join(problems))
print(f"[{os.environ['STAGE']}] GPU: {torch.cuda.get_device_name(0)} {torch.cuda.get_device_capability(0)}")
PY
}

verify_runtime() {
    "$VLLM_PYTHON" - <<'PY'
from importlib.metadata import version
from packaging.version import Version
import transformers
import vllm
assert Version(transformers.__version__) == Version("4.57.1"), transformers.__version__
print("transformers:", transformers.__version__)
assert vllm.__version__.split("+")[0] == "0.21.0", vllm.__version__
import vllm._C
print("vllm:", vllm.__version__)
PY
}

echo "===== Qwen3.6-27B V100 vLLM environment ====="
echo "env:          $VLLM_QWEN_V100_ENV_DIR"
echo "model:        $QWEN_V100_MODEL"
echo "vLLM:         $VLLM_VERSION (source build)"
echo "CUDA module:  $VLLM_CUDA_MODULE ($NVCC_RELEASE)"
echo "Torch:        $TORCH_VERSION+$CUDA_WHEEL_TAG (sm70)"
echo "Transformers: $TRANSFORMERS_SPEC"
echo "build log:    $VLLM_QWEN_V100_BUILD_LOG"

if [[ "${FRESH:-0}" != "1" && -x "$VLLM_PYTHON" ]] \
    && verify_torch_stack "resume check" >/dev/null 2>&1 \
    && verify_runtime >/dev/null 2>&1; then
    echo "Qwen3.6-27B vLLM environment already complete: $VLLM_QWEN_V100_ENV_DIR"
    exit 0
fi

# Do not recursively clear an environment inherited through PBS -V.
# uv reuses an existing environment; a missing interpreter is bootstrapped once.
if [[ ! -x "$VLLM_PYTHON" ]]; then
    uv venv --python 3.12 --seed "$VLLM_QWEN_V100_ENV_DIR"
fi

if ! verify_torch_stack "existing torch" >/dev/null 2>&1; then
    uv pip install --python "$VLLM_PYTHON" --reinstall --index-url "$PYTORCH_INDEX_URL" \
        "torch==$TORCH_VERSION+$CUDA_WHEEL_TAG" \
        "torchaudio==$TORCHAUDIO_VERSION+$CUDA_WHEEL_TAG" \
        "torchvision==$TORCHVISION_VERSION+$CUDA_WHEEL_TAG"
fi
verify_torch_stack "torch install"

mkdir -p "$(dirname "$VLLM_QWEN_V100_SRC_DIR")"
if [[ ! -e "$VLLM_QWEN_V100_SRC_DIR" ]]; then
    git clone --branch "v$VLLM_VERSION" --depth 1 https://github.com/vllm-project/vllm.git "$VLLM_QWEN_V100_SRC_DIR"
fi
[[ -d "$VLLM_QWEN_V100_SRC_DIR/.git" ]] || { echo "ERROR: $VLLM_QWEN_V100_SRC_DIR is not a git checkout." >&2; exit 1; }
[[ "$(git -C "$VLLM_QWEN_V100_SRC_DIR" describe --tags --exact-match 2>/dev/null || true)" == "v$VLLM_VERSION" ]] || {
    echo "ERROR: source checkout is not v$VLLM_VERSION." >&2; exit 1;
}

TORCH_CONSTRAINTS="$VLLM_QWEN_V100_ENV_DIR/torch-constraints.txt"
printf 'torch==%s+%s\ntorchaudio==%s+%s\ntorchvision==%s+%s\n' \
    "$TORCH_VERSION" "$CUDA_WHEEL_TAG" "$TORCHAUDIO_VERSION" "$CUDA_WHEEL_TAG" "$TORCHVISION_VERSION" "$CUDA_WHEEL_TAG" > "$TORCH_CONSTRAINTS"

(
    cd "$VLLM_QWEN_V100_SRC_DIR"
    "$VLLM_PYTHON" use_existing_torch.py --prefix
    uv pip install --python "$VLLM_PYTHON" --constraint "$TORCH_CONSTRAINTS" -r requirements/build/cuda.txt
    uv pip install --python "$VLLM_PYTHON" --constraint "$TORCH_CONSTRAINTS" -r requirements/cuda.txt "$TRANSFORMERS_SPEC"
    uv pip install --python "$VLLM_PYTHON" "cmake>=3.26,<4" ninja packaging
    export PATH="$VLLM_QWEN_V100_ENV_DIR/bin:$PATH"
    export CMAKE_BUILD_DIR="$VLLM_QWEN_V100_BUILD_ROOT/cmake"
    mkdir -p "$CMAKE_BUILD_DIR"
    if command -v ccache >/dev/null 2>&1; then
        export CCACHE_DIR="$VLLM_QWEN_V100_BUILD_ROOT/ccache"
        mkdir -p "$CCACHE_DIR"
        export CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache CMAKE_CUDA_COMPILER_LAUNCHER=ccache
    fi
    PIP_CONSTRAINT="$TORCH_CONSTRAINTS" "$VLLM_PYTHON" -m pip install --no-build-isolation --no-deps -v -e . 2>&1 | tee "$VLLM_QWEN_V100_BUILD_LOG"
)
verify_torch_stack "vLLM source build"

# Keep Qwen's config support and the cu126 torch pins after the source build.
# No vllm-omni package is installed in this environment.
uv pip install --python "$VLLM_PYTHON" --constraint "$TORCH_CONSTRAINTS" "$TRANSFORMERS_SPEC" packaging
verify_runtime

# Use vLLM's Qwen3.5 config registry; Transformers 4.57 does not have this type.
# Only the small config is downloaded here, not all model weights.
QWEN_V100_MODEL="$QWEN_V100_MODEL" "$VLLM_PYTHON" - <<'PY'
import os
from vllm.transformers_utils.config import get_config
config = get_config(os.environ["QWEN_V100_MODEL"], trust_remote_code=False)
assert config.model_type == "qwen3_5", config.model_type
print("Qwen3.6-27B config: OK", config.model_type)
PY

echo "Qwen3.6-27B V100 vLLM environment ready: $VLLM_QWEN_V100_ENV_DIR"
