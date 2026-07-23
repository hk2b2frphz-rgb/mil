#!/usr/bin/env bash
# Build the isolated Python 3.12 runtime used by the V100 Qwen3-TTS jobs.
# Run this on a GPU node so uv's automatic torch backend sees the cluster CUDA
# environment.  Override versions when the cluster driver requires a specific
# vLLM wheel, e.g. VLLM_VERSION=0.24.0 VLLM_OMNI_VERSION=0.24.0.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -f "$PWD/scripts/setup_proxy.sh" ]]; then
    # shellcheck source=/dev/null
    source "$PWD/scripts/setup_proxy.sh"
fi

VLLM_ENV_DIR="${VLLM_ENV_DIR:-$PWD/.venv-vllm-omni}"
# 0.25.0 is currently only an Omni release candidate; use the newest matching
# stable pair published on PyPI by default.
VLLM_VERSION="${VLLM_VERSION:-0.24.0}"
VLLM_OMNI_VERSION="${VLLM_OMNI_VERSION:-0.24.0}"

command -v uv >/dev/null 2>&1 || {
    echo "ERROR: uv is required" >&2
    exit 1
}

uv venv --python 3.12 --seed "$VLLM_ENV_DIR"
uv pip install --python "$VLLM_ENV_DIR/bin/python" \
    "vllm==$VLLM_VERSION" --torch-backend=auto
uv pip install --python "$VLLM_ENV_DIR/bin/python" \
    "vllm-omni==$VLLM_OMNI_VERSION" \
    numpy soundfile sphn torchaudio uroman scipy PyYAML

"$VLLM_ENV_DIR/bin/python" - <<'PY'
import torch
import torchaudio
import vllm
import vllm_omni

print("torch:", torch.__version__)
print("vllm:", vllm.__version__)
print("vllm-omni:", getattr(vllm_omni, "__version__", "unknown"))
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY

echo "vLLM-Omni environment ready: $VLLM_ENV_DIR"
