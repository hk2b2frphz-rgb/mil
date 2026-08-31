#!/usr/bin/env bash
set -euo pipefail

# Rebuild and validate the isolated Gemma 4 runtime used by cascade/SpeechLLM.
# In particular, verifies that PyTorch 2.6 can load its bundled cuDNN 9 on the
# allocated GPU before a long evaluation begins.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

command -v uv >/dev/null 2>&1 || { echo "ERROR: uv is not on PATH." >&2; exit 1; }
uv sync --project gemma_runtime --reinstall

PYTHON="$REPO_ROOT/gemma_runtime/.venv/bin/python"
[[ -x "$PYTHON" ]] || { echo "ERROR: Gemma runtime Python was not created: $PYTHON" >&2; exit 1; }

mapfile -t NVIDIA_LIB_DIRS < <(find "$REPO_ROOT/gemma_runtime/.venv/lib" -type d -path '*/site-packages/nvidia/*/lib' -print | sort)
(( ${#NVIDIA_LIB_DIRS[@]} > 0 )) || { echo "ERROR: no wheel-bundled NVIDIA libraries found." >&2; exit 1; }

CUDNN_FOUND=0
for directory in "${NVIDIA_LIB_DIRS[@]}"; do
    [[ -f "$directory/libcudnn.so.9" ]] && CUDNN_FOUND=1
done
(( CUDNN_FOUND == 1 )) || { echo "ERROR: libcudnn.so.9 was not installed by uv sync." >&2; exit 1; }

NVIDIA_LIB_PATH="$(IFS=:; echo "${NVIDIA_LIB_DIRS[*]}")"
echo "[gemma-runtime] validating CUDA + cuDNN 9"
LD_LIBRARY_PATH="$NVIDIA_LIB_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$PYTHON" - <<'PY'
import torch
import torch.nn.functional as F
from packaging.version import Version

assert Version(torch.__version__.split("+")[0]) >= Version("2.6")
assert torch.cuda.is_available(), "CUDA is unavailable to Gemma runtime"
x = torch.randn((1, 1, 320), device="cuda", dtype=torch.float16)
w = torch.randn((4, 1, 5), device="cuda", dtype=torch.float16)
F.conv1d(x, w)
print(f"OK: torch={torch.__version__} cuda={torch.version.cuda} cudnn={torch.backends.cudnn.version()}")
PY
