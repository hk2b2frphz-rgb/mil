#!/usr/bin/env bash
set -euo pipefail

# Make torchaudio able to read and write audio in the KABURI environment,
# repairing it if it cannot. A no-op when the environment is already healthy,
# so it is safe to call from anywhere -- setup_kaburi_env.sh runs it after
# uv sync, and run_kaburi_tts.pbs runs it in its preflight, which is why a
# broken codec no longer means going back to run another script by hand.
#
#   bash scripts/fix_kaburi_audio_io.sh
#   KABURI_AUTO_REPAIR=0 bash scripts/fix_kaburi_audio_io.sh   # probe only
#
# The problem it exists for: torchaudio 2.9+ implements load and save through
# torchcodec, whose wheels are built per CUDA major version. uv resolves
# torchcodec from PyPI, which can hand back a build for a different CUDA than
# torch, and then every wav access dies with
#
#   OSError: libnvrtc.so.13: cannot open shared object file
#
# not at import but on first file I/O -- which is the reference-pack build,
# long after the GPU and model checks have passed.
#
# Repairs are tried in order, re-probing after each:
#   1. torchcodec from the index matching this torch's CUDA
#   2. the CPU torchcodec -- decoding a wav needs no GPU, and that wheel links
#      none of the missing CUDA runtime. It gives up GPU media decoding, which
#      nothing in this pipeline uses.
#   3. the nvrtc runtime the installed torchcodec asks for
#
# Repairs use uv pip against the project venv, so they can disagree with the
# lockfile; kaburi_uv_env.sh passes --no-sync to `uv run` to keep uv from
# reverting them.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export KABURI_REPO="${KABURI_REPO:-$REPO_ROOT/../kaburi-tts}"
if [[ ! -d "$KABURI_REPO/kaburi_tts" ]]; then
    echo "ERROR: KABURI-TTS checkout not found: $KABURI_REPO" >&2
    exit 1
fi

# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/kaburi_uv_env.sh"
kaburi_uv_run KABURI_UV

VENV_PYTHON="$KABURI_REPO/.venv/bin/python"

# uv pip must cross the same proxy as uv sync did.
PIP_TLS=()
if [[ "${KABURI_NATIVE_TLS:-1}" == "1" ]]; then
    PIP_TLS=("$(kaburi_tls_flag)")
fi

audio_io_ok() {
    # Re-export first: a repair may have just created the nvidia/*/lib
    # directory the loader needs on LD_LIBRARY_PATH, and the path was built
    # before that existed.
    kaburi_export_ldpath
    "${KABURI_UV[@]}" scripts/kaburi_audio_io_check.py
}

install_torchcodec_from() {  # install_torchcodec_from <index-url> [spec]
    uv pip install --python "$VENV_PYTHON" ${PIP_TLS[@]+"${PIP_TLS[@]}"} \
        --index-url "$1" --upgrade "${2:-torchcodec}"
}

if audio_io_ok; then
    exit 0
fi

if [[ "${KABURI_AUTO_REPAIR:-1}" != "1" ]]; then
    echo "ERROR: audio I/O is broken and KABURI_AUTO_REPAIR=0." >&2
    exit 1
fi

echo "[kaburi] torchaudio cannot read audio; repairing torchcodec"
CUDA_TAG="$("${KABURI_UV[@]}" -c \
    'import torch; v=torch.version.cuda or ""; print("cu"+v.replace(".",""))')"
if [[ -z "$CUDA_TAG" || "$CUDA_TAG" == "cu" ]]; then
    CUDA_TAG="cpu"
fi

echo "[kaburi] repair 1/3: torchcodec from the $CUDA_TAG index"
install_torchcodec_from "https://download.pytorch.org/whl/$CUDA_TAG" || true

if ! audio_io_ok; then
    echo "[kaburi] repair 2/3: CPU-only torchcodec"
    install_torchcodec_from "https://download.pytorch.org/whl/cpu" || true
fi

if ! audio_io_ok; then
    NVRTC_MAJOR="$("${KABURI_UV[@]}" - <<'PY' || true
import re
try:
    from torchcodec._core import _metadata  # noqa: F401
except Exception as exc:
    match = re.search(r"libnvrtc\.so\.(\d+)", str(exc))
    print(match.group(1) if match else "")
else:
    print("")
PY
)"
    NVRTC_MAJOR="$(printf '%s' "$NVRTC_MAJOR" | tr -dc '0-9')"
    NVRTC_MAJOR="${NVRTC_MAJOR:-13}"
    echo "[kaburi] repair 3/3: nvidia-cuda-nvrtc-cu$NVRTC_MAJOR"
    uv pip install --python "$VENV_PYTHON" ${PIP_TLS[@]+"${PIP_TLS[@]}"} \
        "nvidia-cuda-nvrtc-cu${NVRTC_MAJOR}" || true
fi

if ! audio_io_ok; then
    echo >&2
    echo "ERROR: torchaudio still cannot read audio in $KABURI_REPO/.venv" >&2
    echo "Three repairs were tried: torchcodec from $CUDA_TAG, the CPU" >&2
    echo "torchcodec, and the nvrtc runtime it asks for. From here, by hand:" >&2
    echo "  $VENV_PYTHON -m pip index versions torchcodec   # what is available" >&2
    echo "  uv pip install --python $VENV_PYTHON \\" >&2
    echo "    --index-url https://download.pytorch.org/whl/cpu 'torchcodec<0.10'" >&2
    echo "Verify with: bash scripts/run_kaburi_audio_check.sh" >&2
    exit 1
fi

echo "[kaburi] audio I/O repaired"
"${KABURI_UV[@]}" -c 'import torchcodec; print("torchcodec:", torchcodec.__version__)'
