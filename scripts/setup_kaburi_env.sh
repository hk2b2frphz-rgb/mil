#!/usr/bin/env bash
set -euo pipefail

# Clone KABURI-TTS next to this repo and build its uv environment.
#
# KABURI cannot live in the main project environment: it pins torch cu128 and
# python <3.12, while this repo is cu121 / python >=3.11. It gets its own uv
# project, the same arrangement as ../moshi-finetune, and the PBS jobs call it
# with `uv run --project "$KABURI_REPO"`.
#
# Usage:
#   bash scripts/setup_kaburi_env.sh
#   KABURI_REPO=/somewhere/kaburi-tts bash scripts/setup_kaburi_env.sh
#
# Behind a TLS-inspecting proxy, `uv sync` dies on the torch wheel with
#   invalid peer certificate: UnknownIssuer
# because uv's bundled certificate store has never heard of the proxy's CA.
# --native-tls (on by default here, see scripts/kaburi_uv_env.sh) switches uv
# to the machine's own store. If the CA is not there either, point uv at the
# file: export SSL_CERT_FILE=/path/to/corp-ca.pem
#
# If it still fails on download-r2.pytorch.org specifically -- the host
# download.pytorch.org redirects to -- the proxy is allowing one host and not
# the other, and no certificate setting will help. Take torch from PyPI
# instead, whose wheels bundle CUDA 12.8 anyway:
#
#   KABURI_TORCH_FROM_PYPI=1 bash scripts/setup_kaburi_env.sh
#
# and pass the same variable to the render jobs, since uv re-resolves on every
# `uv run` and would otherwise pull the pinned wheel back in.
#
# Model weights (acoustic model, codec, raster models; ~5GB) are downloaded
# from Hugging Face on first inference, not here. Behind a proxy, export
# PROXY_URL/HTTPS_PROXY before the first render job.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

KABURI_REPO="${KABURI_REPO:-$REPO_ROOT/../kaburi-tts}"
KABURI_URL="${KABURI_URL:-https://github.com/llm-jp/kaburi-tts.git}"

command -v uv >/dev/null 2>&1 || { echo "ERROR: uv is not on PATH." >&2; exit 1; }
command -v git >/dev/null 2>&1 || { echo "ERROR: git is not on PATH." >&2; exit 1; }

export KABURI_REPO
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/kaburi_uv_env.sh"
kaburi_uv_sync_args SYNC_ARGS
kaburi_uv_run KABURI_UV

if [[ -d "$KABURI_REPO/.git" ]]; then
    echo "[kaburi] existing checkout: $KABURI_REPO"
else
    echo "[kaburi] cloning $KABURI_URL -> $KABURI_REPO"
    git clone "$KABURI_URL" "$KABURI_REPO"
fi

echo "[kaburi] uv sync ${SYNC_ARGS[*]:-(no extra flags)}"
if ! uv sync --project "$KABURI_REPO" ${SYNC_ARGS[@]+"${SYNC_ARGS[@]}"}; then
    echo >&2
    echo "ERROR: uv sync failed. Two failures are expected on this cluster:" >&2
    echo "  invalid peer certificate: UnknownIssuer" >&2
    echo "    -> the proxy's CA is unknown to uv. Try SSL_CERT_FILE=<corp CA>." >&2
    echo "  a dead connection to download-r2.pytorch.org" >&2
    echo "    -> the pinned torch index is unreachable. Re-run with" >&2
    echo "       KABURI_TORCH_FROM_PYPI=1 and pass it to the render jobs too." >&2
    exit 1
fi

echo "[kaburi] import check"
"${KABURI_UV[@]}" - <<'PY'
import torch
import kaburi_tts  # noqa: F401
from irodori_tts.codec import DACVAECodec  # noqa: F401
from kaburi_tts.raster.pipeline import T
print(f"kaburi ok: torch={torch.__version__} canvas_frames={T}")
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    print(f"gpu: sm{major}{minor} (bfloat16 codec decode needs sm80+)")
else:
    print("gpu: none visible here (the PBS job checks this again)")
PY

# torchaudio 2.9+ reads and writes audio through torchcodec, and uv can resolve
# a torchcodec built for a different CUDA than torch. That breaks every wav
# access, but only on first file I/O, so it has to be probed rather than
# imported. The same script runs in the render job's preflight.
echo "[kaburi] audio I/O check"
bash scripts/fix_kaburi_audio_io.sh

echo
echo "[kaburi] done. Point jobs at it with:"
echo "  export KABURI_REPO=$KABURI_REPO"
if [[ "${KABURI_TORCH_FROM_PYPI:-0}" == "1" ]]; then
    echo "  export KABURI_TORCH_FROM_PYPI=1   # this env resolved torch from PyPI"
fi
