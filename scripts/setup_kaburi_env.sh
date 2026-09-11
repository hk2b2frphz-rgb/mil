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

if [[ -d "$KABURI_REPO/.git" ]]; then
    echo "[kaburi] existing checkout: $KABURI_REPO"
else
    echo "[kaburi] cloning $KABURI_URL -> $KABURI_REPO"
    git clone "$KABURI_URL" "$KABURI_REPO"
fi

echo "[kaburi] uv sync"
uv sync --project "$KABURI_REPO"

echo "[kaburi] import check"
uv run --project "$KABURI_REPO" --with pyopenjtalk python - <<'PY'
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

echo "[kaburi] done. Point jobs at it with KABURI_REPO=$KABURI_REPO"
