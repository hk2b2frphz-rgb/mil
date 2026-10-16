#!/usr/bin/env bash
set -euo pipefail

# Run the KABURI audio I/O probe by hand, with the same interpreter, PYTHONPATH
# and LD_LIBRARY_PATH the render jobs use.
#
#   bash scripts/run_kaburi_audio_check.sh
#
# Exists because "can this environment read a wav" is the question that comes
# up after every torchcodec fiddle, and answering it otherwise means retyping
# the uv invocation with the right flags.
#
# A failure here is what scripts/setup_kaburi_env.sh repairs; re-run that.

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

echo "python:         ${KABURI_UV[*]}"
echo "PYTHONPATH:     ${PYTHONPATH:-(unset)}"
echo "LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-(unset)}"
echo
exec "${KABURI_UV[@]}" scripts/kaburi_audio_io_check.py
