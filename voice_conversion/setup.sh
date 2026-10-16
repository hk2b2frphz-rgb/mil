#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
command -v uv >/dev/null || { echo 'Install uv first.' >&2; exit 1; }
uv venv --python 3.9 .venv
PY="$PWD/.venv/bin/python"
uv pip install --python "$PY" 'pip<25' 'setuptools<81' wheel
# CUDA 11.8 wheels retain Volta (sm_70) support. Do not replace with latest CUDA wheels.
uv pip install --python "$PY" torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
uv pip install --python "$PY" -r requirements.txt
mkdir -p vendor
if [[ ! -d vendor/OpenVoice/.git ]]; then
    git clone https://github.com/myshell-ai/OpenVoice.git vendor/OpenVoice
fi
git -C vendor/OpenVoice rev-parse HEAD > openvoice-revision.txt
uv pip install --python "$PY" -e vendor/OpenVoice
"$PY" download_model.py
"$PY" convert.py --check
