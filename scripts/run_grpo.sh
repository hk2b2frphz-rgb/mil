#!/usr/bin/env bash
# Launch GRPO interactivity alignment training for Moshi.
#
# Usage:
#   bash scripts/run_grpo.sh
#   CONFIG=configs/grpo_loneliness.yaml bash scripts/run_grpo.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-configs/grpo_loneliness.yaml}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

if [[ "${DISABLE_TRITON:-0}" == "1" ]]; then
    export TORCHINDUCTOR_DISABLE=1
    export NO_TORCH_COMPILE=1
    export TORCH_COMPILE_DISABLE=1
fi

command -v uv >/dev/null 2>&1 || {
    echo "ERROR: uv is not available on PATH." >&2
    exit 1
}

echo "===== GRPO interactivity alignment ====="
echo "config:  $CONFIG"
echo "cuda:    $CUDA_VISIBLE_DEVICES"
echo "started: $(date -Iseconds)"
echo "========================================="

uv run python scripts/grpo/train_grpo.py --config "$CONFIG"

echo "finished: $(date -Iseconds)"
