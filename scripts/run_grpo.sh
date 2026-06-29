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

DS_CONFIG="${DS_CONFIG:-configs/deepspeed_grpo_zero3.json}"
NUM_PROC="${NUM_PROC:-2}"
SEGMENT_DIR="${SEGMENT_DIR:-}"

if [[ -z "$SEGMENT_DIR" ]]; then
    for candidate in \
        data/runs/qwen_dialogues_1000/grpo_segments \
        data/runs/qwen_dialogues_10000/grpo_segments; do
        if [[ -d "$candidate" ]]; then
            SEGMENT_DIR="$candidate"
            break
        fi
    done
fi
if [[ -z "$SEGMENT_DIR" ]]; then
    echo "ERROR: no segment_dir found. Run segment extraction first." >&2
    exit 1
fi
export GRPO_SEGMENT_DIR="$SEGMENT_DIR"

echo "===== GRPO interactivity alignment ====="
echo "config:      $CONFIG"
echo "deepspeed:   $DS_CONFIG"
echo "num_proc:    $NUM_PROC"
echo "segment_dir: $SEGMENT_DIR"
echo "cuda:        $CUDA_VISIBLE_DEVICES"
echo "started:     $(date -Iseconds)"
echo "========================================="

uv run accelerate launch \
    --num_processes "$NUM_PROC" \
    --use_deepspeed \
    --deepspeed_config_file "$DS_CONFIG" \
    scripts/grpo/train_grpo.py --config "$CONFIG"

echo "finished: $(date -Iseconds)"
