#!/usr/bin/env bash
set -euo pipefail

# Run the aizuchi TTS smokes on an interactive node, without qsub.
#
#   bash scripts/run_tts_smoke.sh              # KABURI, 3 dialogues
#   bash scripts/run_tts_smoke.sh kaburi 3
#   bash scripts/run_tts_smoke.sh qwen 3
#   bash scripts/run_tts_smoke.sh both 3       # both, then a side-by-side table
#
# Same bodies the PBS smokes use (scripts/2026-09-04/aizuchi_normal_*_smoke.pbs
# are these settings plus queue directives), so the timings carry over. What
# this adds is running in the foreground on one GPU, a wall-clock line per
# backend, and the comparison table at the end when both are rendered.
#
# Needs an allocated GPU. KABURI needs sm80+ (A100) because its codec decodes
# in bfloat16; the Qwen3 arm uses vLLM-Omni, which is patched for V100 here --
# so "both" only works where the node has a GPU each backend accepts, and
# otherwise run one arm per node and pass both directories to
# scripts/report_tts_smoke.py by hand afterwards.
#
# Prerequisites, both one-off:
#   bash scripts/setup_kaburi_env.sh          (KABURI arm)
#   bash scripts/setup_vllm_omni_v100_env.sh  (Qwen3 arm)
#
# Overrides: NUM_DIALOGUES, CUDA_VISIBLE_DEVICES, AIZUCHI_VERSION,
# DIALOGUES_JSONL, CLONE_OUT_DIR_MOSHI, REPORT_PROJECTION.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/run_id_utils.sh"

BACKEND="${1:-kaburi}"
case "$BACKEND" in
    kaburi|qwen|both) ;;
    *)
        echo "usage: bash scripts/run_tts_smoke.sh [kaburi|qwen|both] [NUM_DIALOGUES]" >&2
        exit 1
        ;;
esac

NUM_DIALOGUES="${2:-${NUM_DIALOGUES:-3}}"
if ! [[ "$NUM_DIALOGUES" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: NUM_DIALOGUES must be a positive integer: $NUM_DIALOGUES" >&2
    exit 1
fi

AIZUCHI_VERSION="${AIZUCHI_VERSION:-v6}"
SOURCE_BATCH_ID="qwen_aizuchi_3000_normal_${AIZUCHI_VERSION}"
DIALOGUES_JSONL="${DIALOGUES_JSONL:-$REPO_ROOT/data/runs/aizuchi_normal_3000_${AIZUCHI_VERSION}/dialogue/llm_dialogues/dialogues.jsonl}"
CLONE_OUT_DIR_MOSHI="${CLONE_OUT_DIR_MOSHI:-$REPO_ROOT/data/clone_examples/99999}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
REPORT_PROJECTION="${REPORT_PROJECTION:-3000}"
STAMP="$(run_id_stamp)"

if [[ ! -s "$DIALOGUES_JSONL" ]]; then
    echo "ERROR: dialogues JSONL was not found or is empty: $DIALOGUES_JSONL" >&2
    echo "Generate it first with scripts/2026-09-04/aizuchi_normal_dialogue_3000.pbs" >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "WARNING: nvidia-smi is missing; is this an interactive node with a GPU?" >&2
fi

echo "===== TTS smoke ($BACKEND) ====="
echo "repo:          $REPO_ROOT"
echo "dialogues:     $DIALOGUES_JSONL"
echo "num_dialogues: $NUM_DIALOGUES"
echo "gpu:           $CUDA_VISIBLE_DEVICES"
echo "moshi clone:   $CLONE_OUT_DIR_MOSHI"
echo "started_at:    $(date -Iseconds)"
echo "================================"

KABURI_TRAINING_DIR=""
QWEN_TRAINING_DIR=""

run_kaburi() {
    local batch_id="aizuchi_normal_3000_${AIZUCHI_VERSION}_kaburi_smoke_${STAMP}"
    local out_root="$REPO_ROOT/data/runs/smoke/$batch_id"
    KABURI_TRAINING_DIR="$out_root/shard_000/training_set"
    echo
    echo ">>> KABURI-TTS -> $out_root"
    local started elapsed
    started="$(date +%s)"
    (
        export CLONE_OUT_DIR_MOSHI CUDA_VISIBLE_DEVICES SOURCE_BATCH_ID DIALOGUES_JSONL
        export BATCH_ID="$batch_id"
        export OUT_ROOT="$out_root"
        export NUM_DIALOGUES NUM_SHARDS=1 SPARE_RATIO=0 RESUME=0 LOG_EVERY=1
        export REPORT_PROJECTION
        export SMOKE_REPORT_LABEL="${SMOKE_REPORT_LABEL_KABURI:-kaburi}"
        bash scripts/run_kaburi_tts.pbs
    )
    elapsed="$(( $(date +%s) - started ))"
    echo "<<< KABURI-TTS finished in ${elapsed}s wall (model load included)"
}

run_qwen() {
    local batch_id="aizuchi_normal_3000_${AIZUCHI_VERSION}_qwen_smoke_${STAMP}"
    local out_root="$REPO_ROOT/data/runs/smoke/$batch_id"
    QWEN_TRAINING_DIR="$out_root/shard_000/training_set"
    echo
    echo ">>> Qwen3-TTS -> $out_root"
    local started elapsed
    started="$(date +%s)"
    (
        export CLONE_OUT_DIR_MOSHI CUDA_VISIBLE_DEVICES SOURCE_BATCH_ID DIALOGUES_JSONL
        export BATCH_ID="$batch_id"
        export OUT_ROOT="$out_root"
        export NUM_DIALOGUES NUM_SHARDS=1 SPARE_RATIO=0 RESUME=0 LOG_EVERY=1
        bash scripts/run_qwen_tts_vllm_3000_4gpu.pbs
    )
    elapsed="$(( $(date +%s) - started ))"
    echo "<<< Qwen3-TTS finished in ${elapsed}s wall (model load included)"
    echo
    uv run python scripts/report_tts_smoke.py \
        --training-dir "$QWEN_TRAINING_DIR" \
        --label "${SMOKE_REPORT_LABEL_QWEN:-qwen3}" \
        --projection "$REPORT_PROJECTION" \
        --json-out "$out_root/smoke_report.json"
}

case "$BACKEND" in
    kaburi) run_kaburi ;;
    qwen)   run_qwen ;;
    both)
        run_kaburi
        run_qwen
        echo
        echo "===== side by side ====="
        uv run python scripts/report_tts_smoke.py \
            --training-dir "$KABURI_TRAINING_DIR" --label kaburi \
            --training-dir "$QWEN_TRAINING_DIR" --label qwen3 \
            --projection "$REPORT_PROJECTION" \
            --json-out "$REPO_ROOT/data/runs/smoke/compare_${STAMP}.json"
        ;;
esac

echo
echo "finished_at: $(date -Iseconds)"
