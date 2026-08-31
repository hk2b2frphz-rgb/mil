#!/usr/bin/env bash
set -euo pipefail

# Real-dialogue evaluation for an audio-in/text-out SpeechLLM baseline.
# This is an HPC/PBS-safe sibling of run_real_cascade_eval.sh: it calls no
# OpenAI API and writes only deterministic scores plus local judge input.
#
#   MODEL_ID=speechllm_qwen2audio bash scripts/run_real_speechllm_eval.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/setup_proxy.sh"
source "$SCRIPT_DIR/kokoro_uv_env.sh"

MODEL_ID="${MODEL_ID:?Set MODEL_ID (e.g. speechllm_qwen2audio)}"
if ! [[ "$MODEL_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: MODEL_ID may contain only letters, digits, dot, underscore, and hyphen." >&2
    exit 1
fi
RUN_ID="${RUN_ID:-${MODEL_ID}_$(date +%Y%m%d_%H%M%S)}"
TEST_DATA_DIR="${TEST_DATA_DIR:-$REPO_ROOT/data/test_data/real_dialogue}"
REAL_DATASET_DIR="${REAL_DATASET_DIR:-$REPO_ROOT/data/eval_sets/real_response}"
REAL_OUT_DIR="${REAL_OUT_DIR:-$REPO_ROOT/eval_runs/real_response/$RUN_ID}"
REAL_SEEDS="${REAL_SEEDS:-0}"
REAL_CONTEXT_SEC="${REAL_CONTEXT_SEC:-0}"
REAL_CASES_PER_TASK="${REAL_CASES_PER_TASK:-}"
REAL_MOS_BACKEND="${REAL_MOS_BACKEND:-utmos}"
REAL_MOS_DEVICE="${REAL_MOS_DEVICE:-cpu}"
REAL_MAX_LATENCY_SEC="${REAL_MAX_LATENCY_SEC:-}"
REBUILD_REAL_DATASET="${REBUILD_REAL_DATASET:-0}"

SPEECHLLM_MODEL="${SPEECHLLM_MODEL:-Qwen/Qwen2-Audio-7B-Instruct}"
SPEECHLLM_PYTHON="${SPEECHLLM_PYTHON:-}"
SPEECHLLM_DEVICE_MAP="${SPEECHLLM_DEVICE_MAP:-auto}"
SPEECHLLM_DTYPE="${SPEECHLLM_DTYPE:-auto}"
SPEECHLLM_MAX_NEW_TOKENS="${SPEECHLLM_MAX_NEW_TOKENS:-200}"
SPEECHLLM_TIMEOUT_SEC="${SPEECHLLM_TIMEOUT_SEC:-300}"
SPEECHLLM_TTS_BACKEND="${SPEECHLLM_TTS_BACKEND:-kokoro}"
SPEECHLLM_TTS_MODEL="${SPEECHLLM_TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}"
if [[ "$SPEECHLLM_TTS_BACKEND" == "kokoro" ]]; then
    SPEECHLLM_TTS_SPEAKER="${SPEECHLLM_TTS_SPEAKER:-jf_alpha}"
else
    SPEECHLLM_TTS_SPEAKER="${SPEECHLLM_TTS_SPEAKER:-Serena}"
fi
SPEECHLLM_DEVICE="${SPEECHLLM_DEVICE:-cuda}"
SPEECHLLM_TTS_DTYPE="${SPEECHLLM_TTS_DTYPE:-float16}"

export TORCHINDUCTOR_DISABLE=1 NO_TORCH_COMPILE="${NO_TORCH_COMPILE:-1}" TORCH_COMPILE_DISABLE=1
export PYTHONUNBUFFERED=1 REAL_GIT_COMMIT="${REAL_GIT_COMMIT:-$(git rev-parse --short HEAD 2>/dev/null || echo unknown)}"
command -v uv >/dev/null 2>&1 || { echo "ERROR: uv is not available on PATH." >&2; exit 1; }

mkdir -p "$REAL_OUT_DIR"
exec > >(tee "$REAL_OUT_DIR/run.log") 2>&1
echo "===== Real dialogue response evaluation (SpeechLLM) ====="
echo "run_id:       $RUN_ID"
echo "model_id:     $MODEL_ID"
echo "speechllm:    $SPEECHLLM_MODEL"
echo "output:       $REAL_OUT_DIR"
echo "started_at:   $(date -Iseconds)"

if [[ ! -f "$REAL_DATASET_DIR/manifest.json" || "$REBUILD_REAL_DATASET" == "1" ]]; then
    build_args=(uv run python eval/build_real_test_dataset.py --test-data-dir "$TEST_DATA_DIR" --out-dir "$REAL_DATASET_DIR" --context-sec "$REAL_CONTEXT_SEC" --lead-in-sec "${REAL_LEAD_IN_SEC:-3.0}")
    [[ "$REBUILD_REAL_DATASET" == "1" ]] && build_args+=(--overwrite)
    "${build_args[@]}"
fi

kokoro_uv_run UV_RUN "$SPEECHLLM_TTS_BACKEND" real-speechllm
inference_args=(
    "${UV_RUN[@]}" python eval/run_local_baseline_full_duplex.py
    --system speechllm --dataset-dir "$REAL_DATASET_DIR" --out-dir "$REAL_OUT_DIR/inference"
    --model-id "$MODEL_ID" --tasks all --seeds "$REAL_SEEDS" --overwrite
    --speechllm-model "$SPEECHLLM_MODEL" --speechllm-device-map "$SPEECHLLM_DEVICE_MAP"
    --speechllm-dtype "$SPEECHLLM_DTYPE" --speechllm-max-new-tokens "$SPEECHLLM_MAX_NEW_TOKENS"
    --speechllm-timeout-sec "$SPEECHLLM_TIMEOUT_SEC"
    --tts-backend "$SPEECHLLM_TTS_BACKEND" --tts-model "$SPEECHLLM_TTS_MODEL"
    --tts-speaker "$SPEECHLLM_TTS_SPEAKER" --device "$SPEECHLLM_DEVICE" --dtype "$SPEECHLLM_TTS_DTYPE"
)
[[ -n "$SPEECHLLM_PYTHON" ]] && inference_args+=(--speechllm-python "$SPEECHLLM_PYTHON")
[[ -n "$REAL_CASES_PER_TASK" ]] && inference_args+=(--cases-per-task "$REAL_CASES_PER_TASK")
"${inference_args[@]}"

eval_args=(uv run python eval/evaluate_real_response.py --run-dir "$REAL_OUT_DIR/inference" --out-dir "$REAL_OUT_DIR/benchmark_results" --mos-backend "$REAL_MOS_BACKEND" --mos-device "$REAL_MOS_DEVICE")
[[ -n "$REAL_MAX_LATENCY_SEC" ]] && eval_args+=(--max-latency-sec "$REAL_MAX_LATENCY_SEC")
"${eval_args[@]}"
if [[ "${REAL_BACKCHANNEL:-1}" == "1" ]]; then
    uv run python eval/evaluate_real_dialogue_backchannel.py --run-dir "$REAL_OUT_DIR/inference" --out "$REAL_OUT_DIR/benchmark_results/backchannel.json" --tolerance-sec "${REAL_BC_TOLERANCE_SEC:-1.0}"
fi
uv run python eval/pack_real_dialogue_judge_input.py --per-case "$REAL_OUT_DIR/benchmark_results/per_case.jsonl" --out "$REAL_OUT_DIR/real_judge_input.jsonl"
echo "[real-speechllm] summary: $REAL_OUT_DIR/benchmark_results/summary.json"
echo "[real-speechllm] No OpenAI/Azure API was called by this server run."
