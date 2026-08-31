#!/usr/bin/env bash
set -euo pipefail

# Synthetic Full-Duplex-Bench-JA track for the audio-in/text-out SpeechLLM.
# This is intentionally separate from the real-data batch track.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/setup_proxy.sh"
source "$SCRIPT_DIR/kokoro_uv_env.sh"
MODEL_ID="${MODEL_ID:?Set MODEL_ID (e.g. speechllm_qwen2audio)}"
RUN_ID="${RUN_ID:-${MODEL_ID}_$(date +%Y%m%d_%H%M%S)}"
FDB_DATA_DIR="${FDB_DATA_DIR:-$REPO_ROOT/data/full_duplex_ja_v2_nogreeting}"
FDB_OUT_DIR="${FDB_OUT_DIR:-$REPO_ROOT/eval_runs/full_duplex/$RUN_ID}"
FDB_SCENARIOS="${FDB_SCENARIOS:-$REPO_ROOT/eval_sets/full_duplex_ja/scenarios_expanded.jsonl}"
FDB_TTS_BACKEND="${FDB_TTS_BACKEND:-qwen3}"
FDB_TTS_MODEL="${FDB_TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}"
FDB_TTS_SPEAKER="${FDB_TTS_SPEAKER:-Ono_Anna}"
FDB_TTS_BACKGROUND_SPEAKER="${FDB_TTS_BACKGROUND_SPEAKER:-Uncle_Fu}"
SPEECHLLM_MODEL="${SPEECHLLM_MODEL:-Qwen/Qwen2-Audio-7B-Instruct}"
SPEECHLLM_TTS_BACKEND="${SPEECHLLM_TTS_BACKEND:-qwen3}"
SPEECHLLM_TTS_MODEL="${SPEECHLLM_TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}"
SPEECHLLM_TTS_SPEAKER="${SPEECHLLM_TTS_SPEAKER:-Serena}"
mkdir -p "$FDB_OUT_DIR"
kokoro_uv_run FDB_UV_RUN "$FDB_TTS_BACKEND" speechllm-fdb
kokoro_uv_run UV_RUN "$SPEECHLLM_TTS_BACKEND" speechllm-fdb
if [[ ! -f "$FDB_DATA_DIR/manifest.json" || "${REFRESH_FDB_DATA:-0}" == "1" ]]; then
    build_args=("${FDB_UV_RUN[@]}" python eval/build_full_duplex_ja_dataset.py --scenarios "$FDB_SCENARIOS" --out-dir "$FDB_DATA_DIR" --tts-backend "$FDB_TTS_BACKEND" --tts-model "$FDB_TTS_MODEL" --tts-speaker "$FDB_TTS_SPEAKER" --tts-background-speaker "$FDB_TTS_BACKGROUND_SPEAKER" --device "${FDB_DEVICE:-cuda}" --dtype "${FDB_DTYPE:-float16}" --no-opening-greeting --overwrite)
    "${build_args[@]}"
fi
run_args=("${UV_RUN[@]}" python eval/run_local_baseline_full_duplex.py --system speechllm --dataset-dir "$FDB_DATA_DIR" --out-dir "$FDB_OUT_DIR/inference" --model-id "$MODEL_ID" --tasks "${FDB_TASKS:-all}" --seeds "${FDB_SEEDS:-0}" --require-v15-profile --speechllm-model "$SPEECHLLM_MODEL" --speechllm-device-map "${SPEECHLLM_DEVICE_MAP:-auto}" --speechllm-dtype "${SPEECHLLM_DTYPE:-auto}" --speechllm-max-new-tokens "${SPEECHLLM_MAX_NEW_TOKENS:-200}" --speechllm-timeout-sec "${SPEECHLLM_TIMEOUT_SEC:-300}" --tts-backend "$SPEECHLLM_TTS_BACKEND" --tts-model "$SPEECHLLM_TTS_MODEL" --tts-speaker "$SPEECHLLM_TTS_SPEAKER" --device "${SPEECHLLM_DEVICE:-cuda}" --dtype "${SPEECHLLM_TTS_DTYPE:-float16}" --overwrite)
[[ -n "${SPEECHLLM_PYTHON:-}" ]] && run_args+=(--speechllm-python "$SPEECHLLM_PYTHON")
[[ -n "${FDB_CASES_PER_TASK:-}" ]] && run_args+=(--cases-per-task "$FDB_CASES_PER_TASK")
"${run_args[@]}"
uv run python eval/full_duplex_official_timing.py --run-dir "$FDB_OUT_DIR/inference"
eval_args=(uv run python eval/evaluate_full_duplex_ja.py --run-dir "$FDB_OUT_DIR/inference" --out-dir "$FDB_OUT_DIR/benchmark_results")
[[ -f "${FDB_BACKCHANNEL_GT:-$REPO_ROOT/eval_sets/full_duplex_ja/backchannel_gt.json}" ]] && eval_args+=(--backchannel-gt "${FDB_BACKCHANNEL_GT:-$REPO_ROOT/eval_sets/full_duplex_ja/backchannel_gt.json}")
"${eval_args[@]}"
uv run python eval/pack_full_duplex_azure.py --per-case "$FDB_OUT_DIR/benchmark_results/per_case.jsonl" --out "$FDB_OUT_DIR/azure_judge_input.jsonl"
echo "[speechllm-fdb] No OpenAI/Azure API was called by this server run."
