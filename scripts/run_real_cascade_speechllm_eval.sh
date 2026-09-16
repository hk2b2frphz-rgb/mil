#!/usr/bin/env bash
set -euo pipefail

# Interactive-node runner: evaluates the cascade (ASR->LLM->TTS) and SpeechLLM
# (audio->text->TTS) baselines back to back on the real hand-annotated
# dialogue test set, using the same metrics/table as the Moshi rows.
#
# Run this inside an already allocated interactive GPU session (not via qsub):
#
#   bash scripts/run_real_cascade_speechllm_eval.sh
#
# Each system gets its own RUN_ID stamped with today's date, so both land
# under eval_runs/real_response/<model_id>_<YYYYMMDD>/. Set SKIP_CASCADE=1 or
# SKIP_SPEECHLLM=1 to run only one side. CASCADE_* / SPEECHLLM_* / REAL_* env
# vars are forwarded unchanged to scripts/run_real_cascade_eval.sh and
# scripts/run_real_speechllm_eval.sh -- see eval/README.md.
#
# TTS defaults to qwen3, not kokoro (kokoro's voice reads as machine-y).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

command -v nvidia-smi >/dev/null 2>&1 || {
    echo "ERROR: run this in a GPU allocation; nvidia-smi is unavailable." >&2
    exit 2
}

RUN_DATE="$(date +%Y%m%d)"
CASCADE_MODEL_ID="${CASCADE_MODEL_ID:-cascade1_$RUN_DATE}"
SPEECHLLM_MODEL_ID="${SPEECHLLM_MODEL_ID:-speechllm1_$RUN_DATE}"

CASCADE_TTS_BACKEND="${CASCADE_TTS_BACKEND:-qwen3}"
SPEECHLLM_TTS_BACKEND="${SPEECHLLM_TTS_BACKEND:-qwen3}"

echo "===== cascade + speechllm real-dialogue evaluation ====="
echo "date:              $RUN_DATE"
echo "cascade model_id:   $CASCADE_MODEL_ID (tts=$CASCADE_TTS_BACKEND)"
echo "speechllm model_id: $SPEECHLLM_MODEL_ID (tts=$SPEECHLLM_TTS_BACKEND)"
echo "=========================================================="

if [[ "${SKIP_CASCADE:-0}" != "1" ]]; then
    echo "[run_real_cascade_speechllm_eval] cascade..."
    MODEL_ID="$CASCADE_MODEL_ID" CASCADE_TTS_BACKEND="$CASCADE_TTS_BACKEND" \
        bash scripts/run_real_cascade_eval.sh
else
    echo "[run_real_cascade_speechllm_eval] SKIP_CASCADE=1: skipping cascade"
fi

if [[ "${SKIP_SPEECHLLM:-0}" != "1" ]]; then
    echo "[run_real_cascade_speechllm_eval] speechllm..."
    MODEL_ID="$SPEECHLLM_MODEL_ID" SPEECHLLM_TTS_BACKEND="$SPEECHLLM_TTS_BACKEND" \
        bash scripts/run_real_speechllm_eval.sh
else
    echo "[run_real_cascade_speechllm_eval] SKIP_SPEECHLLM=1: skipping speechllm"
fi

echo "[run_real_cascade_speechllm_eval] done."
[[ "${SKIP_CASCADE:-0}" != "1" ]] && echo "cascade summary:    eval_runs/real_response/$CASCADE_MODEL_ID/benchmark_results/summary.json"
[[ "${SKIP_SPEECHLLM:-0}" != "1" ]] && echo "speechllm summary:  eval_runs/real_response/$SPEECHLLM_MODEL_ID/benchmark_results/summary.json"
