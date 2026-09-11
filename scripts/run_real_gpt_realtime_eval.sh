#!/usr/bin/env bash
set -euo pipefail

# LOCAL PC ONLY.  This script intentionally has no .pbs companion and refuses
# scheduler environments because GPT Realtime needs outbound WebSocket access.
#
#   export OPENAI_API_KEY='...'
#   MODEL_ID=gpt_realtime bash scripts/run_real_gpt_realtime_eval.sh

if [[ -n "${PBS_JOBID:-}${SLURM_JOB_ID:-}${LSB_JOBID:-}" ]]; then
    echo "ERROR: GPT Realtime evaluation is local-only; do not submit this script through PBS." >&2
    exit 2
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: set OPENAI_API_KEY in the local shell before running this script." >&2
    exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
MODEL_ID="${MODEL_ID:-gpt_realtime}"
RUN_ID="${RUN_ID:-${MODEL_ID}_$(date +%Y%m%d_%H%M%S)}"
LOCAL_PYTHON="${LOCAL_PYTHON:-python}"
TEST_DATA_DIR="${TEST_DATA_DIR:-$REPO_ROOT/data/test_data/real_dialogue}"
REAL_DATASET_DIR="${REAL_DATASET_DIR:-$REPO_ROOT/data/eval_sets/real_response}"
REAL_OUT_DIR="${REAL_OUT_DIR:-$REPO_ROOT/eval_runs/real_response/$RUN_ID}"

"$LOCAL_PYTHON" -c "import websocket" 2>/dev/null || { echo "ERROR: install websocket-client locally: $LOCAL_PYTHON -m pip install websocket-client" >&2; exit 2; }
mkdir -p "$REAL_OUT_DIR"
if [[ ! -f "$REAL_DATASET_DIR/manifest.json" || "${REBUILD_REAL_DATASET:-0}" == "1" ]]; then
    build_args=("$LOCAL_PYTHON" eval/build_real_test_dataset.py --test-data-dir "$TEST_DATA_DIR" --out-dir "$REAL_DATASET_DIR" --context-sec "${REAL_CONTEXT_SEC:-0}" --lead-in-sec "${REAL_LEAD_IN_SEC:-3.0}")
    [[ "${REBUILD_REAL_DATASET:-0}" == "1" ]] && build_args+=(--overwrite)
    "${build_args[@]}"
fi
run_args=("$LOCAL_PYTHON" eval/run_gpt_realtime_eval.py --dataset-dir "$REAL_DATASET_DIR" --out-dir "$REAL_OUT_DIR/inference" --model-id "$MODEL_ID" --model "${GPT_REALTIME_MODEL:-gpt-realtime}" --voice "${GPT_REALTIME_VOICE:-marin}" --seeds "${REAL_SEEDS:-0}" --tasks "${REAL_TASKS:-all}" --input-mode "${GPT_REALTIME_INPUT_MODE:-realtime}" --turn-detection "${GPT_REALTIME_TURN_DETECTION:-server_vad}" --chunk-ms "${GPT_REALTIME_CHUNK_MS:-100}" --response-timeout-sec "${GPT_REALTIME_TIMEOUT_SEC:-90}" --max-output-tokens "${GPT_REALTIME_MAX_OUTPUT_TOKENS:-200}" --overwrite)
[[ -n "${REAL_CASES_PER_TASK:-}" ]] && run_args+=(--cases-per-task "$REAL_CASES_PER_TASK")
"${run_args[@]}"
"$LOCAL_PYTHON" eval/evaluate_real_response.py --run-dir "$REAL_OUT_DIR/inference" --out-dir "$REAL_OUT_DIR/benchmark_results" --mos-backend "${REAL_MOS_BACKEND:-utmos}" --mos-device "${REAL_MOS_DEVICE:-cpu}"
"$LOCAL_PYTHON" eval/evaluate_real_dialogue_backchannel.py --run-dir "$REAL_OUT_DIR/inference" --out "$REAL_OUT_DIR/benchmark_results/backchannel.json" --tolerance-sec "${REAL_BC_TOLERANCE_SEC:-1.0}"
"$LOCAL_PYTHON" eval/pack_real_dialogue_judge_input.py --per-case "$REAL_OUT_DIR/benchmark_results/per_case.jsonl" --out "$REAL_OUT_DIR/real_judge_input.jsonl"
echo "[gpt-realtime] summary: $REAL_OUT_DIR/benchmark_results/summary.json"
