#!/usr/bin/env bash
set -euo pipefail

# LOCAL PC ONLY.  This script intentionally has no .pbs companion and refuses
# scheduler environments because GPT Realtime needs outbound WebSocket access.
#
# It uses the same API environment as the judge.  For the in-house
# OpenAI-compatible gateway:
#
#   export OPENAI_API_KEY='...'
#   export OPENAI_BASE_URL='https://api.rdg-genai.crl.hitachi.co.jp/v1'
#   MODEL_ID=gpt_realtime bash scripts/run_real_gpt_realtime_eval.sh
#
# For an Azure resource, set AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT and
# AZURE_OPENAI_REALTIME_DEPLOYMENT instead.  GPT_REALTIME_PROVIDER defaults to
# auto, which follows whichever of the two is configured.

if [[ -n "${PBS_JOBID:-}${SLURM_JOB_ID:-}${LSB_JOBID:-}" ]]; then
    echo "ERROR: GPT Realtime evaluation is local-only; do not submit this script through PBS." >&2
    exit 2
fi
GPT_REALTIME_PROVIDER="${GPT_REALTIME_PROVIDER:-auto}"
# The evaluator re-checks all of this; these guards only fail before the dataset
# build, which is slow enough to be worth not running on a bad configuration.
if [[ "$GPT_REALTIME_PROVIDER" == "azure" ]]; then
    if [[ -z "${AZURE_OPENAI_KEY:-}" || -z "${AZURE_OPENAI_ENDPOINT:-}" ]]; then
        echo "ERROR: set AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT in the local shell (same variables as the Azure judge)." >&2
        exit 2
    fi
    if [[ -z "${AZURE_OPENAI_REALTIME_DEPLOYMENT:-}${AZURE_OPENAI_DEPLOYMENT:-}${GPT_REALTIME_MODEL:-}" ]]; then
        echo "ERROR: set AZURE_OPENAI_REALTIME_DEPLOYMENT to the realtime deployment name." >&2
        exit 2
    fi
elif [[ "$GPT_REALTIME_PROVIDER" == "openai" || -n "${OPENAI_BASE_URL:-}${OPENAI_API_BASE:-}" ]]; then
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
        echo "ERROR: set OPENAI_API_KEY for ${OPENAI_BASE_URL:-${OPENAI_API_BASE:-api.openai.com}}." >&2
        exit 2
    fi
elif [[ -z "${AZURE_OPENAI_ENDPOINT:-}" ]]; then
    echo "ERROR: set OPENAI_BASE_URL + OPENAI_API_KEY (OpenAI-compatible gateway) or AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_KEY (Azure)." >&2
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
# REAL_BATCH_DIR points at a batch produced by scripts/run_full_duplex_eval_batch.sh.
# Writing into its <output_name>/ layout is what lets this locally-run system
# appear in the same combined_summary.json as the systems evaluated on the HPC,
# instead of in a table of its own that cannot be compared.
OUTPUT_NAME="${OUTPUT_NAME:-$MODEL_ID}"
if [[ -n "${REAL_BATCH_DIR:-}" ]]; then
    if ! [[ "$OUTPUT_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "ERROR: OUTPUT_NAME must match [A-Za-z0-9._-]+ to sit in a batch directory: $OUTPUT_NAME" >&2
        exit 2
    fi
    REAL_OUT_DIR="${REAL_OUT_DIR:-$REAL_BATCH_DIR/$OUTPUT_NAME}"
fi
REAL_OUT_DIR="${REAL_OUT_DIR:-$REPO_ROOT/eval_runs/real_response/$RUN_ID}"

"$LOCAL_PYTHON" -c "import websocket" 2>/dev/null || { echo "ERROR: install websocket-client locally: $LOCAL_PYTHON -m pip install websocket-client" >&2; exit 2; }
mkdir -p "$REAL_OUT_DIR"
if [[ ! -f "$REAL_DATASET_DIR/manifest.json" || "${REBUILD_REAL_DATASET:-0}" == "1" ]]; then
    build_args=("$LOCAL_PYTHON" eval/build_real_test_dataset.py --test-data-dir "$TEST_DATA_DIR" --out-dir "$REAL_DATASET_DIR" --context-sec "${REAL_CONTEXT_SEC:-0}" --lead-in-sec "${REAL_LEAD_IN_SEC:-3.0}")
    [[ "${REBUILD_REAL_DATASET:-0}" == "1" ]] && build_args+=(--overwrite)
    "${build_args[@]}"
fi
run_args=("$LOCAL_PYTHON" eval/run_gpt_realtime_eval.py --dataset-dir "$REAL_DATASET_DIR" --out-dir "$REAL_OUT_DIR/inference" --model-id "$MODEL_ID" --provider "$GPT_REALTIME_PROVIDER" --voice "${GPT_REALTIME_VOICE:-marin}" --seeds "${REAL_SEEDS:-0}" --tasks "${REAL_TASKS:-all}" --input-mode "${GPT_REALTIME_INPUT_MODE:-realtime}" --turn-detection "${GPT_REALTIME_TURN_DETECTION:-server_vad}" --chunk-ms "${GPT_REALTIME_CHUNK_MS:-100}" --response-timeout-sec "${GPT_REALTIME_TIMEOUT_SEC:-90}" --max-output-tokens "${GPT_REALTIME_MAX_OUTPUT_TOKENS:-200}" --realtime-schema "${OPENAI_REALTIME_SCHEMA:-ga}" --connect-retries "${GPT_REALTIME_CONNECT_RETRIES:-4}" --case-retries "${GPT_REALTIME_CASE_RETRIES:-3}")
# REAL_RESUME=1 keeps cases that already have their .meta.json, so a run that
# died partway is continued with RUN_ID set to the same directory instead of
# paying for every completed case again.
if [[ "${REAL_RESUME:-0}" != "1" ]]; then
    run_args+=(--overwrite)
fi
[[ -n "${GPT_REALTIME_MODEL:-}" ]] && run_args+=(--model "$GPT_REALTIME_MODEL")
[[ -n "${REAL_CASES_PER_TASK:-}" ]] && run_args+=(--cases-per-task "$REAL_CASES_PER_TASK")
"${run_args[@]}"
# Scoring is the batch's, not this script's: same evaluators, same thresholds,
# same judge-input packing, so the row is readable next to the HPC systems.
REAL_START_SEC="${REAL_START_SEC:-$SECONDS}"
REAL_PY_RUNNER="$LOCAL_PYTHON"
source "$REPO_ROOT/scripts/real_eval_metrics.sh"

if [[ -n "${REAL_BATCH_DIR:-}" ]]; then
    # Replace any earlier row for this output_name rather than appending a
    # second one; combine_real_summaries.py keeps the last row it reads, but a
    # file that grows on every rerun is hard to read by hand.
    status_file="$REAL_BATCH_DIR/batch_status.jsonl"
    elapsed="$(( SECONDS - REAL_START_SEC ))"
    if [[ -f "$status_file" ]]; then
        grep -v "\"output_name\":\"$OUTPUT_NAME\"" "$status_file" > "$status_file.tmp" || true
        mv "$status_file.tmp" "$status_file"
    fi
    printf '{"model_id":"%s","output_name":"%s","status":"%s","elapsed_sec":"%s"}
'         "$MODEL_ID" "$OUTPUT_NAME" "ok" "$elapsed" >> "$status_file"
    echo "[gpt-realtime] batch row: $status_file ($OUTPUT_NAME)"
    echo "[gpt-realtime] rebuild the combined table with:"
    echo "  uv run python eval/combine_real_summaries.py --batch-dir $REAL_BATCH_DIR --status-file $status_file --out $REAL_BATCH_DIR/combined_summary.json"
fi
echo "[gpt-realtime] summary: $REAL_OUT_DIR/benchmark_results/summary.json"
