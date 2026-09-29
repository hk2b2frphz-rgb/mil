#!/usr/bin/env bash
set -euo pipefail

# V100 32GB target: native audio-in/audio-out Qwen2.5-Omni-3B evaluation.
# No network API is called; this script is safe to invoke from the PBS batch.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/setup_proxy.sh"
MODEL_ID="${MODEL_ID:?Set MODEL_ID (e.g. qwen25_omni_3b)}"
RUN_ID="${RUN_ID:-${MODEL_ID}_$(date +%Y%m%d_%H%M%S)}"
TEST_DATA_DIR="${TEST_DATA_DIR:-$REPO_ROOT/data/test_data/real_dialogue}"
REAL_DATASET_DIR="${REAL_DATASET_DIR:-$REPO_ROOT/data/eval_sets/real_response}"
REAL_OUT_DIR="${REAL_OUT_DIR:-$REPO_ROOT/eval_runs/real_response/$RUN_ID}"
QWEN25_OMNI_MODEL="${QWEN25_OMNI_MODEL:-Qwen/Qwen2.5-Omni-3B}"
QWEN25_OMNI_PYTHON="${QWEN25_OMNI_PYTHON:-}"
QWEN25_OMNI_DEVICE_MAP="${QWEN25_OMNI_DEVICE_MAP:-auto}"
# V100 has no practical bf16 path; 3B needs fp16 and leaves room for audio/KV.
QWEN25_OMNI_DTYPE="${QWEN25_OMNI_DTYPE:-float16}"
QWEN25_OMNI_SPEAKER="${QWEN25_OMNI_SPEAKER:-Chelsie}"
QWEN25_OMNI_MAX_NEW_TOKENS="${QWEN25_OMNI_MAX_NEW_TOKENS:-200}"
QWEN25_OMNI_TIMEOUT_SEC="${QWEN25_OMNI_TIMEOUT_SEC:-600}"
mkdir -p "$REAL_OUT_DIR"
exec > >(tee "$REAL_OUT_DIR/run.log") 2>&1

if [[ ! -f "$REAL_DATASET_DIR/manifest.json" || "${REBUILD_REAL_DATASET:-0}" == "1" ]]; then
    build_args=(uv run python eval/build_real_test_dataset.py --test-data-dir "$TEST_DATA_DIR" --out-dir "$REAL_DATASET_DIR" --context-sec "${REAL_CONTEXT_SEC:-0}" --lead-in-sec "${REAL_LEAD_IN_SEC:-3.0}")
    [[ "${REBUILD_REAL_DATASET:-0}" == "1" ]] && build_args+=(--overwrite)
    "${build_args[@]}"
fi
run_args=(uv run python eval/run_local_baseline_full_duplex.py --system qwen25_omni --dataset-dir "$REAL_DATASET_DIR" --out-dir "$REAL_OUT_DIR/inference" --model-id "$MODEL_ID" --tasks all --seeds "${REAL_SEEDS:-0}" --overwrite --qwen25-omni-model "$QWEN25_OMNI_MODEL" --qwen25-omni-device-map "$QWEN25_OMNI_DEVICE_MAP" --qwen25-omni-dtype "$QWEN25_OMNI_DTYPE" --qwen25-omni-speaker "$QWEN25_OMNI_SPEAKER" --qwen25-omni-max-new-tokens "$QWEN25_OMNI_MAX_NEW_TOKENS" --qwen25-omni-timeout-sec "$QWEN25_OMNI_TIMEOUT_SEC")
[[ -n "$QWEN25_OMNI_PYTHON" ]] && run_args+=(--qwen25-omni-python "$QWEN25_OMNI_PYTHON")
[[ -n "${REAL_CASES_PER_TASK:-}" ]] && run_args+=(--cases-per-task "$REAL_CASES_PER_TASK")
"${run_args[@]}"
eval_args=(uv run python eval/evaluate_real_response.py --run-dir "$REAL_OUT_DIR/inference" --out-dir "$REAL_OUT_DIR/benchmark_results" --mos-backend "${REAL_MOS_BACKEND:-utmos}" --mos-device "${REAL_MOS_DEVICE:-cpu}")
[[ -n "${REAL_MAX_LATENCY_SEC:-}" ]] && eval_args+=(--max-latency-sec "$REAL_MAX_LATENCY_SEC")
"${eval_args[@]}"
uv run python eval/evaluate_real_dialogue_backchannel.py --run-dir "$REAL_OUT_DIR/inference" --out "$REAL_OUT_DIR/benchmark_results/backchannel.json" --tolerance-sec "${REAL_BC_TOLERANCE_SEC:-1.0}"
uv run python eval/pack_real_dialogue_judge_input.py --per-case "$REAL_OUT_DIR/benchmark_results/per_case.jsonl" --out "$REAL_OUT_DIR/real_judge_input.jsonl"
echo "[qwen25-omni] summary: $REAL_OUT_DIR/benchmark_results/summary.json"
