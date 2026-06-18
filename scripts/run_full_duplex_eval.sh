#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODEL_ID="${MODEL_ID:?Set MODEL_ID to a stable label such as base, lora_h01, or full_f01}"
if ! [[ "$MODEL_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: MODEL_ID may contain only letters, digits, dot, underscore, and hyphen." >&2
    exit 1
fi
RUN_ID="${RUN_ID:-${MODEL_ID}_$(date +%Y%m%d_%H%M%S)}"
HF_REPO="${HF_REPO:-llm-jp/llm-jp-moshi-v1}"
MODEL_WEIGHT="${MODEL_WEIGHT:-${MOSHI_WEIGHT:-}}"
MODEL_CONFIG="${MODEL_CONFIG:-}"
MIMI_WEIGHT="${MIMI_WEIGHT:-}"
TOKENIZER="${TOKENIZER:-}"
FDB_SCENARIOS="${FDB_SCENARIOS:-$REPO_ROOT/eval_sets/full_duplex_ja/scenarios.jsonl}"
FDB_DATA_DIR="${FDB_DATA_DIR:-$REPO_ROOT/data/full_duplex_ja}"
FDB_OUT_DIR="${FDB_OUT_DIR:-$REPO_ROOT/eval_runs/full_duplex/$RUN_ID}"
FDB_TASKS="${FDB_TASKS:-all}"
FDB_SEEDS="${FDB_SEEDS:-0}"
FDB_TAIL_SEC="${FDB_TAIL_SEC:-8}"
FDB_TTS_SPEED="${FDB_TTS_SPEED:-1.1}"
REFRESH_FDB_DATA="${REFRESH_FDB_DATA:-0}"
HALF="${HALF:-1}"

command -v uv >/dev/null 2>&1 || {
    echo "ERROR: uv is not available on PATH." >&2
    exit 1
}

if [[ -n "$MODEL_WEIGHT" && ! -f "$MODEL_WEIGHT" ]]; then
    echo "ERROR: MODEL_WEIGHT does not exist: $MODEL_WEIGHT" >&2
    exit 1
fi
if [[ -n "$MODEL_CONFIG" && ! -f "$MODEL_CONFIG" ]]; then
    echo "ERROR: MODEL_CONFIG does not exist: $MODEL_CONFIG" >&2
    exit 1
fi

mkdir -p "$FDB_OUT_DIR"
exec > >(tee -a "$FDB_OUT_DIR/run.log") 2>&1

echo "===== Full-Duplex-Bench-JA ====="
echo "run_id:       $RUN_ID"
echo "model_id:     $MODEL_ID"
echo "hf_repo:      $HF_REPO"
echo "model_weight: ${MODEL_WEIGHT:-<hf default>}"
echo "model_config: ${MODEL_CONFIG:-<hf default>}"
echo "tasks:        $FDB_TASKS"
echo "seeds:        $FDB_SEEDS"
echo "data:         $FDB_DATA_DIR"
echo "output:       $FDB_OUT_DIR"
echo "dtype:        $([[ "$HALF" == "1" ]] && echo float16 || echo bfloat16)"
echo "upstream:     Full-Duplex-Bench@3e799c45a045256f47d5f1c9cda90157e2d2ec9e"
echo "started_at:   $(date -Iseconds)"
echo "================================"

if [[ "$REFRESH_FDB_DATA" == "1" || ! -f "$FDB_DATA_DIR/manifest.json" ]]; then
    build_args=(
        uv run python eval/build_full_duplex_ja_dataset.py
        --scenarios "$FDB_SCENARIOS"
        --out-dir "$FDB_DATA_DIR"
        --tts-speed "$FDB_TTS_SPEED"
        --overwrite
    )
    "${build_args[@]}"
else
    echo "[fdb] reusing Japanese dataset: $FDB_DATA_DIR"
fi

inference_args=(
    uv run python eval/run_full_duplex_bench.py
    --dataset-dir "$FDB_DATA_DIR"
    --out-dir "$FDB_OUT_DIR/inference"
    --model-id "$MODEL_ID"
    --hf-repo "$HF_REPO"
    --tasks "$FDB_TASKS"
    --seeds "$FDB_SEEDS"
    --tail-sec "$FDB_TAIL_SEC"
    --overwrite
)
if [[ "$HALF" == "1" ]]; then inference_args+=(--half); fi
if [[ -n "$MODEL_WEIGHT" ]]; then inference_args+=(--moshi-weight "$MODEL_WEIGHT"); fi
if [[ -n "$MODEL_CONFIG" ]]; then inference_args+=(--config "$MODEL_CONFIG"); fi
if [[ -n "$MIMI_WEIGHT" ]]; then inference_args+=(--mimi-weight "$MIMI_WEIGHT"); fi
if [[ -n "$TOKENIZER" ]]; then inference_args+=(--tokenizer "$TOKENIZER"); fi
if [[ -n "${TEMP:-}" ]]; then inference_args+=(--temp "$TEMP"); fi
if [[ -n "${TEMP_TEXT:-}" ]]; then inference_args+=(--temp-text "$TEMP_TEXT"); fi
if [[ -n "${CFG_COEF:-}" ]]; then inference_args+=(--cfg-coef "$CFG_COEF"); fi

"${inference_args[@]}"

uv run python eval/evaluate_full_duplex_ja.py \
    --run-dir "$FDB_OUT_DIR/inference" \
    --out-dir "$FDB_OUT_DIR/benchmark_results"

uv run python eval/pack_full_duplex_azure.py \
    --per-case "$FDB_OUT_DIR/benchmark_results/per_case.jsonl" \
    --out "$FDB_OUT_DIR/azure_judge_input.jsonl"

echo "[fdb] benchmark summary: $FDB_OUT_DIR/benchmark_results/summary.json"
echo "[fdb] Azure input only:  $FDB_OUT_DIR/azure_judge_input.jsonl"
echo "[fdb] No OpenAI/Azure API was called by this server run."
echo "finished_at: $(date -Iseconds)"
