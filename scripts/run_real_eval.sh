#!/usr/bin/env bash
set -euo pipefail

# 実データ応答評価の 1 モデル分ドライバ。バッチ(run_full_duplex_eval_batch.sh)
# から 1 行につき 1 回呼ばれる。単体でも動く。
#
# 入力は人手アノテーション由来のテストデータ。合成 TTS は一切作らない。
# OpenAI/Azure API も呼ばない(judge 入力を書き出すだけ)。
#
#   MODEL_ID=lora_h01 MODEL_WEIGHT=/path/model.safetensors \
#     bash scripts/run_real_eval.sh
#
# MODEL_ID=gold のときはモデルを動かさず、実際の相談員の応答を出力として置く。
# 指標の実質的な上限として、他のモデルと同じ表に並ぶ。
#
#   MODEL_ID=gold bash scripts/run_real_eval.sh

trap '
    status=$?
    echo "[real] ERROR: command failed (exit $status): $BASH_COMMAND" >&2
    case "$status" in
        137) echo "[real]   exit 137 = SIGKILL -- GPU/host OOM か walltime 超過。" >&2 ;;
        139) echo "[real]   exit 139 = SIGSEGV -- CUDA ドライバ/カーネルのクラッシュ。" >&2 ;;
    esac
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=memory.used,memory.total --format=csv >&2
' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

source "$SCRIPT_DIR/setup_proxy.sh"

MODEL_ID="${MODEL_ID:?Set MODEL_ID (e.g. base, gold, lora_h01)}"
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

TEST_DATA_DIR="${TEST_DATA_DIR:-$REPO_ROOT/data/test_data/real_dialogue}"
REAL_DATASET_DIR="${REAL_DATASET_DIR:-$REPO_ROOT/data/eval_sets/real_response}"
REAL_OUT_DIR="${REAL_OUT_DIR:-$REPO_ROOT/eval_runs/real_response/$RUN_ID}"
REAL_SEEDS="${REAL_SEEDS:-0}"
REAL_TAIL_SEC="${REAL_TAIL_SEC:-8}"
REAL_CONTEXT_SEC="${REAL_CONTEXT_SEC:-0}"
REAL_CASES_PER_TASK="${REAL_CASES_PER_TASK:-}"
REAL_MOS_BACKEND="${REAL_MOS_BACKEND:-utmos}"
REAL_MOS_DEVICE="${REAL_MOS_DEVICE:-cpu}"
REAL_MAX_LATENCY_SEC="${REAL_MAX_LATENCY_SEC:-}"
REBUILD_REAL_DATASET="${REBUILD_REAL_DATASET:-0}"
HALF="${HALF:-1}"

export TORCHINDUCTOR_DISABLE=1
export NO_TORCH_COMPILE="${NO_TORCH_COMPILE:-1}"
export TORCH_COMPILE_DISABLE=1
export PYTHONUNBUFFERED=1
export REAL_GIT_COMMIT="${REAL_GIT_COMMIT:-$(git rev-parse --short HEAD 2>/dev/null || echo unknown)}"

command -v uv >/dev/null 2>&1 || {
    echo "ERROR: uv is not available on PATH." >&2
    exit 1
}

# full-FT のエクスポートは model.safetensors と moshi_lm_kwargs.json を同じ
# ディレクトリに出すので、MODEL_CONFIG 未指定なら隣を見て拾う。
if [[ -z "$MODEL_CONFIG" && -n "$MODEL_WEIGHT" ]]; then
    AUTO_MODEL_CONFIG="$(dirname "$MODEL_WEIGHT")/moshi_lm_kwargs.json"
    if [[ -f "$AUTO_MODEL_CONFIG" ]]; then
        MODEL_CONFIG="$AUTO_MODEL_CONFIG"
        echo "[real] auto-detected MODEL_CONFIG: $MODEL_CONFIG"
    fi
fi
if [[ -n "$MODEL_WEIGHT" && ! -f "$MODEL_WEIGHT" ]]; then
    echo "ERROR: MODEL_WEIGHT does not exist: $MODEL_WEIGHT" >&2
    exit 1
fi

mkdir -p "$REAL_OUT_DIR"
exec > >(tee "$REAL_OUT_DIR/run.log") 2>&1

echo "===== Real dialogue response evaluation ====="
echo "run_id:       $RUN_ID"
echo "model_id:     $MODEL_ID"
echo "model_weight: ${MODEL_WEIGHT:-<hf default>}"
echo "test_data:    $TEST_DATA_DIR"
echo "dataset:      $REAL_DATASET_DIR"
echo "output:       $REAL_OUT_DIR"
echo "mos_backend:  $REAL_MOS_BACKEND"
echo "git_commit:   $REAL_GIT_COMMIT"
echo "started_at:   $(date -Iseconds)"
echo "============================================="

# 1) データセット。全モデルで共有するので、無ければ作る。作り直しは明示指定時
#    のみ(バッチ実行中に他のモデルが読んでいる最中の作り直しを避ける)。
if [[ ! -f "$REAL_DATASET_DIR/manifest.json" || "$REBUILD_REAL_DATASET" == "1" ]]; then
    build_args=(
        uv run python eval/build_real_test_dataset.py
        --test-data-dir "$TEST_DATA_DIR"
        --out-dir "$REAL_DATASET_DIR"
        --context-sec "$REAL_CONTEXT_SEC"
    )
    [[ "$REBUILD_REAL_DATASET" == "1" ]] && build_args+=(--overwrite)
    "${build_args[@]}"
else
    echo "[real] 既存のデータセットを使います: $REAL_DATASET_DIR"
    echo "[real]   作り直すなら REBUILD_REAL_DATASET=1"
fi

# 2) 推論。gold はモデルを動かさず実際の相談員の応答を置く。
if [[ "$MODEL_ID" == "gold" ]]; then
    echo "[real] MODEL_ID=gold: モデルは動かしません。実データを参照します。"
    gold_args=(
        uv run python eval/run_gold_reference.py
        --dataset-dir "$REAL_DATASET_DIR"
        --out-dir "$REAL_OUT_DIR/inference"
        --model-id "$MODEL_ID"
        --seeds "$REAL_SEEDS"
        --tail-sec "$REAL_TAIL_SEC"
        --overwrite
    )
    "${gold_args[@]}"
else
    inference_args=(
        uv run python eval/run_full_duplex_bench.py
        --dataset-dir "$REAL_DATASET_DIR"
        --out-dir "$REAL_OUT_DIR/inference"
        --model-id "$MODEL_ID"
        --hf-repo "$HF_REPO"
        --tasks all
        --seeds "$REAL_SEEDS"
        --tail-sec "$REAL_TAIL_SEC"
        --overwrite
    )
    if [[ -n "$REAL_CASES_PER_TASK" ]]; then
        inference_args+=(--cases-per-task "$REAL_CASES_PER_TASK")
    fi
    if [[ "$HALF" == "1" ]]; then inference_args+=(--half); fi
    if [[ -n "$MODEL_WEIGHT" ]]; then inference_args+=(--moshi-weight "$MODEL_WEIGHT"); fi
    if [[ -n "$MODEL_CONFIG" ]]; then inference_args+=(--config "$MODEL_CONFIG"); fi
    if [[ -n "$MIMI_WEIGHT" ]]; then inference_args+=(--mimi-weight "$MIMI_WEIGHT"); fi
    if [[ -n "$TOKENIZER" ]]; then inference_args+=(--tokenizer "$TOKENIZER"); fi
    "${inference_args[@]}"
fi

# 3) 評価。応答率・応答速度・音声品質。
eval_args=(
    uv run python eval/evaluate_real_response.py
    --run-dir "$REAL_OUT_DIR/inference"
    --out-dir "$REAL_OUT_DIR/benchmark_results"
    --mos-backend "$REAL_MOS_BACKEND"
    --mos-device "$REAL_MOS_DEVICE"
)
if [[ -n "$REAL_MAX_LATENCY_SEC" ]]; then
    eval_args+=(--max-latency-sec "$REAL_MAX_LATENCY_SEC")
fi
"${eval_args[@]}"

# 4) LLM-as-a-judge 入力(モデルと相談員の両方を採点対象として出す)。
#    gold は自分自身が相談員なので、人間側の行は重複させない。
judge_args=(
    uv run python eval/pack_real_dialogue_judge_input.py
    --per-case "$REAL_OUT_DIR/benchmark_results/per_case.jsonl"
    --out "$REAL_OUT_DIR/real_judge_input.jsonl"
)
if [[ "$MODEL_ID" == "gold" ]]; then
    judge_args+=(--skip-human)
fi
"${judge_args[@]}"

echo "[real] summary:          $REAL_OUT_DIR/benchmark_results/summary.json"
echo "[real] per_case:         $REAL_OUT_DIR/benchmark_results/per_case.jsonl"
echo "[real] judge input only: $REAL_OUT_DIR/real_judge_input.jsonl"
echo "[real] No OpenAI/Azure API was called by this server run."
echo "finished_at: $(date -Iseconds)"
