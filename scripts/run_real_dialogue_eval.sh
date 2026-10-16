#!/usr/bin/env bash
set -euo pipefail

# 実対話リプレイ評価の実行ドライバ。
#
# scripts/run_full_duplex_eval.sh の派生だが、入力 TTS の構築が無い(ユーザー側は
# 実音声そのもの)点と、v1.5 静的重畳プロトコルを名乗らない点が違う。合成
# Full-Duplex-Bench-JA 側は一切触らないので、両者は独立して回せる。
#
# 前提: 各対話ディレクトリで人手の作業が済んでいること。
#   1) scripts/analyze_real_dialogue.py   (ダイアライゼーション)
#   2) scripts/separate_real_dialogue.py  (音源分離 + 再書き起こし)
#   3) counselor_A.txt か counselor_B.txt を置く
#   4) <話者>_segments_review.tsv の keep 列で間引く
#
# 使い方:
#   MODEL_ID=lora_h01 MODEL_WEIGHT=/path/model.safetensors \
#     ANALYSIS_DIRS="data/real_dialogue/pilot/対話1 data/real_dialogue/pilot/対話2" \
#     bash scripts/run_real_dialogue_eval.sh

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

MODEL_ID="${MODEL_ID:?Set MODEL_ID to a stable label such as base, lora_h01, or full_f01}"
if ! [[ "$MODEL_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: MODEL_ID may contain only letters, digits, dot, underscore, and hyphen." >&2
    exit 1
fi
ANALYSIS_DIRS="${ANALYSIS_DIRS:?Set ANALYSIS_DIRS to one or more separated dialogue dirs}"
RUN_ID="${RUN_ID:-${MODEL_ID}_$(date +%Y%m%d_%H%M%S)}"
HF_REPO="${HF_REPO:-llm-jp/llm-jp-moshi-v1}"
MODEL_WEIGHT="${MODEL_WEIGHT:-${MOSHI_WEIGHT:-}}"
MODEL_CONFIG="${MODEL_CONFIG:-}"
MIMI_WEIGHT="${MIMI_WEIGHT:-}"
TOKENIZER="${TOKENIZER:-}"

REAL_DATA_DIR="${REAL_DATA_DIR:-$REPO_ROOT/data/real_dialogue_eval/$RUN_ID}"
REAL_OUT_DIR="${REAL_OUT_DIR:-$REPO_ROOT/eval_runs/real_dialogue/$RUN_ID}"
REAL_CONTEXT_SEC="${REAL_CONTEXT_SEC:-30}"
REAL_TASKS="${REAL_TASKS:-all}"
REAL_CASES_PER_TASK="${REAL_CASES_PER_TASK:-}"
REAL_SEEDS="${REAL_SEEDS:-0}"
REAL_TAIL_SEC="${REAL_TAIL_SEC:-30}"
REAL_BC_TOLERANCE_SEC="${REAL_BC_TOLERANCE_SEC:-1.5}"
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

echo "===== Real dialogue replay evaluation ====="
echo "run_id:       $RUN_ID"
echo "model_id:     $MODEL_ID"
echo "model_weight: ${MODEL_WEIGHT:-<hf default>}"
echo "analysis:     $ANALYSIS_DIRS"
echo "context_sec:  $REAL_CONTEXT_SEC"
echo "dataset:      $REAL_DATA_DIR"
echo "output:       $REAL_OUT_DIR"
echo "protocol:     open-loop replay (NOT Full-Duplex-Bench v1.5)"
echo "git_commit:   $REAL_GIT_COMMIT"
echo "started_at:   $(date -Iseconds)"
echo "==========================================="

# 1) 実データからデータセットを構築
build_args=(
    uv run python eval/build_real_dialogue_dataset.py
    --out-dir "$REAL_DATA_DIR"
    --context-sec "$REAL_CONTEXT_SEC"
    --overwrite
    --analysis-dir
)
# shellcheck disable=SC2206
analysis_array=($ANALYSIS_DIRS)
build_args+=("${analysis_array[@]}")
"${build_args[@]}"

# 2) 推論。--require-v15-profile は付けない(実対話は静的重畳プロトコルを
#    満たさないため。ここで付けると manifest の protocol 名で弾かれる)。
inference_args=(
    uv run python eval/run_full_duplex_bench.py
    --dataset-dir "$REAL_DATA_DIR"
    --out-dir "$REAL_OUT_DIR/inference"
    --model-id "$MODEL_ID"
    --hf-repo "$HF_REPO"
    --tasks "$REAL_TASKS"
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

uv run python eval/full_duplex_official_timing.py --run-dir "$REAL_OUT_DIR/inference"

# 3) 決定的評価。--require-overlap は付けない。開ループなので重畳が必ず起きる
#    保証が無く、起きたかどうかは評価器が overlap_achieved で記録する。
evaluation_status=0
if uv run python eval/evaluate_full_duplex_ja.py \
    --run-dir "$REAL_OUT_DIR/inference" \
    --out-dir "$REAL_OUT_DIR/benchmark_results"; then
    :
else
    evaluation_status=$?
    echo "[real] WARNING: deterministic evaluation was partial (exit $evaluation_status)." >&2
fi

# 4) 相槌一致度(実際の相談員の相槌が正解)
uv run python eval/evaluate_real_backchannel.py \
    --per-case "$REAL_OUT_DIR/benchmark_results/per_case.jsonl" \
    --out "$REAL_OUT_DIR/benchmark_results/backchannel_agreement.json" \
    --tolerance-sec "$REAL_BC_TOLERANCE_SEC"

# 5) LLM-as-a-judge 入力(モデルと相談員の両方を採点対象として出す)
uv run python eval/pack_real_dialogue_judge_input.py \
    --per-case "$REAL_OUT_DIR/benchmark_results/per_case.jsonl" \
    --out "$REAL_OUT_DIR/real_judge_input.jsonl"

echo "[real] summary:            $REAL_OUT_DIR/benchmark_results/summary.json"
echo "[real] backchannel:        $REAL_OUT_DIR/benchmark_results/backchannel_agreement.json"
echo "[real] judge input only:   $REAL_OUT_DIR/real_judge_input.jsonl"
echo "[real] No OpenAI/Azure API was called by this server run."
echo "finished_at: $(date -Iseconds)"
exit "$evaluation_status"
