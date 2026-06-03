#!/usr/bin/env bash
# Generate a fresh synthetic dataset of N dialogues, then immediately launch the
# named experiment against the new run dir.
#
# Usage:
#   bash scripts/generate_and_run.sh <EXP_NAME> <NUM_CASES>
#
# Example (3h dataset → exp002):
#   bash scripts/generate_and_run.sh exp002_lora_3h_data 250
#
# Behavior:
#   1) run_pipeline.sh の STEPS=use_cases,dialogues,audio を NUM_CASES で実行し、
#      新しい data/runs/<日付>/ を生成する。
#   2) その新 run dir を引数にして scripts/run_experiment.sh <EXP_NAME> を起動する。
#
# Notes:
#   - データ生成と学習の合計時間は ~250 対話で数時間規模。長時間用に nohup や tmux で
#     起動するのを推奨。
#   - 既存の run dir を流用したい場合は、このスクリプトは使わず
#     scripts/run_experiment.sh <EXP> <SRC_RUN_DIR> を直接叩く。

if [[ -z "${BASH_VERSION:-}" ]]; then
    echo "ERROR: bash で実行してください" >&2
    exit 1
fi
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: bash $0 <EXP_NAME> <NUM_CASES>" >&2
    echo "  例: bash $0 exp002_lora_3h_data 250" >&2
    exit 1
fi
EXP_NAME="$1"
NUM_CASES="$2"

if ! [[ "$NUM_CASES" =~ ^[0-9]+$ ]] || [[ "$NUM_CASES" -lt 1 ]]; then
    echo "ERROR: NUM_CASES は正の整数で指定してください (got: $NUM_CASES)" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

EXP_DIR="$REPO_ROOT/experiments/$EXP_NAME"
if [[ ! -d "$EXP_DIR" ]]; then
    echo "ERROR: 実験フォルダがありません: $EXP_DIR" >&2
    echo "  先に experiments/<EXP_NAME>/{config.yaml,HYPERPARAMS.md} を用意してください" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 1) データ生成。run_pipeline.sh が新しい RUN_ID を切るので、こちらでも同じ ID を
#    再利用するために RUN_ID を先に作って渡す（後でその dir を確実に参照したい）。
# ---------------------------------------------------------------------------
RUN_ID="$(date +%Y-%m-%d_%H%M%S)"
RUNS_ROOT="${RUNS_ROOT:-$REPO_ROOT/data/runs}"
OUT_ROOT="$RUNS_ROOT/$RUN_ID"

echo "===================================================================="
echo "[gen_and_run] EXP_NAME : $EXP_NAME"
echo "[gen_and_run] NUM_CASES: $NUM_CASES"
echo "[gen_and_run] RUN_ID   : $RUN_ID"
echo "[gen_and_run] OUT_ROOT : $OUT_ROOT"
echo "===================================================================="

RUN_ID="$RUN_ID" \
    NUM_CASES="$NUM_CASES" \
    STEPS=use_cases,dialogues,audio \
    bash "$SCRIPT_DIR/run_pipeline.sh"

if [[ ! -f "$OUT_ROOT/training_set/synthetic_moshi_train.jsonl" ]]; then
    echo "ERROR: データ生成後に manifest が見つかりません: $OUT_ROOT/training_set/synthetic_moshi_train.jsonl" >&2
    exit 1
fi

echo
echo "===================================================================="
echo "[gen_and_run] データ生成完了。実験 $EXP_NAME を起動します。"
echo "[gen_and_run] source: $OUT_ROOT"
echo "===================================================================="
echo

# ---------------------------------------------------------------------------
# 2) 新しい run dir を引数に実験を起動。
# ---------------------------------------------------------------------------
bash "$SCRIPT_DIR/run_experiment.sh" "$EXP_NAME" "$OUT_ROOT"
