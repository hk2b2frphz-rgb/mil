#!/usr/bin/env bash
# The real-dialogue evaluation, defined once.
#
# Every system on the real track must be scored by the same code with the same
# thresholds, or the numbers cannot be compared across the batch table.  The
# GPT Realtime runner cannot go through PBS, so it is the one system whose
# inference happens elsewhere -- which is exactly why the scoring is kept here
# instead of being copied into its script, where it had already drifted (no
# --max-latency-sec, no REAL_BACKCHANNEL gate).
#
# Callers set REAL_OUT_DIR and MODEL_ID, then source this file.  REAL_PY_RUNNER
# is the interpreter prefix ("uv run python" on the server, a plain python on a
# local PC without uv).

: "${REAL_OUT_DIR:?real_eval_metrics.sh requires REAL_OUT_DIR}"
REAL_PY_RUNNER="${REAL_PY_RUNNER:-uv run python}"
REAL_MOS_BACKEND="${REAL_MOS_BACKEND:-utmos}"
REAL_MOS_DEVICE="${REAL_MOS_DEVICE:-cpu}"
REAL_MAX_LATENCY_SEC="${REAL_MAX_LATENCY_SEC:-}"

# 3) 評価。応答率・応答速度・音声品質。
eval_args=(
    $REAL_PY_RUNNER eval/evaluate_real_response.py
    --run-dir "$REAL_OUT_DIR/inference"
    --out-dir "$REAL_OUT_DIR/benchmark_results"
    --mos-backend "$REAL_MOS_BACKEND"
    --mos-device "$REAL_MOS_DEVICE"
)
if [[ -n "$REAL_MAX_LATENCY_SEC" ]]; then
    eval_args+=(--max-latency-sec "$REAL_MAX_LATENCY_SEC")
fi
"${eval_args[@]}"

# 3b) 相槌。User 発話中の別軸。応答評価とは指標の形が違うので別ファイルに出す。
#     REAL_BACKCHANNEL=0 で飛ばせる。
if [[ "${REAL_BACKCHANNEL:-1}" == "1" ]]; then
    $REAL_PY_RUNNER eval/evaluate_real_dialogue_backchannel.py \
        --run-dir "$REAL_OUT_DIR/inference" \
        --out "$REAL_OUT_DIR/benchmark_results/backchannel.json" \
        --tolerance-sec "${REAL_BC_TOLERANCE_SEC:-1.0}"
fi

# 4) LLM-as-a-judge 入力(モデルと相談員の両方を採点対象として出す)。
#    gold は自分自身が相談員なので、人間側の行は重複させない。
judge_args=(
    $REAL_PY_RUNNER eval/pack_real_dialogue_judge_input.py
    --per-case "$REAL_OUT_DIR/benchmark_results/per_case.jsonl"
    --out "$REAL_OUT_DIR/real_judge_input.jsonl"
)
if [[ "${MODEL_ID:-}" == "gold" ]]; then
    judge_args+=(--skip-human)
fi
"${judge_args[@]}"

echo "[real] summary:          $REAL_OUT_DIR/benchmark_results/summary.json"
echo "[real] backchannel:      $REAL_OUT_DIR/benchmark_results/backchannel.json"
echo "[real] per_case:         $REAL_OUT_DIR/benchmark_results/per_case.jsonl"
echo "[real] judge input only: $REAL_OUT_DIR/real_judge_input.jsonl"
