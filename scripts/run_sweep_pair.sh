#!/usr/bin/env bash
# Generate one ~3h dataset or use SRC_RUN_DIR, then run two or more LoRA
# hyperparameter patterns on it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BASE_EXP="${BASE_EXP:-lora_base_config}"
NUM_CASES="${NUM_CASES:-250}"
SWEEP_PATTERNS="${SWEEP_PATTERNS:-h01 h02}"
# '+' as well as ',': a chained handoff passes the remaining patterns
# through qsub -v, which already uses commas to separate assignments.
SWEEP_PATTERNS="${SWEEP_PATTERNS//[,+]/ }"
NPROC="${NPROC:-1}"
RUN_ID="${RUN_ID:-run_$(date +%Y%m%d_%H%M%S)}"
INPUT_SRC_RUN_DIR="${SRC_RUN_DIR:-}"

# shellcheck source=/dev/null
source "$SCRIPT_DIR/sweep_chain.sh"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/train_timebox.sh"

export NPROC
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-job_lora_hp}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:///$REPO_ROOT/mlruns/mlflow.db}"
export MLFLOW_ARTIFACT_ROOT="${MLFLOW_ARTIFACT_ROOT:-file:$REPO_ROOT/mlruns/artifacts}"

if [[ -z "$INPUT_SRC_RUN_DIR" ]] && {
    ! [[ "$NUM_CASES" =~ ^[0-9]+$ ]] || [[ "$NUM_CASES" -lt 2 ]];
}; then
    echo "ERROR: NUM_CASES must be an integer >= 2 for train/eval split. got: $NUM_CASES" >&2
    exit 1
fi

# shellcheck source=/dev/null
source "$SCRIPT_DIR/sweep_patterns.sh"

echo "===== sweep ====="
echo "run_id:      $RUN_ID"
echo "base_exp:    $BASE_EXP"
echo "num_cases:   $NUM_CASES"
echo "src_run_dir: ${INPUT_SRC_RUN_DIR:-<generate new>}"
echo "patterns:    $SWEEP_PATTERNS"
echo "mlflow_exp:  $MLFLOW_EXPERIMENT_NAME"
echo "mlflow_uri:  $MLFLOW_TRACKING_URI"
echo "artifact:    $MLFLOW_ARTIFACT_ROOT"
echo "cuda:        $CUDA_VISIBLE_DEVICES"
echo "started_at:  $(date -Iseconds)"
echo "================="

command -v uv >/dev/null 2>&1 || {
    echo "ERROR: uv is not available on PATH." >&2
    exit 1
}

echo "[sweep] checking MLflow availability"
if command -v timeout >/dev/null 2>&1; then
    MLFLOW_CHECK_CMD=(timeout "${MLFLOW_IMPORT_TIMEOUT:-120}" uv run python -c "import mlflow")
else
    MLFLOW_CHECK_CMD=(uv run python -c "import mlflow")
fi
if ! "${MLFLOW_CHECK_CMD[@]}" >/dev/null 2>&1; then
    echo "[sweep] installing mlflow into project uv environment"
    if ! uv pip install mlflow; then
        echo "[sweep] WARN: failed to install mlflow (no network on this node?); continuing without live sync" >&2
    fi
fi

echo
if [[ -n "$INPUT_SRC_RUN_DIR" ]]; then
    echo "[sweep] using existing shared dataset"
    SRC_RUN_DIR="$INPUT_SRC_RUN_DIR"
else
    echo "[sweep] generating shared dataset"
    RUN_ID="$RUN_ID" \
    NUM_CASES="$NUM_CASES" \
    STEPS=use_cases,dialogues,audio \
    bash scripts/run_pipeline.sh

    SRC_RUN_DIR="./data/runs/${RUN_ID}"
fi

if [[ ! -f "$SRC_RUN_DIR/training_set/synthetic_moshi_train.jsonl" ]]; then
    echo "ERROR: training manifest not found: $SRC_RUN_DIR/training_set/synthetic_moshi_train.jsonl" >&2
    exit 1
fi

sweep_chain_init
timebox_init "$SWEEP_START_EPOCH"

for pattern in $SWEEP_PATTERNS; do
    if sweep_chain_completed "$pattern"; then
        echo "[sweep] $pattern already done in an earlier link; skipping"
        continue
    fi
    pattern_started="$(date +%s)"

    apply_pattern "$pattern"
    export MLFLOW_RUN_NAME="${RUN_ID}_${pattern}"
    SWEEP_EXP_NAME="_sweeps/${RUN_ID}_${pattern}"
    SWEEP_EXP_DIR="experiments/$SWEEP_EXP_NAME"
    mkdir -p "$SWEEP_EXP_DIR"
    cp "experiments/$BASE_EXP/config.yaml" "$SWEEP_EXP_DIR/config.yaml"
    if [[ -f "experiments/$BASE_EXP/HYPERPARAMS.md" ]]; then
        cp "experiments/$BASE_EXP/HYPERPARAMS.md" "$SWEEP_EXP_DIR/HYPERPARAMS.md"
    fi

    export REFRESH_EXP_DATA=1

    # Deterministic run dir, so a job that picks this pattern up later can find
    # the checkpoints the previous one wrote.
    export RUN_TS="${RUN_ID}_${pattern}"
    CKPT_ROOT="$SWEEP_EXP_DIR/checkpoints/$RUN_TS/checkpoints"

    unset HP_RESUME_FROM HP_RESUME_STEP
    if RESUME_INFO="$(timebox_latest_ckpt "$CKPT_ROOT")"; then
        export HP_RESUME_STEP="${RESUME_INFO%%$'	'*}"
        export HP_RESUME_FROM="${RESUME_INFO#*$'	'}"
        echo "[sweep] resuming $pattern from step $HP_RESUME_STEP"
    elif ! sweep_chain_have_time; then
        # Only worth deferring a pattern that has not started; one already part
        # way through makes progress every job and must not be deferred forever.
        sweep_chain_handoff
    fi

    echo
    echo "[sweep] running $pattern"
    echo "  exp=$SWEEP_EXP_NAME"
    echo "  lr=$HP_LR rank=$HP_LORA_RANK scaling=$HP_LORA_SCALING batch=$HP_BATCH_SIZE micro=$HP_NUM_MICROBATCHES steps=$HP_MAX_STEPS wd=$HP_WEIGHT_DECAY pct_start=$HP_PCT_START"
    if timebox_run "$TIMEBOX_DEADLINE" bash scripts/run_experiment.sh "$SWEEP_EXP_NAME" "$SRC_RUN_DIR"; then
        sweep_chain_record "$pattern" "$(( $(date +%s) - pattern_started ))"
    else
        rc=$?
        [[ "$rc" == "124" ]] || exit "$rc"
        echo "[sweep] $pattern ran out of time; the next job continues it"
        sweep_chain_handoff
    fi
done

echo
echo "finished_at: $(date -Iseconds)"
