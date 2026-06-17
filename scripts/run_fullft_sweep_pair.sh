#!/usr/bin/env bash
# Generate one ~3h dataset or use SRC_RUN_DIR, then run two or more full
# fine-tuning patterns on it with nu-dialogue/moshi-finetune.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BASE_EXP="${BASE_EXP:-exp100_full_ft}"
NUM_CASES="${NUM_CASES:-250}"
SWEEP_PATTERNS="${SWEEP_PATTERNS:-f01 f02}"
SWEEP_PATTERNS="${SWEEP_PATTERNS//,/ }"
NPROC="${NPROC:-2}"
RUN_ID="${RUN_ID:-full_$(date +%Y%m%d_%H%M%S)}"
INPUT_SRC_RUN_DIR="${SRC_RUN_DIR:-}"

export NPROC
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-job_fullft_hp}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:///$REPO_ROOT/mlruns/mlflow.db}"
export MLFLOW_ARTIFACT_ROOT="${MLFLOW_ARTIFACT_ROOT:-file:$REPO_ROOT/mlruns/artifacts}"
export NU_MOSHI_FT_REPO="${NU_MOSHI_FT_REPO:-$REPO_ROOT/../moshi-finetune-nu-dialogue}"
export NU_DATA_DIR="${NU_DATA_DIR:-$REPO_ROOT/data/nu_fullft/$RUN_ID}"

if [[ -z "$INPUT_SRC_RUN_DIR" ]] && {
    ! [[ "$NUM_CASES" =~ ^[0-9]+$ ]] || [[ "$NUM_CASES" -lt 2 ]];
}; then
    echo "ERROR: NUM_CASES must be an integer >= 2 for train/eval split. got: $NUM_CASES" >&2
    exit 1
fi

clear_hp_env() {
    unset HP_LR HP_WEIGHT_DECAY HP_PCT_START
    unset HP_BATCH_SIZE HP_NUM_MICROBATCHES HP_MAX_STEPS
    unset HP_CKPT_FREQ HP_EVAL_FREQ HP_LOG_FREQ HP_MAX_NORM
    unset HP_DURATION_SEC HP_GRADIENT_CHECKPOINTING HP_PARAM_DTYPE HP_SEED
    unset HP_LORA_RANK HP_LORA_SCALING HP_LORA_ENABLE HP_LORA_FT_EMBED
}

apply_pattern() {
    local pattern="$1"
    clear_hp_env

    # nu-dialogue/moshi-finetune default learning rate.
    HP_LR=3e-5
    HP_WEIGHT_DECAY=0.1
    HP_PCT_START=0
    HP_BATCH_SIZE=1
    HP_NUM_MICROBATCHES=8
    HP_MAX_STEPS=1200
    HP_CKPT_FREQ=120
    HP_EVAL_FREQ=60
    HP_LOG_FREQ=5
    HP_MAX_NORM=1.0
    HP_DURATION_SEC=60

    case "$pattern" in
        f01) ;;                                      # A100-safe fixed-LR baseline
        f02) HP_DURATION_SEC=80 ;;                   # longer context, more memory
        f03) HP_DURATION_SEC=40 ;;                   # shorter context, lower activation memory
        f04) HP_WEIGHT_DECAY=0.01 ;;                 # weaker regularization
        f05) HP_WEIGHT_DECAY=0.2 ;;                  # stronger regularization
        f06) HP_NUM_MICROBATCHES=16 ;;               # same microbatch, larger effective batch
        f07) HP_DURATION_SEC=30 ;;                   # emergency low-memory context
        f08) HP_MAX_NORM=0.5 ;;                      # tighter gradient clipping
        f09) HP_MAX_STEPS=800; HP_CKPT_FREQ=80; HP_EVAL_FREQ=40 ;;   # shorter exposure
        f10) HP_MAX_STEPS=1600; HP_CKPT_FREQ=160; HP_EVAL_FREQ=80 ;; # longer exposure
        lr_3e-5) ;;                                  # nu-dialogue default LR
        lr_2e-5) HP_LR=2e-5 ;;                       # slightly lower LR
        lr_1e-5) HP_LR=1e-5 ;;                       # lower LR
        lr_5e-6) HP_LR=5e-6 ;;                       # conservative LR
        *)
            echo "ERROR: unknown full-FT sweep pattern: $pattern" >&2
            exit 1
            ;;
    esac

    export HP_LR HP_WEIGHT_DECAY HP_PCT_START
    export HP_BATCH_SIZE HP_NUM_MICROBATCHES HP_MAX_STEPS HP_DURATION_SEC
    export HP_CKPT_FREQ HP_EVAL_FREQ HP_LOG_FREQ HP_MAX_NORM
}

echo "===== full-ft sweep ====="
echo "run_id:      $RUN_ID"
echo "base_exp:    $BASE_EXP"
echo "num_cases:   $NUM_CASES"
echo "src_run_dir: ${INPUT_SRC_RUN_DIR:-<generate new>}"
echo "patterns:    $SWEEP_PATTERNS"
echo "mlflow_exp:  $MLFLOW_EXPERIMENT_NAME"
echo "mlflow_uri:  $MLFLOW_TRACKING_URI"
echo "artifact:    $MLFLOW_ARTIFACT_ROOT"
echo "cuda:        $CUDA_VISIBLE_DEVICES"
echo "nproc:       $NPROC"
echo "nu_repo:     $NU_MOSHI_FT_REPO"
echo "nu_data:     $NU_DATA_DIR"
echo "started_at:  $(date -Iseconds)"
echo "========================="

command -v uv >/dev/null 2>&1 || {
    echo "ERROR: uv is not available on PATH." >&2
    exit 1
}

echo "[full-ft sweep] checking MLflow availability"
if command -v timeout >/dev/null 2>&1; then
    MLFLOW_CHECK_CMD=(timeout "${MLFLOW_IMPORT_TIMEOUT:-120}" uv run python -c "import mlflow")
else
    MLFLOW_CHECK_CMD=(uv run python -c "import mlflow")
fi
if ! "${MLFLOW_CHECK_CMD[@]}" >/dev/null 2>&1; then
    echo "[full-ft sweep] installing mlflow into project uv environment"
    uv pip install mlflow
fi

echo
if [[ -n "$INPUT_SRC_RUN_DIR" ]]; then
    echo "[full-ft sweep] using existing shared dataset"
    SRC_RUN_DIR="$INPUT_SRC_RUN_DIR"
else
    echo "[full-ft sweep] generating shared 3h dataset"
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

for pattern in $SWEEP_PATTERNS; do
    apply_pattern "$pattern"
    export MLFLOW_RUN_NAME="${RUN_ID}_${pattern}"
    SWEEP_EXP_NAME="_fullft_sweeps/${RUN_ID}_${pattern}"
    SWEEP_EXP_DIR="experiments/$SWEEP_EXP_NAME"
    mkdir -p "$SWEEP_EXP_DIR"
    cp "experiments/$BASE_EXP/config.yaml" "$SWEEP_EXP_DIR/config.yaml"
    if [[ -f "experiments/$BASE_EXP/HYPERPARAMS.md" ]]; then
        cp "experiments/$BASE_EXP/HYPERPARAMS.md" "$SWEEP_EXP_DIR/HYPERPARAMS.md"
    fi

    export REFRESH_EXP_DATA=1

    echo
    echo "[full-ft sweep] running $pattern"
    echo "  exp=$SWEEP_EXP_NAME"
    echo "  lr=$HP_LR batch=$HP_BATCH_SIZE micro=$HP_NUM_MICROBATCHES steps=$HP_MAX_STEPS wd=$HP_WEIGHT_DECAY pct_start=$HP_PCT_START max_norm=$HP_MAX_NORM duration=${HP_DURATION_SEC:-100}"
    bash scripts/run_nu_fullft_experiment.sh "$SWEEP_EXP_NAME" "$SRC_RUN_DIR"
done

echo
echo "finished_at: $(date -Iseconds)"
