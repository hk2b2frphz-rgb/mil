#!/usr/bin/env bash
# Generate one ~3h dataset, then run two or more LoRA hyperparameter patterns on it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BASE_EXP="${BASE_EXP:-exp001_lora_baseline}"
NUM_CASES="${NUM_CASES:-250}"
SWEEP_PATTERNS="${SWEEP_PATTERNS:-h01 h02}"
NPROC="${NPROC:-1}"
RUN_ID="${RUN_ID:-run_$(date +%Y%m%d_%H%M%S)}"

export NPROC
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-job_sweep}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file:$REPO_ROOT/mlruns}"

if ! [[ "$NUM_CASES" =~ ^[0-9]+$ ]] || [[ "$NUM_CASES" -lt 2 ]]; then
    echo "ERROR: NUM_CASES must be an integer >= 2 for train/eval split. got: $NUM_CASES" >&2
    exit 1
fi

clear_hp_env() {
    unset HP_LR HP_WEIGHT_DECAY HP_PCT_START
    unset HP_LORA_RANK HP_LORA_SCALING HP_LORA_ENABLE HP_LORA_FT_EMBED
    unset HP_BATCH_SIZE HP_NUM_MICROBATCHES HP_MAX_STEPS
    unset HP_CKPT_FREQ HP_EVAL_FREQ HP_LOG_FREQ HP_MAX_NORM
    unset HP_DURATION_SEC HP_GRADIENT_CHECKPOINTING HP_PARAM_DTYPE HP_SEED
}

apply_pattern() {
    local pattern="$1"
    clear_hp_env

    HP_LR=2e-6
    HP_WEIGHT_DECAY=0.1
    HP_PCT_START=0.05
    HP_LORA_RANK=32
    HP_LORA_SCALING=2.0
    HP_BATCH_SIZE=8
    HP_NUM_MICROBATCHES=1
    HP_MAX_STEPS=1200
    HP_CKPT_FREQ=120
    HP_EVAL_FREQ=60
    HP_LOG_FREQ=5

    case "$pattern" in
        h01) ;;                                      # baseline
        h02) HP_LR=5e-6 ;;                           # higher learning rate
        h03) HP_LR=1e-6 ;;                           # lower learning rate
        h04) HP_LORA_RANK=64 ;;                      # larger adapter
        h05) HP_LORA_RANK=16 ;;                      # smaller adapter
        h06) HP_LORA_SCALING=1.0 ;;                  # lower LoRA alpha/r
        h07) HP_LORA_SCALING=4.0 ;;                  # higher LoRA alpha/r
        h08) HP_BATCH_SIZE=4; HP_NUM_MICROBATCHES=2 ;; # same effective batch, lower per-step memory
        h09) HP_PCT_START=0.10 ;;                    # longer warmup
        h10) HP_WEIGHT_DECAY=0.01 ;;                 # weaker regularization
        *)
            echo "ERROR: unknown sweep pattern: $pattern" >&2
            exit 1
            ;;
    esac

    export HP_LR HP_WEIGHT_DECAY HP_PCT_START
    export HP_LORA_RANK HP_LORA_SCALING HP_BATCH_SIZE HP_NUM_MICROBATCHES
    export HP_MAX_STEPS HP_CKPT_FREQ HP_EVAL_FREQ HP_LOG_FREQ
}

echo "===== sweep ====="
echo "run_id:      $RUN_ID"
echo "base_exp:    $BASE_EXP"
echo "num_cases:   $NUM_CASES"
echo "patterns:    $SWEEP_PATTERNS"
echo "mlflow_exp:  $MLFLOW_EXPERIMENT_NAME"
echo "mlflow_uri:  $MLFLOW_TRACKING_URI"
echo "cuda:        $CUDA_VISIBLE_DEVICES"
echo "started_at:  $(date -Iseconds)"
echo "================="

command -v uv >/dev/null 2>&1 || {
    echo "ERROR: uv is not available on PATH." >&2
    exit 1
}

if ! uv run python -c "import mlflow" >/dev/null 2>&1; then
    echo "[sweep] installing mlflow into project uv environment"
    uv pip install mlflow
fi

echo
echo "[sweep] generating shared dataset"
RUN_ID="$RUN_ID" \
NUM_CASES="$NUM_CASES" \
STEPS=use_cases,dialogues,audio \
bash scripts/run_pipeline.sh

SRC_RUN_DIR="./data/runs/${RUN_ID}"
if [[ ! -f "$SRC_RUN_DIR/training_set/synthetic_moshi_train.jsonl" ]]; then
    echo "ERROR: generated manifest not found: $SRC_RUN_DIR/training_set/synthetic_moshi_train.jsonl" >&2
    exit 1
fi

for pattern in $SWEEP_PATTERNS; do
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

    echo
    echo "[sweep] running $pattern"
    echo "  exp=$SWEEP_EXP_NAME"
    echo "  lr=$HP_LR rank=$HP_LORA_RANK scaling=$HP_LORA_SCALING batch=$HP_BATCH_SIZE micro=$HP_NUM_MICROBATCHES steps=$HP_MAX_STEPS wd=$HP_WEIGHT_DECAY pct_start=$HP_PCT_START"
    bash scripts/run_experiment.sh "$SWEEP_EXP_NAME" "$SRC_RUN_DIR"
done

echo
echo "finished_at: $(date -Iseconds)"
