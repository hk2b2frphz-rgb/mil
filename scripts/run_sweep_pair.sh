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

clear_hp_env() {
    unset HP_LR HP_WEIGHT_DECAY HP_PCT_START HP_LR_SCHEDULER
    unset HP_LORA_RANK HP_LORA_SCALING HP_LORA_ENABLE HP_LORA_FT_EMBED
    unset HP_BATCH_SIZE HP_NUM_MICROBATCHES HP_MAX_STEPS
    unset HP_CKPT_FREQ HP_EVAL_FREQ HP_LOG_FREQ HP_MAX_NORM
    unset HP_DURATION_SEC HP_GRADIENT_CHECKPOINTING HP_PARAM_DTYPE HP_SEED
    unset HP_EARLY_STOPPING_PATIENCE HP_EARLY_STOPPING_MIN_DELTA
}

apply_pattern() {
    local pattern="$1"
    clear_hp_env

    # Baseline is now the kyutai example (moshi-finetune/example/moshi_7B.yaml)
    # verbatim, except duration_sec (see below).
    #
    # We had drifted: rank 32 against the example's 128, batch 8 against 16, lr
    # raised to 5e-6, and warmup_constant instead of the upstream one_cycle. The
    # LR was raised to get the fixed opening greeting to stick; it did not, and
    # the greeting instead came out as a degenerate repeated syllable while the
    # timing and voice were fine. That points at adapter capacity rather than
    # step count, and rank 32 was the largest single deviation from the example.
    # So: go back to the published combination and re-measure from there,
    # re-applying deviations one at a time instead of carrying four at once.
    HP_LR=2e-6
    HP_WEIGHT_DECAY=0.1
    HP_PCT_START=0.05
    # No HP_LR_SCHEDULER: the example does not set one, so this falls through to
    # the upstream one_cycle default. warmup_constant is a local addition.
    HP_LORA_RANK=128
    HP_LORA_SCALING=2.0
    HP_BATCH_SIZE=16
    HP_NUM_MICROBATCHES=1
    HP_MAX_STEPS=2000
    # The one value deliberately kept away from the example (which uses 100).
    # Set explicitly (rather than left to the experiment's config.yaml) so a
    # dialogue is never split into multiple training samples: moshi-finetune
    # crops duration_sec per sample, and any chunk after the first starts
    # mid-conversation without the fixed opening greeting, teaching the model
    # not to greet from an empty context. The corpus sits under 150s except
    # for a single 240s outlier, which is not worth padding every other
    # sample for.
    HP_DURATION_SEC=170
    HP_CKPT_FREQ=100
    HP_EVAL_FREQ=100
    HP_LOG_FREQ=1
    # Early stopping on eval loss, counted in evaluations: 6 x eval_freq = 600
    # steps without improvement before stopping. Generous on purpose -- the LoRA
    # runs have been undertrained rather than overfitted, so this is a ceiling
    # that stops a run once it is genuinely flat, not a tight leash.
    # The example has no equivalent (it sets do_eval: false); this and
    # keep_best_only are local additions kept on top of the example's values.
    HP_EARLY_STOPPING_PATIENCE=6
    HP_EARLY_STOPPING_MIN_DELTA=0.001

    case "$pattern" in
        # These patterns pin their own values so a rerun reproduces the run made
        # under that name rather than silently inheriting a changed baseline.
        # Everything they do not pin now comes from the kyutai example, so a
        # pattern re-run after the baseline change is NOT comparable to the same
        # pattern run before it -- use a fresh RUN_ID.
        h01) HP_LR=2e-6 ;;                           # == the example LR (now also the default)
        h01_long) HP_LR=2e-6; HP_MAX_STEPS=3600 ;;   # h01 with more exposure
        h02) HP_LR=5e-6 ;;                           # higher learning rate
        h03) HP_LR=1e-6 ;;                           # lower learning rate
        h04) HP_LORA_RANK=64 ;;                      # smaller adapter than the example's 128
        h05) HP_LORA_RANK=16 ;;                      # much smaller adapter
        h06) HP_LORA_SCALING=1.0 ;;                  # lower LoRA alpha/r
        h07) HP_LORA_SCALING=4.0 ;;                  # higher LoRA alpha/r
        h08) HP_BATCH_SIZE=4; HP_NUM_MICROBATCHES=2 ;; # same effective batch, lower per-step memory
        h09) HP_PCT_START=0.10 ;;                    # longer warmup
        h10) HP_WEIGHT_DECAY=0.01 ;;                 # weaker regularization
        lr_2e-6) HP_LR=2e-6; HP_MAX_STEPS=1200 ;;    # Kyutai example default LR
        lr_1p5e-6) HP_LR=1.5e-6; HP_MAX_STEPS=1200 ;; # slightly lower LR
        lr_1e-6) HP_LR=1e-6; HP_MAX_STEPS=1200 ;;    # lower LR
        lr_5e-7) HP_LR=5e-7; HP_MAX_STEPS=1200 ;;    # conservative LR
        fixed_1e-6)
            HP_LR=1e-6
            HP_MAX_STEPS=1200
            HP_LR_SCHEDULER=warmup_constant
            ;;                                        # 5% warmup, then fixed LR
        onecycle_2e-6)
            HP_LR=2e-6
            HP_MAX_STEPS=1200
            HP_LR_SCHEDULER=one_cycle
            ;;                                        # previous baseline reference
        *)
            echo "ERROR: unknown sweep pattern: $pattern" >&2
            exit 1
            ;;
    esac

    export HP_LR HP_WEIGHT_DECAY HP_PCT_START HP_LR_SCHEDULER
    export HP_LORA_RANK HP_LORA_SCALING HP_BATCH_SIZE HP_NUM_MICROBATCHES
    export HP_MAX_STEPS HP_CKPT_FREQ HP_EVAL_FREQ HP_LOG_FREQ HP_DURATION_SEC
    export HP_EARLY_STOPPING_PATIENCE HP_EARLY_STOPPING_MIN_DELTA
}

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
    uv pip install mlflow
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

for pattern in $SWEEP_PATTERNS; do
    if sweep_chain_completed "$pattern"; then
        echo "[sweep] $pattern already done in an earlier link; skipping"
        continue
    fi
    sweep_chain_have_time || sweep_chain_handoff
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

    echo
    echo "[sweep] running $pattern"
    echo "  exp=$SWEEP_EXP_NAME"
    echo "  lr=$HP_LR rank=$HP_LORA_RANK scaling=$HP_LORA_SCALING batch=$HP_BATCH_SIZE micro=$HP_NUM_MICROBATCHES steps=$HP_MAX_STEPS wd=$HP_WEIGHT_DECAY pct_start=$HP_PCT_START"
    bash scripts/run_experiment.sh "$SWEEP_EXP_NAME" "$SRC_RUN_DIR"
    sweep_chain_record "$pattern" "$(( $(date +%s) - pattern_started ))"
done

echo
echo "finished_at: $(date -Iseconds)"
