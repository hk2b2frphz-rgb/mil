#!/usr/bin/env bash
# Resume the complete trainer state in one stable directory, including after
# walltime. A normal trainer exit also covers early stopping below max_steps.
set -euo pipefail
export SRC_RUN_DIR="$ARM_ROOT/paired"
test -s "$COMPARE_ROOT/paired_summary.json"
test -s "$SRC_RUN_DIR/training_set/synthetic_moshi_train.jsonl"
source scripts/setup_proxy.sh
source scripts/setup_pbs_distributed.sh
source scripts/train_timebox.sh
export NPROC="${COMPARE_NPROC:-2}"
export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NPROC - 1)))"
export HP_SEED="$SEED" HP_LR=7e-6 HP_WEIGHT_DECAY=0.1 HP_PCT_START=0
export HP_BATCH_SIZE=1 HP_NUM_MICROBATCHES=8 HP_LOG_FREQ=5 HP_MAX_NORM=1.0
export HP_DURATION_SEC=170 HP_MAX_EPOCHS=12 HP_WARMUP_EPOCHS=1
export HP_EVAL_EVERY_EPOCH=0.5 HP_CKPT_EVERY_EPOCH=1
export HP_EARLY_STOPPING_PATIENCE=4 HP_EARLY_STOPPING_MIN_DELTA=0.001
export NU_EVAL_FRACTION=0.1 NU_MAX_LENGTH=2125 NU_NUM_TRAIN_EPOCHS=12
export NU_DATA_DIR="$PWD/data/nu_fullft/$RUN_ID"
export NU_OUTPUT_DIR="$PWD/experiments/_fullft_sweeps/$RUN_ID/checkpoints/nu_chain"
export NU_RESUME=1 NU_KEEP_TOP_K=3 NU_KEEP_BEST_ONLY=0
export SKIP_AUTO_POSTPROCESS=1 SKIP_AUTO_EVAL=1
export TQDM_DISABLE=1
CHAIN_INDEX="${CHAIN_INDEX:-1}"
MAX_LINKS="${MAX_LINKS:-8}"
STOP_FILE="${STOP_FILE:-$HOME/.miltoka/stop_fullft_chain}"
mkdir -p "$COMPARE_ROOT/jobs"
previous_info="$(timebox_latest_fullft_ckpt "$NU_OUTPUT_DIR" 2>/dev/null || true)"
previous_step="${previous_info%%$'\t'*}"
previous_step="${previous_step:-0}"
unset NU_RESUME_STEP_DIR
if [[ -n "$previous_info" ]]; then
    export NU_RESUME_STEP_DIR="${previous_info#*$'\t'}"
elif compgen -G "$NU_OUTPUT_DIR/step_*" >/dev/null; then
    echo 'Checkpoints exist but none is settled and usable; refusing an unsafe resume.' >&2
    exit 1
fi
timebox_init
set +e
timebox_run "$TIMEBOX_DEADLINE" bash scripts/run_nu_fullft_experiment.sh "_fullft_sweeps/${RUN_ID}_f01" "$SRC_RUN_DIR"
status=$?
set -e
if [[ "$status" == 124 ]]; then
    # The interrupted runner may not have had time to write its progress file.
    # Inspect a settled, usable checkpoint rather than trusting stale progress.
    timebox_latest_fullft_ckpt "$NU_OUTPUT_DIR" > "$COMPARE_ROOT/jobs/${ARM}_resume_checkpoint.txt" || {
        echo 'No usable checkpoint to resume; stopping chain.' >&2; exit 1;
    }
    IFS=$'\t' read -r next_step next_path < "$COMPARE_ROOT/jobs/${ARM}_resume_checkpoint.txt"
    [[ "$next_step" -gt "$previous_step" ]] || {
        echo 'No checkpoint progress in this link; stopping chain.' >&2; exit 1;
    }
    [[ ! -f "$STOP_FILE" ]] || { echo "Stopped by $STOP_FILE"; exit 0; }
    [[ "$CHAIN_INDEX" -lt "$MAX_LINKS" ]] || { echo 'MAX_LINKS reached; resubmit manually.' >&2; exit 1; }
    next=$((CHAIN_INDEX + 1))
    qsub -V -v "CHAIN_INDEX=$next,MAX_LINKS=$MAX_LINKS,COMPARE_ID=$COMPARE_ID,COMPARE_N=$COMPARE_N" \
        "scripts/2026-10-07/train_${ARM}.pbs" | tee "$COMPARE_ROOT/jobs/${ARM}_train_next.txt"
    exit 0
fi
[[ "$status" == 0 ]] || exit "$status"
# Export explicitly after a successful trainer exit, including early stopping.
# The generic runner's auto mode only recognizes reaching max_steps.
exp_dir="$PWD/experiments/_fullft_sweeps/${RUN_ID}_f01"
best_json="$COMPARE_ROOT/jobs/${ARM}_best_checkpoint.json"
cat "$exp_dir"/run_nu_*.log > "$COMPARE_ROOT/jobs/${ARM}_combined_metrics.log"
uv run python scripts/select_best_checkpoint.py --mode fullft \
    --log-file "$COMPARE_ROOT/jobs/${ARM}_combined_metrics.log" \
    --checkpoints-dir "$NU_OUTPUT_DIR" --output-json "$best_json"
best_step="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_path"])' "$best_json")"
export_dir="$exp_dir/exported/best"
uv run python scripts/export_fullft_checkpoint.py --step-dir "$best_step" \
    --out-dir "$export_dir" --nu-repo "${NU_MOSHI_FT_REPO:-$PWD/../moshi-finetune-nu-dialogue}" \
    --intermediate-dir "$exp_dir/exported/intermediate_${PBS_JOBID:-manual}_$(date +%s)" \
    --model-dtype bfloat16
exported="$export_dir/model.safetensors"
test -s "$exported"
printf '%s\n' "$exported" > "$COMPARE_ROOT/jobs/${ARM}_model.txt"
# Benchmarking gets its own reservation after training/export, including an
# early-stopped run. Re-evaluate manually by submitting eval_<arm>.pbs.
qsub -V -v "COMPARE_ID=$COMPARE_ID,COMPARE_N=$COMPARE_N" "scripts/2026-10-07/eval_${ARM}.pbs" \
    | tee "$COMPARE_ROOT/jobs/${ARM}_eval.txt"
