#!/usr/bin/env bash
set -uo pipefail

# Local, GPU-free check for run_full_duplex_lora_chain.pbs. It replaces data
# generation, training, and qsub inside a temporary sandbox.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHAIN_SRC="$REPO_ROOT/scripts/run_full_duplex_lora_chain.pbs"
TIMEBOX_SRC="$REPO_ROOT/scripts/train_timebox.sh"
SANDBOX="$(mktemp -d)"
trap 'rm -rf "$SANDBOX"' EXIT

PASS=0
FAIL=0

ok() {
    PASS=$((PASS + 1))
    echo "  [OK]   $1"
}

ng() {
    FAIL=$((FAIL + 1))
    echo "  [FAIL] $1"
}

check() {
    if eval "$2"; then ok "$1"; else ng "$1"; fi
}

setup_sandbox() {
    rm -rf "$SANDBOX/work"
    mkdir -p "$SANDBOX/work/scripts" "$SANDBOX/work/bin"
    cp "$CHAIN_SRC" "$SANDBOX/work/scripts/run_full_duplex_lora_chain.pbs"
    cp "$TIMEBOX_SRC" "$SANDBOX/work/scripts/train_timebox.sh"
    printf ':\n' > "$SANDBOX/work/scripts/setup_proxy.sh"

    cat > "$SANDBOX/work/scripts/run_full_duplex_training_data.sh" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${TEST_DATA_RESULT:-success}" == "timeout" ]]; then
    exit 124
fi
mkdir -p "$OUT_ROOT/training_set"
printf '{"path":"a.wav"}\n{"path":"b.wav"}\n' \
    > "$OUT_ROOT/training_set/synthetic_moshi_train.jsonl"
STUB

    cat > "$SANDBOX/work/scripts/run_experiment.sh" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${RUN_EXPERIMENT_POSTPROCESS_ONLY:-0}" == "1" ]]; then
    exit "${TEST_FINALIZE_RC:-0}"
fi
if [[ "${TEST_TRAIN_RESULT:-success}" == "timeout" ]]; then
    step="${TEST_CKPT_STEP:-50}"
    dir="experiments/$1/checkpoints/$RUN_TS/checkpoints/checkpoint_$(printf '%06d' "$step")/consolidated"
    mkdir -p "$dir"
    printf x > "$dir/lora.safetensors"
    exit 124
fi
exit 0
STUB

    cat > "$SANDBOX/work/bin/qsub" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
queue=""
select=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -V) shift ;;
        -q) queue="$2"; shift 2 ;;
        -l)
            [[ "$2" == select=* ]] && select="${2#select=}"
            shift 2
            ;;
        *) shift ;;
    esac
done
printf 'stage=%s index=%s queue=%s select=%s resume_step=%s resume_from=%s finalize_run=%s\n' \
    "${PIPELINE_STAGE:-}" "${CHAIN_INDEX:-}" "$queue" "$select" \
    "${TRAIN_RESUME_STEP:-}" "${TRAIN_RESUME_FROM:-}" "${FINALIZE_RUN_TS:-}" \
    >> "$SUBMIT_LOG"
echo stub.job
STUB
    chmod +x "$SANDBOX/work/scripts/"*.sh "$SANDBOX/work/bin/qsub"
    : > "$SANDBOX/submit.log"
    : > "$SANDBOX/run.log"
}

run_chain() {
    (
        cd "$SANDBOX/work" || exit 1
        export PATH="$SANDBOX/work/bin:$PATH"
        export PBS_O_WORKDIR="$SANDBOX/work"
        export SUBMIT_LOG="$SANDBOX/submit.log"
        env TIMEBOX_SETTLE_SEC=0 TIMEBOX_POLL_SEC=1 TIMEBOX_GRACE_SEC=1 \
            TOTAL_STEPS=100 EXP_NAME=test CHAIN_ID=check CHAIN_INDEX=1 \
            DATA_RUNNER=scripts/run_full_duplex_training_data.sh \
            "$@" /bin/bash scripts/run_full_duplex_lora_chain.pbs
    ) >> "$SANDBOX/run.log" 2>&1
    echo $?
}

echo "[1] Data timeout continues on the data queue"
setup_sandbox
status="$(run_chain TEST_DATA_RESULT=timeout PIPELINE_STAGE=data OUT_ROOT="$SANDBOX/work/data")"
check "exit status is zero after successor submission" "[[ '$status' == 0 ]]"
check "next stage remains data" "grep -q 'stage=data index=2 queue=xan_s select=1:res=small' '$SANDBOX/submit.log'"

echo "[2] Completed data moves to the training queue"
setup_sandbox
status="$(run_chain TEST_DATA_RESULT=success PIPELINE_STAGE=data OUT_ROOT="$SANDBOX/work/data")"
check "exit status is zero" "[[ '$status' == 0 ]]"
check "next stage is fresh training" "grep -q 'stage=train index=2 queue=xan_s.*resume_step=0' '$SANDBOX/submit.log'"

echo "[3] Training timeout resumes from a settled checkpoint"
setup_sandbox
mkdir -p "$SANDBOX/work/data/training_set"
printf '{"path":"a.wav"}\n{"path":"b.wav"}\n' \
    > "$SANDBOX/work/data/training_set/synthetic_moshi_train.jsonl"
status="$(run_chain TEST_TRAIN_RESULT=timeout TEST_CKPT_STEP=50 \
    PIPELINE_STAGE=train SRC_RUN_DIR="$SANDBOX/work/data")"
check "exit status is zero" "[[ '$status' == 0 ]]"
check "next training link starts at checkpoint 50" \
    "grep -q 'stage=train index=2 queue=xan_s.*resume_step=50 resume_from=.*checkpoint_000050.*/lora.safetensors' '$SANDBOX/submit.log'"

echo "[4] A final checkpoint continues with postprocessing"
setup_sandbox
mkdir -p "$SANDBOX/work/data/training_set"
printf '{"path":"a.wav"}\n{"path":"b.wav"}\n' \
    > "$SANDBOX/work/data/training_set/synthetic_moshi_train.jsonl"
status="$(run_chain TEST_TRAIN_RESULT=timeout TEST_CKPT_STEP=100 \
    PIPELINE_STAGE=train SRC_RUN_DIR="$SANDBOX/work/data")"
check "exit status is zero" "[[ '$status' == 0 ]]"
check "next stage is finalize and keeps the run directory" \
    "grep -q 'stage=finalize index=2 queue=xan_s.*finalize_run=check_train01' '$SANDBOX/submit.log'"

echo "[5] Successful training and finalization stop the chain"
setup_sandbox
mkdir -p "$SANDBOX/work/data/training_set"
printf '{"path":"a.wav"}\n{"path":"b.wav"}\n' \
    > "$SANDBOX/work/data/training_set/synthetic_moshi_train.jsonl"
status="$(run_chain TEST_TRAIN_RESULT=success PIPELINE_STAGE=train SRC_RUN_DIR="$SANDBOX/work/data")"
check "successful training exits zero" "[[ '$status' == 0 ]]"
check "successful training does not submit" "[[ ! -s '$SANDBOX/submit.log' ]]"

setup_sandbox
mkdir -p "$SANDBOX/work/data/training_set"
printf '{"path":"a.wav"}\n{"path":"b.wav"}\n' \
    > "$SANDBOX/work/data/training_set/synthetic_moshi_train.jsonl"
status="$(run_chain PIPELINE_STAGE=finalize FINALIZE_RUN_TS=previous_train \
    SRC_RUN_DIR="$SANDBOX/work/data")"
check "successful finalization exits zero" "[[ '$status' == 0 ]]"
check "successful finalization does not submit" "[[ ! -s '$SANDBOX/submit.log' ]]"

echo "[6] Stop file and chain cap prevent runaway submission"
setup_sandbox
mkdir -p "$SANDBOX/work/data"
touch "$SANDBOX/work/data/stop"
status="$(run_chain TEST_DATA_RESULT=timeout PIPELINE_STAGE=data \
    OUT_ROOT="$SANDBOX/work/data" CHAIN_STOP_FILE="$SANDBOX/work/data/stop")"
check "stop file exits zero" "[[ '$status' == 0 ]]"
check "stop file prevents qsub" "[[ ! -s '$SANDBOX/submit.log' ]]"

setup_sandbox
status="$(run_chain TEST_DATA_RESULT=success PIPELINE_STAGE=data \
    OUT_ROOT="$SANDBOX/work/data" MAX_CHAIN_JOBS=1)"
check "chain cap rejects another link" "[[ '$status' != 0 ]]"
check "chain cap prevents qsub" "[[ ! -s '$SANDBOX/submit.log' ]]"

echo "PASS: $PASS  FAIL: $FAIL"
if [[ "$FAIL" -ne 0 ]]; then
    tail -100 "$SANDBOX/run.log"
    exit 1
fi
