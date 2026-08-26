#!/usr/bin/env bash
# Chain a sweep across walltime, one pattern at a time.
#
# A sweep runs N patterns back to back in a single job. When walltime expires
# partway through, every pattern after the current one is simply lost -- which
# is what happened to the LoRA sweep. This library lets the driver hand the
# unfinished patterns to a fresh job instead.
#
# The granularity is a whole pattern: a job stops before starting a pattern it
# does not have time to finish. It cannot rescue a *single* pattern that does
# not fit in one walltime -- for that, split the steps with
# scripts/run_train_chain.pbs.
#
# Completed patterns and their durations are recorded in
#   experiments/pbs_logs/<RUN_ID>_sweep_state.tsv
# so a successor skips what is done and knows how long a pattern takes.
#
# Source this from a sweep driver, then:
#   sweep_chain_init
#   for pattern in $SWEEP_PATTERNS; do
#       sweep_chain_completed "$pattern" && continue
#       sweep_chain_have_time || sweep_chain_handoff   # exits 0
#       started=$(date +%s)
#       ... run the pattern ...
#       sweep_chain_record "$pattern" "$(( $(date +%s) - started ))"
#   done
#
# The driver must have RUN_ID, SWEEP_PATTERNS and SRC_RUN_DIR set, and the PBS
# wrapper must export SELF_PBS so we know which job file to resubmit.
#
# Turn it off with SWEEP_CHAIN=0. End a running chain with:
#   touch ~/.miltoka/stop_sweep_chain

sweep_chain_init() {
    SWEEP_CHAIN="${SWEEP_CHAIN:-1}"
    SWEEP_CHAIN_INDEX="${SWEEP_CHAIN_INDEX:-1}"
    SWEEP_MAX_CHAIN_JOBS="${SWEEP_MAX_CHAIN_JOBS:-8}"
    # Stop this long before walltime so the successor is queued while we still
    # have room to exit cleanly.
    SWEEP_CHAIN_LEAD_SEC="${SWEEP_CHAIN_LEAD_SEC:-900}"
    SWEEP_STOP_FILE="${SWEEP_STOP_FILE:-$HOME/.miltoka/stop_sweep_chain}"
    # Used only for the first pattern of the first link, before any pattern has
    # been timed. 0 means "no idea, just run it".
    SWEEP_PATTERN_EST_SEC="${SWEEP_PATTERN_EST_SEC:-0}"
    SWEEP_SELF_PBS="${SELF_PBS:-}"
    SWEEP_START_EPOCH="$(date +%s)"

    mkdir -p experiments/pbs_logs
    SWEEP_STATE_FILE="${SWEEP_STATE_FILE:-experiments/pbs_logs/${RUN_ID}_sweep_state.tsv}"
    [[ -f "$SWEEP_STATE_FILE" ]] || : > "$SWEEP_STATE_FILE"

    if [[ "$SWEEP_CHAIN_INDEX" -gt "$SWEEP_MAX_CHAIN_JOBS" ]]; then
        echo "ERROR: SWEEP_CHAIN_INDEX=$SWEEP_CHAIN_INDEX exceeds SWEEP_MAX_CHAIN_JOBS=$SWEEP_MAX_CHAIN_JOBS." >&2
        exit 1
    fi

    # Ask PBS what walltime we actually got, rather than trusting the directive
    # in the job file which qsub -l may have overridden.
    if [[ -z "${SWEEP_WALLTIME_SEC:-}" && -n "${PBS_JOBID:-}" ]] && command -v qstat >/dev/null 2>&1; then
        local hms
        hms="$(qstat -f "$PBS_JOBID" 2>/dev/null \
            | tr -d ' \t' | grep -m1 '^Resource_List.walltime=' | cut -d= -f2 || true)"
        if [[ "$hms" =~ ^([0-9]+):([0-9]{2}):([0-9]{2})$ ]]; then
            SWEEP_WALLTIME_SEC=$(( 10#${BASH_REMATCH[1]} * 3600 + 10#${BASH_REMATCH[2]} * 60 + 10#${BASH_REMATCH[3]} ))
        fi
    fi
    SWEEP_WALLTIME_SEC="${SWEEP_WALLTIME_SEC:-86400}"
    SWEEP_DEADLINE=$(( SWEEP_START_EPOCH + SWEEP_WALLTIME_SEC - SWEEP_CHAIN_LEAD_SEC ))

    echo "[chain] link ${SWEEP_CHAIN_INDEX}/${SWEEP_MAX_CHAIN_JOBS}, walltime ${SWEEP_WALLTIME_SEC}s, state ${SWEEP_STATE_FILE}"
    if [[ "$SWEEP_CHAIN" != "1" ]]; then
        echo "[chain] disabled (SWEEP_CHAIN=0); patterns left over at walltime will be lost"
    elif [[ -z "$SWEEP_SELF_PBS" ]]; then
        echo "[chain] SELF_PBS is not set, so there is nothing to resubmit; running unchained"
        SWEEP_CHAIN=0
    fi
}

sweep_chain_completed() {
    cut -f1 "$SWEEP_STATE_FILE" 2>/dev/null | grep -qxF "$1"
}

sweep_chain_record() {
    printf '%s\t%s\t%s\n' "$1" "$2" "$(date -Iseconds)" >> "$SWEEP_STATE_FILE"
    echo "[chain] $1 finished in $2s"
}

# Longest pattern seen so far. A sweep's patterns differ mostly in LR and
# adapter size, so the slowest one so far is a fair guess for the next.
sweep_chain_estimate() {
    local worst
    worst="$(cut -f2 "$SWEEP_STATE_FILE" 2>/dev/null | sort -n | tail -n1)"
    if [[ -n "$worst" && "$worst" =~ ^[0-9]+$ ]]; then
        echo "$worst"
    else
        echo "$SWEEP_PATTERN_EST_SEC"
    fi
}

sweep_chain_have_time() {
    [[ "$SWEEP_CHAIN" == "1" ]] || return 0
    local est remaining
    est="$(sweep_chain_estimate)"
    # Nothing timed yet: refusing here would hand off forever without ever
    # running a pattern, so go ahead and measure one.
    [[ "$est" -gt 0 ]] || return 0
    remaining=$(( SWEEP_DEADLINE - $(date +%s) ))
    if [[ "$remaining" -lt "$est" ]]; then
        echo "[chain] ${remaining}s left, a pattern takes about ${est}s; handing over."
        return 1
    fi
    return 0
}

# Submit a job for whatever is still unfinished, then exit this one cleanly.
sweep_chain_handoff() {
    local pending=""
    local pattern
    for pattern in $SWEEP_PATTERNS; do
        sweep_chain_completed "$pattern" && continue
        pending="${pending:+${pending}+}${pattern}"
    done

    if [[ -z "$pending" ]]; then
        echo "[chain] nothing left to run."
        exit 0
    fi
    if [[ -e "$SWEEP_STOP_FILE" ]]; then
        echo "[chain] stop file present ($SWEEP_STOP_FILE); leaving these unrun: ${pending//+/ }"
        exit 0
    fi

    local next_index=$(( SWEEP_CHAIN_INDEX + 1 ))
    if [[ "$next_index" -gt "$SWEEP_MAX_CHAIN_JOBS" ]]; then
        echo "[chain] reached SWEEP_MAX_CHAIN_JOBS=${SWEEP_MAX_CHAIN_JOBS}; leaving these unrun: ${pending//+/ }" >&2
        exit 1
    fi
    local elapsed=$(( $(date +%s) - SWEEP_START_EPOCH ))
    if [[ "$elapsed" -lt "${SWEEP_MIN_JOB_SEC:-300}" ]]; then
        echo "[chain] this link lasted only ${elapsed}s; refusing to resubmit the same failure." >&2
        exit 1
    fi
    command -v qsub >/dev/null 2>&1 || {
        echo "[chain] qsub is not available here; cannot resubmit." >&2
        exit 1
    }

    # SWEEP_PATTERNS is passed '+'-separated: qsub -v splits on commas, and the
    # drivers normalize both separators back to spaces.
    local next_id
    next_id="$(qsub -V \
        -v "SWEEP_CHAIN=1,SWEEP_CHAIN_INDEX=${next_index},SWEEP_MAX_CHAIN_JOBS=${SWEEP_MAX_CHAIN_JOBS},SWEEP_CHAIN_LEAD_SEC=${SWEEP_CHAIN_LEAD_SEC},SWEEP_STOP_FILE=${SWEEP_STOP_FILE},SWEEP_STATE_FILE=${SWEEP_STATE_FILE},SELF_PBS=${SWEEP_SELF_PBS},RUN_ID=${RUN_ID},SRC_RUN_DIR=${SRC_RUN_DIR},SWEEP_PATTERNS=${pending},BASE_EXP=${BASE_EXP},MLFLOW_EXPERIMENT_NAME=${MLFLOW_EXPERIMENT_NAME}" \
        "$SWEEP_SELF_PBS")" || {
        echo "[chain] qsub failed; these were not run: ${pending//+/ }" >&2
        exit 1
    }

    echo "[chain] link ${next_index}/${SWEEP_MAX_CHAIN_JOBS} queued as ${next_id}"
    echo "[chain] it will run: ${pending//+/ }"
    echo "[chain] on the same dataset: ${SRC_RUN_DIR}"
    exit 0
}
