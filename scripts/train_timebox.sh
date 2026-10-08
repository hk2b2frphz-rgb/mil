#!/usr/bin/env bash
# Stop training before walltime kills it, and pick up from the last checkpoint.
#
# Splitting a run into fixed step slices requires knowing the throughput in
# advance, and getting it wrong wastes a whole reservation. This does the
# obvious thing instead: train until an hour before walltime, stop, and let the
# next job resume from whatever checkpoint had been written by then. Nothing in
# the trainer needs to change -- moshi-finetune already writes a checkpoint
# every HP_CKPT_FREQ steps, so at most that many steps are lost per handoff.
#
#   timebox_init                      # sets TIMEBOX_DEADLINE from the walltime
#   timebox_run "$TIMEBOX_DEADLINE" bash scripts/run_experiment.sh ...
#   # returns 124 if it had to stop training, otherwise the command's status
#   timebox_latest_ckpt <ckpt-root>   # "<step><TAB><path>" of the newest one

# Seconds to leave between stopping training and walltime: enough to shut the
# trainer down, find the checkpoint and submit the successor.
TIMEBOX_LEAD_SEC="${TIMEBOX_LEAD_SEC:-3600}"
# How long to let the trainer exit on its own before killing it.
TIMEBOX_GRACE_SEC="${TIMEBOX_GRACE_SEC:-300}"
TIMEBOX_POLL_SEC="${TIMEBOX_POLL_SEC:-30}"
# A checkpoint touched this recently may still be half-written; skip it and use
# the one before it.
TIMEBOX_SETTLE_SEC="${TIMEBOX_SETTLE_SEC:-60}"

timebox_walltime_sec() {
    if [[ -n "${TIMEBOX_WALLTIME_SEC:-}" ]]; then
        echo "$TIMEBOX_WALLTIME_SEC"
        return 0
    fi
    local hms=""
    if [[ -n "${PBS_JOBID:-}" ]] && command -v qstat >/dev/null 2>&1; then
        hms="$(qstat -f "$PBS_JOBID" 2>/dev/null \
            | tr -d ' \t' | grep -m1 '^Resource_List.walltime=' | cut -d= -f2 || true)"
    fi
    if [[ "$hms" =~ ^([0-9]+):([0-9]{2}):([0-9]{2})$ ]]; then
        echo $(( 10#${BASH_REMATCH[1]} * 3600 + 10#${BASH_REMATCH[2]} * 60 + 10#${BASH_REMATCH[3]} ))
    else
        echo 86400
    fi
}

timebox_init() {
    local start="${1:-$(date +%s)}"
    TIMEBOX_WALLTIME_SEC="$(timebox_walltime_sec)"
    TIMEBOX_DEADLINE=$(( start + TIMEBOX_WALLTIME_SEC - TIMEBOX_LEAD_SEC ))
    if [[ "$TIMEBOX_DEADLINE" -le "$start" ]]; then
        echo "[timebox] walltime ${TIMEBOX_WALLTIME_SEC}s is shorter than the ${TIMEBOX_LEAD_SEC}s lead; not timeboxing." >&2
        TIMEBOX_DEADLINE=0
    else
        echo "[timebox] walltime ${TIMEBOX_WALLTIME_SEC}s; training stops after $(( TIMEBOX_DEADLINE - start ))s"
    fi
    export TIMEBOX_DEADLINE TIMEBOX_WALLTIME_SEC
}

# Run a command, stopping it at the deadline. 124 means it was stopped.
timebox_run() {
    local deadline="$1"; shift
    if [[ "$deadline" -le 0 ]]; then
        "$@"
        return $?
    fi

    # Own process group, so the whole torchrun/python tree goes down with it --
    # killing the wrapper shell alone would leave training holding the GPU.
    if command -v setsid >/dev/null 2>&1; then
        setsid "$@" &
    else
        "$@" &
    fi
    local pid=$!
    local pgid
    pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')"
    pgid="${pgid:-$pid}"

    while kill -0 "$pid" 2>/dev/null; do
        if [[ "$(date +%s)" -ge "$deadline" ]]; then
            echo "[timebox] time budget reached; stopping training."
            kill -TERM "-$pgid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
            local waited=0
            while kill -0 "$pid" 2>/dev/null && [[ "$waited" -lt "$TIMEBOX_GRACE_SEC" ]]; do
                sleep 10
                waited=$(( waited + 10 ))
            done
            if kill -0 "$pid" 2>/dev/null; then
                echo "[timebox] still alive after ${TIMEBOX_GRACE_SEC}s; killing."
                kill -KILL "-$pgid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
            fi
            wait "$pid" 2>/dev/null || true
            return 124
        fi
        sleep "$TIMEBOX_POLL_SEC"
    done
    wait "$pid"
}

# Newest usable checkpoint under <root> (the dir holding checkpoint_NNNNNN).
timebox_latest_ckpt() {
    local root="$1"
    local now best_step="" best_path=""
    now="$(date +%s)"
    [[ -d "$root" ]] || return 1

    local dir file step mtime
    for dir in "$root"/checkpoint_*; do
        [[ -d "$dir" ]] || continue
        file="$dir/consolidated/lora.safetensors"
        [[ -s "$file" ]] || continue
        step="${dir##*checkpoint_}"
        [[ "$step" =~ ^[0-9]+$ ]] || continue
        step=$(( 10#$step ))
        # Skip anything still being written: we may have killed the trainer
        # mid-save, and half a checkpoint is worse than an older whole one.
        mtime="$(stat -c %Y "$file" 2>/dev/null || echo 0)"
        if [[ "$(( now - mtime ))" -lt "$TIMEBOX_SETTLE_SEC" ]]; then
            echo "[timebox] ignoring checkpoint_$step: written $(( now - mtime ))s ago, may be incomplete" >&2
            continue
        fi
        if [[ -z "$best_step" || "$step" -gt "$best_step" ]]; then
            best_step="$step"
            best_path="$file"
        fi
    done

    [[ -n "$best_step" ]] || return 1
    printf '%s\t%s\n' "$best_step" "$best_path"
}

# --------------------------------------------------------------- full-FT side
# nu-dialogue checkpoints look different: <root>/step_<N>/<tag>/ holding
# DeepSpeed ZeRO shards, and there is no consolidated weight until something
# runs zero_to_fp32 over them.
timebox_latest_fullft_ckpt() {
    local root="$1"
    local tag="${2:-pytorch_model}"
    local now best_step="" best_dir=""
    now="$(date +%s)"
    [[ -d "$root" ]] || return 1

    local dir tagdir step newest
    for dir in "$root"/step_*; do
        [[ -d "$dir" ]] || continue
        step="${dir##*step_}"
        [[ "$step" =~ ^[0-9]+$ ]] || continue
        tagdir="$dir/$tag"
        [[ -d "$tagdir" ]] || continue
        compgen -G "$tagdir/*model_states*" >/dev/null 2>&1 \
            || compgen -G "$tagdir/*zero_pp_rank*" >/dev/null 2>&1 \
            || continue
        newest="$(find "$tagdir" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1)"
        if [[ -n "$newest" && "$(( now - newest ))" -lt "$TIMEBOX_SETTLE_SEC" ]]; then
            echo "[timebox] ignoring step_${step}: written $(( now - newest ))s ago, may be incomplete" >&2
            continue
        fi
        step=$(( 10#$step ))
        if [[ -z "$best_step" || "$step" -gt "$best_step" ]]; then
            best_step="$step"
            best_dir="$dir"
        fi
    done

    [[ -n "$best_step" ]] || return 1
    printf '%s\t%s\n' "$best_step" "$best_dir"
}

# Turn a ZeRO checkpoint into the MoshiForFinetuning directory that training
# loads via --model_dir. Stage 2 of the export (clean_moshi) is for inference
# and is skipped here.
timebox_export_fullft_resume() {
    local step_dir="$1"
    local resume_dir="$2"
    echo "[timebox] consolidating $step_dir -> $resume_dir"
    uv run python scripts/export_fullft_checkpoint.py \
        --step-dir "$step_dir" \
        --out-dir "${resume_dir}_unused" \
        --intermediate-dir "$resume_dir" \
        --intermediate-only \
        --overwrite
}
