#!/usr/bin/env bash
# Source this from PBS jobs before launching training. It assigns a stable
# per-job distributed init port unless the user explicitly set MASTER_PORT.

if [[ -z "${MASTER_PORT:-}" ]]; then
    pbs_job_key="${PBS_JOBID:-${JOB_ID_SHORT:-}}"
    pbs_job_key="${pbs_job_key%%.*}"
    pbs_job_key="${pbs_job_key%%[*}"
    pbs_job_digits="$(printf '%s' "$pbs_job_key" | tr -cd '0-9')"

    if [[ -n "$pbs_job_digits" ]]; then
        MASTER_PORT=$((20000 + (10#$pbs_job_digits % 20000)))
    else
        MASTER_PORT=$((20000 + ($(date +%s) % 20000)))
    fi
fi

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT

echo "[dist] master_addr=${MASTER_ADDR} master_port=${MASTER_PORT}"
