#!/usr/bin/env bash
# Shared helpers for timestamped default run ids in data-generation launchers.

_run_id_job_tag() {
    local job_tag="${PBS_ARRAY_JOBID:-${PBS_JOBID:-}}"
    job_tag="${job_tag%%.*}"
    job_tag="${job_tag%%\[*}"
    job_tag="${job_tag//[^A-Za-z0-9_-]/}"
    printf '%s' "$job_tag"
}

run_id_stamp() {
    local stamp="${RUN_STAMP:-}"
    local job_tag

    job_tag="$(_run_id_job_tag)"
    if [[ -z "$stamp" ]]; then
        if [[ -n "${PBS_ARRAY_INDEX:-}${PBS_ARRAYID:-}" && -n "$job_tag" ]]; then
            stamp="$job_tag"
        else
            stamp="$(date +%Y%m%d_%H%M%S)"
            if [[ -n "$job_tag" ]]; then
                stamp="${stamp}_${job_tag}"
            fi
        fi
    fi
    stamp="${stamp//[^A-Za-z0-9_-]/_}"

    printf '%s' "$stamp"
}

default_run_id() {
    local prefix="$1"
    printf '%s_%s' "$prefix" "$(run_id_stamp)"
}
