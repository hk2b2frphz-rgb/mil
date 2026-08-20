#!/usr/bin/env bash
# Build the `qsub -v` fragment that carries the cluster proxy into a job.
#
# scripts/setup_proxy.sh reads PROXY_URL (or the PROXY_HOST/PORT/USER/PASS/
# SCHEME form).  Finding none, it disables the proxy and unsets every
# http(s)_proxy variable -- the compute node then cannot reach huggingface.co
# and every Qwen3-TTS shard dies on the model download with
# "MaxRetryError HTTPSConnectionPool(host='huggingface.co', port=443)".
# Carrying these over lowercase -v is what keeps that from happening, in the
# chained jobs as much as in the first one.
#
# Usage:
#   source screw_poc/pbs/proxy_env.sh
#   proxy_env_warn_if_empty
#   qsub -v "$(proxy_env_fragment),TTS_RUN_DIR=..." job.pbs

# NO_PROXY is deliberately absent: its value is a comma-separated list, which
# `qsub -v` cannot carry.  setup_proxy.sh rebuilds it inside the job.
PROXY_ENV_VARS=(
    PROXY_URL
    PROXY_HOST
    PROXY_PORT
    PROXY_USER
    PROXY_PASS
    PROXY_SCHEME
    HTTP_PROXY
    HTTPS_PROXY
)

# Print the set PROXY_* assignments as a comma-joined `qsub -v` fragment.
# Prints nothing when none are set; fails when a value cannot survive -v.
proxy_env_fragment() {
    local name value
    local out=()
    for name in "${PROXY_ENV_VARS[@]}"; do
        value="${!name:-}"
        [[ -z "$value" ]] && continue
        if [[ "$value" == *,* ]]; then
            echo "ERROR: $name contains a comma, which qsub -v cannot carry." >&2
            echo "       Set PROXY_URL without commas, or URL-encode them." >&2
            return 1
        fi
        out+=("${name}=${value}")
    done
    [[ "${#out[@]}" -eq 0 ]] && return 0
    local IFS=,
    printf '%s' "${out[*]}"
}

# Names of the PROXY_* variables that are set, for logging.  Names only: a
# password inside PROXY_URL must not reach the terminal or the job log.
proxy_env_names() {
    local name
    local out=()
    for name in "${PROXY_ENV_VARS[@]}"; do
        [[ -n "${!name:-}" ]] && out+=("$name")
    done
    local IFS=' '
    printf '%s' "${out[*]}"
}

# Say so at submit time rather than letting the job find out on the compute
# node several minutes in.
proxy_env_warn_if_empty() {
    local fragment
    fragment="$(proxy_env_fragment)" || return 1
    if [[ -n "$fragment" ]]; then
        echo "proxy: forwarding $(proxy_env_names) via -v"
        return 0
    fi
    echo "WARNING: no PROXY_* variable is set in this shell." >&2
    echo "         setup_proxy.sh will report '[proxy] disabled' inside the job," >&2
    echo "         and the Qwen3-TTS shards will fail on the huggingface.co" >&2
    echo "         download.  Either:" >&2
    echo "           export PROXY_URL=http://<host>:<port>   # then re-run this" >&2
    echo "         or pre-populate the HF cache from a node that has access." >&2
}
