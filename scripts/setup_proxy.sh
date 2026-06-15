#!/usr/bin/env bash
# Source this from PBS jobs to configure proxy variables without storing
# credentials in the repository.
#
# Supported inputs:
#   PROXY_URL=http://user:pass@proxy.example.com:8080
# or:
#   PROXY_HOST=proxy.example.com
#   PROXY_PORT=8080
#   PROXY_USER=<username>
#   PROXY_PASS=<password>
#   PROXY_SCHEME=http
#
# If PROXY_USER or PROXY_PASS contains URL-special characters, set PROXY_URL
# directly with those values URL-encoded.

if [[ -n "${PROXY_URL:-}" ]]; then
    proxy_url="$PROXY_URL"
elif [[ -n "${PROXY_HOST:-}" ]]; then
    proxy_scheme="${PROXY_SCHEME:-http}"
    proxy_host="$PROXY_HOST"

    if [[ "$proxy_host" == *"://"* ]]; then
        proxy_base="$proxy_host"
    else
        proxy_base="${proxy_scheme}://${proxy_host}"
    fi

    if [[ -n "${PROXY_PORT:-}" && "$proxy_base" != *":${PROXY_PORT}" ]]; then
        proxy_base="${proxy_base}:${PROXY_PORT}"
    fi

    if [[ -n "${PROXY_USER:-}" || -n "${PROXY_PASS:-}" ]]; then
        proxy_url="${proxy_base/:\/\//:\/\/${PROXY_USER:-}:${PROXY_PASS:-}@}"
    else
        proxy_url="$proxy_base"
    fi
else
    return 0 2>/dev/null || exit 0
fi

export http_proxy="$proxy_url"
export https_proxy="$proxy_url"
export HTTP_PROXY="$proxy_url"
export HTTPS_PROXY="$proxy_url"

proxy_no_proxy="${NO_PROXY:-${no_proxy:-localhost,127.0.0.1,::1}}"
export no_proxy="$proxy_no_proxy"
export NO_PROXY="$proxy_no_proxy"

proxy_display="${proxy_url#*://}"
proxy_display="${proxy_display#*@}"
proxy_display="${proxy_display%%/*}"
echo "[proxy] enabled: ${proxy_display}"
