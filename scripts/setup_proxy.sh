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
#
# PBS jobs use `#PBS -V`, which can carry stale HTTP_PROXY/HTTPS_PROXY values
# from the submission host. Those implicit values are deliberately cleared
# unless PROXY_USE_ENV=1 is set. Prefer PROXY_URL or PROXY_HOST explicitly.

# Idempotent: PBS wrappers source this before invoking the reusable .sh
# runners, while users may invoke a runner directly on a compute node.
if [[ "${PROXY_CONFIG_INITIALIZED:-0}" == "1" ]]; then
    return 0 2>/dev/null || exit 0
fi
unset PROXY_CONFIG_VALID

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
elif [[ "${PROXY_USE_ENV:-0}" == "1" ]]; then
    proxy_url="${https_proxy:-${http_proxy:-${HTTPS_PROXY:-${HTTP_PROXY:-}}}}"
    if [[ -z "$proxy_url" ]]; then
        echo "ERROR: PROXY_USE_ENV=1 but no HTTP_PROXY/HTTPS_PROXY value is set." >&2
        return 1 2>/dev/null || exit 1
    fi
else
    # Do not let #PBS -V silently reuse a proxy hostname that only resolves on
    # the login/submission host. urllib reports that case as getaddrinfo -2.
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
    export PROXY_CONFIG_INITIALIZED=1
    echo "[proxy] disabled (set PROXY_URL/PROXY_HOST explicitly if required)"
    return 0 2>/dev/null || exit 0
fi

# Avoid a stale SOCKS/all-protocol proxy taking precedence in some clients.
unset all_proxy ALL_PROXY

proxy_python=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 &&
       "$candidate" -c "import socket, urllib.parse" >/dev/null 2>&1; then
        proxy_python="$candidate"
        break
    fi
done

if [[ -n "$proxy_python" ]]; then
    if ! normalized_proxy_url=$(PROXY_CHECK_URL="$proxy_url" "$proxy_python" - <<'PY'
import os
import socket
import sys
from urllib.parse import urlsplit

url = os.environ["PROXY_CHECK_URL"]
if any(char.isspace() for char in url):
    print("ERROR: proxy URL contains whitespace.", file=sys.stderr)
    raise SystemExit(1)
parsed = urlsplit(url if "://" in url else f"http://{url}")
if parsed.scheme not in {"http", "https"}:
    print(f"ERROR: unsupported proxy scheme: {parsed.scheme!r}", file=sys.stderr)
    raise SystemExit(1)
host = parsed.hostname
try:
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
except ValueError as exc:
    print(f"ERROR: invalid proxy port: {exc}", file=sys.stderr)
    raise SystemExit(1)
if not host:
    print("ERROR: proxy URL has no hostname.", file=sys.stderr)
    raise SystemExit(1)
if not 1 <= port <= 65535:
    print(f"ERROR: proxy port is out of range: {port}", file=sys.stderr)
    raise SystemExit(1)
try:
    socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
except socket.gaierror as exc:
    print(
        f"ERROR: proxy hostname cannot be resolved on this node: {host}:{port} ({exc})",
        file=sys.stderr,
    )
    raise SystemExit(1)
print(parsed.geturl())
PY
    ); then
        # A proxy inherited from a login node is frequently not resolvable on
        # a compute node.  Let callers use their direct network route rather
        # than leaking the bad value to requests/httpx.  Set PROXY_REQUIRED=1
        # for installations where direct access is intentionally forbidden.
        if [[ "${PROXY_REQUIRED:-0}" != "1" ]]; then
            unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
            export PROXY_CONFIG_INITIALIZED=1
            echo "[proxy] disabled: proxy is not resolvable on this node; using direct network" >&2
            return 0 2>/dev/null || exit 0
        fi
        echo "ERROR: invalid/unresolvable proxy configuration." >&2
        echo "Submit with a compute-node-resolvable PROXY_URL/PROXY_HOST, or unset PROXY_REQUIRED to use direct network." >&2
        return 1 2>/dev/null || exit 1
    fi
    # The validation step also normalizes shorthand such as "proxy:8080" to
    # a URI with an explicit scheme. requests/httpx reject the shorthand even
    # though urllib's resolver check can parse it.
    proxy_url="$normalized_proxy_url"
fi

export http_proxy="$proxy_url"
export https_proxy="$proxy_url"
export HTTP_PROXY="$proxy_url"
export HTTPS_PROXY="$proxy_url"
export PROXY_CONFIG_VALID=1
export PROXY_CONFIG_INITIALIZED=1

proxy_no_proxy="${NO_PROXY:-${no_proxy:-localhost,127.0.0.1,::1}}"
export no_proxy="$proxy_no_proxy"
export NO_PROXY="$proxy_no_proxy"

proxy_display="${proxy_url#*://}"
proxy_display="${proxy_display#*@}"
proxy_display="${proxy_display%%/*}"
echo "[proxy] enabled: ${proxy_display}"
