#!/usr/bin/env bash
# Run inside the allocated node, with the same environment as TTS workers.
set -euo pipefail
echo "[tts-net] node=$(hostname) job=${PBS_JOBID:-local}"
source "$(dirname "${BASH_SOURCE[0]}")/setup_proxy.sh"
echo "[tts-net] HF_HOME=${HF_HOME:-<default>} HF_HUB_CACHE=${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-<default>}}"
case "${HF_HUB_OFFLINE:-0}" in
    1|[Oo][Nn]|[Yy][Ee][Ss]|[Tt][Rr][Uu][Ee])
        echo "[tts-net] offline mode: skipping Hub probe; all model files must be cached"
        exit 0
        ;;
esac
if ! command -v curl >/dev/null 2>&1; then
    echo "[tts-net] WARNING: curl unavailable; Hub reachability was not checked" >&2
    exit 0
fi
# Bound retries before launching four GPU workers. Never print proxy credentials.
if curl --silent --show-error --fail --head --output /dev/null \
    --connect-timeout 10 --max-time 30 "${HF_ENDPOINT:-https://huggingface.co}"; then
    echo "[tts-net] Hub reachable (model downloads are checked during model loading)"
else
    result=$?
    case "$result" in
        5) echo "ERROR: compute node cannot resolve the configured proxy hostname." >&2 ;;
        6) echo "ERROR: compute node cannot resolve the Hub hostname on this route." >&2 ;;
        7) echo "ERROR: connection to the proxy or Hub was refused/unreachable." >&2 ;;
        28) echo "ERROR: proxy/Hub connection timed out within the 30s probe." >&2 ;;
        *) echo "ERROR: Hub probe failed (curl exit $result); see the error above." >&2 ;;
    esac
    echo "TTS workers were not started. Check this node's DNS and submission proxy settings." >&2
    exit "$result"
fi
