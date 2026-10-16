#!/bin/bash
set -euo pipefail

# Only needed in Claude Code on the web / cloud sessions.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

pip install --quiet --user vastai

echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$CLAUDE_ENV_FILE"

# VAST_API_KEY is set as an environment variable on the cloud environment
# (claude.ai/code -> environment settings), not committed to the repo.
if [ -n "${VAST_API_KEY:-}" ]; then
  "$HOME/.local/bin/vastai" set api-key "$VAST_API_KEY" >/dev/null 2>&1 || true
fi
