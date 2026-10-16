#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$POC_ROOT/.." && pwd)"
TTS_RUN_DIR="${1:-$POC_ROOT/artifacts/tts}"
EXP_DIR="$POC_ROOT/experiments/lora_base_config"

if [[ ! -f "$TTS_RUN_DIR/training_set/synthetic_moshi_train.jsonl" ]]; then
    echo "ERROR: TTS training manifest not found under: $TTS_RUN_DIR" >&2
    echo "Run screw_poc/scripts/run_tts.sh first." >&2
    exit 1
fi

mkdir -p "$EXP_DIR"
cp "$POC_ROOT/config/moshi_lora.yaml" "$EXP_DIR/config.yaml"

# The shared launcher joins its experiment root with this relative path.  The
# '..' keeps all generated experiment data and checkpoints under screw_poc/.
cd "$REPO_ROOT"
REFRESH_EXP_DATA="${REFRESH_EXP_DATA:-1}" \
    bash scripts/run_experiment.sh \
    "../screw_poc/experiments/lora_base_config" \
    "$TTS_RUN_DIR"
