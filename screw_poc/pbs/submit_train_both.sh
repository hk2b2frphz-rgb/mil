#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

command -v qsub >/dev/null 2>&1 || {
    echo "ERROR: qsub is not available on PATH." >&2
    exit 1
}

TTS_RUN_DIR="${TTS_RUN_DIR:-$REPO_ROOT/screw_poc/artifacts/tts_1000/merged}"
DEPENDENCY_JOB_ID="${1:-}"
qsub_args=(-V -v "TTS_RUN_DIR=$TTS_RUN_DIR")
if [[ -n "$DEPENDENCY_JOB_ID" ]]; then
    qsub_args+=(-W "depend=afterok:${DEPENDENCY_JOB_ID}")
fi

lora_job="$(qsub "${qsub_args[@]}" screw_poc/pbs/run_train.pbs)"
full_job="$(qsub "${qsub_args[@]}" screw_poc/pbs/run_train_full.pbs)"

echo "dataset:      $TTS_RUN_DIR"
echo "LoRA job:     $lora_job"
echo "Full-FT job:  $full_job"
if [[ -n "$DEPENDENCY_JOB_ID" ]]; then
    echo "dependency:   afterok:$DEPENDENCY_JOB_ID"
fi
